"""Equality tests for the exact server performance changes (thumbnails, masks, hashing)."""
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from videowipe.server import app as server_app
from videowipe.server import jobs

# tests/ is not a package; pytest puts it on sys.path, so reuse the existing fixtures.
from test_server import FakeEngine, _post_video, _wait_for_state, client  # noqa: F401


_REAL_CAPTURE = cv2.VideoCapture


def _write_distinct_video(path, width=96, height=64, frames=12):
    """A video whose every frame differs, so a wrong frame index changes the bytes."""
    rng = np.random.default_rng(7)
    chunks = []
    for index in range(frames):
        frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        frame[50:60, 10 + index:40 + index] = 255
        frame[:, : index + 1] = (index * 20) % 255
        chunks.append(frame.tobytes())
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{width}x{height}", "-pix_fmt", "bgr24", "-r", "4", "-i", "pipe:0",
         "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path)],
        input=b"".join(chunks), check=True)


def _original_frame_bytes(source, frame_index):
    """Verbatim copy of the original per-request decode and encode."""
    reader = _REAL_CAPTURE(str(source))
    try:
        for _ in range(frame_index):
            assert reader.grab()
        ok, frame = reader.read()
        assert ok
    finally:
        reader.release()
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


class _CountingCapture:
    """Delegating wrapper (subclassing cv2.VideoCapture is not safe)."""
    opened = grabs = retrieves = reads = seeks = 0

    def __init__(self, *args):
        type(self).opened += 1
        self._inner = _REAL_CAPTURE(*args)

    def grab(self):
        type(self).grabs += 1
        return self._inner.grab()

    def retrieve(self, *args):
        type(self).retrieves += 1
        return self._inner.retrieve(*args)

    def read(self, *args):
        type(self).reads += 1
        return self._inner.read(*args)

    def set(self, *args):
        type(self).seeks += 1
        return self._inner.set(*args)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture()
def counting(monkeypatch):
    for name in ("opened", "grabs", "retrieves", "reads", "seeks"):
        monkeypatch.setattr(_CountingCapture, name, 0)
    monkeypatch.setattr(server_app.cv2, "VideoCapture", _CountingCapture)
    return _CountingCapture


@pytest.fixture()
def evidence_job(client, tmp_path):
    test_client, _ = client
    video = tmp_path / "distinct.mp4"
    _write_distinct_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    _wait_for_state(test_client, job_id, "preview_ready")
    job = jobs.get_job(job_id)
    directory = server_app._plan_directory(job)
    path = directory / "clean_candidates.json"
    data = json.loads(path.read_text())
    data["candidates"][0]["presence_frames"] = [5, 6, 9]
    data["candidates"][1]["presence_frames"] = [3, 4]
    path.write_text(json.dumps(data))
    return test_client, job, Path(job.video_path)


EVIDENCE = [0, 3, 5]


def test_thumbnails_are_identical_and_share_one_sequential_pass(evidence_job, counting):
    test_client, job, source = evidence_job
    expected = {n: _original_frame_bytes(source, n) for n in range(12)}
    assert len({v for v in expected.values()}) == 12
    first = test_client.get(f"/jobs/{job.id}/frame?frame_index=3")
    assert first.status_code == 200 and first.content == expected[3]
    # One capture, one ascending pass to the last target, retrieve only at targets.
    assert (counting.opened, counting.grabs, counting.retrieves, counting.reads, counting.seeks) == (1, 6, 3, 0, 0)
    for n in EVIDENCE:
        assert (Path(job.output_dir) / f"evidence-{n}.jpg").read_bytes() == expected[n]
    for n in EVIDENCE:
        response = test_client.get(f"/jobs/{job.id}/frame?frame_index={n}")
        assert response.status_code == 200 and response.content == expected[n]
        assert response.headers["content-type"] == "image/jpeg"
    assert counting.opened == 1  # every later request is a cache hit


def test_free_scrub_frames_are_exact_and_never_cached(evidence_job, counting):
    test_client, job, source = evidence_job
    for n in (1, 2, 4, 7, 11):
        response = test_client.get(f"/jobs/{job.id}/frame?frame_index={n}")
        assert response.status_code == 200 and response.content == _original_frame_bytes(source, n)
    assert not list(Path(job.output_dir).glob("evidence-*"))
    assert counting.opened == 5 and counting.reads == 5 and counting.retrieves == 0
    assert counting.grabs == sum([1, 2, 4, 7, 11])


def test_cached_thumbnail_is_served_without_parsing_plan_or_candidates(evidence_job, monkeypatch):
    test_client, job, source = evidence_job
    assert test_client.get(f"/jobs/{job.id}/frame?frame_index=5").status_code == 200

    def forbidden(*args, **kwargs):
        raise AssertionError("plan or candidates parsed for a cached evidence frame")

    monkeypatch.setattr(server_app, "load_wipe_plan", forbidden)
    monkeypatch.setattr(server_app, "_load_candidates", forbidden)
    response = test_client.get(f"/jobs/{job.id}/frame?frame_index=3")
    assert response.status_code == 200 and response.content == _original_frame_bytes(source, 3)


def test_stale_cache_for_a_non_evidence_frame_is_not_trusted(evidence_job, counting):
    test_client, job, source = evidence_job
    assert test_client.get(f"/jobs/{job.id}/frame?frame_index=0").status_code == 200
    stale = Path(job.output_dir) / "evidence-2.jpg"
    stale.write_bytes(b"not a real frame")
    response = test_client.get(f"/jobs/{job.id}/frame?frame_index=2")
    assert response.status_code == 200 and response.content == _original_frame_bytes(source, 2)


def test_changed_candidates_invalidate_the_remembered_evidence(evidence_job):
    test_client, job, source = evidence_job
    assert test_client.get(f"/jobs/{job.id}/frame?frame_index=5").status_code == 200
    path = server_app._plan_directory(job) / "clean_candidates.json"
    data = json.loads(path.read_text())
    data["candidates"][0]["presence_frames"] = [8]
    path.write_text(json.dumps(data))
    stale = Path(job.output_dir) / "evidence-5.jpg"
    stale.write_bytes(b"stale")
    response = test_client.get(f"/jobs/{job.id}/frame?frame_index=5")
    assert response.status_code == 200 and response.content == _original_frame_bytes(source, 5)


def test_thumbnail_error_statuses_are_unchanged(evidence_job, tmp_path):
    test_client, job, source = evidence_job
    assert test_client.get(f"/jobs/{job.id}/frame?frame_index=12").status_code == 400
    assert test_client.get(f"/jobs/{job.id}/frame?frame_index=99999").status_code == 400
    assert test_client.get(f"/jobs/{job.id}/frame?frame_index=-1").status_code == 422
    assert test_client.get("/jobs/missing/frame?frame_index=0").status_code == 404
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"secret")
    link = Path(job.output_dir) / "evidence-3.jpg"
    link.symlink_to(outside)
    assert test_client.get(f"/jobs/{job.id}/frame?frame_index=3").status_code == 409
    link.unlink()
    link.symlink_to(tmp_path / "missing.jpg")  # dangling: treated as uncached, refused when writing
    assert test_client.get(f"/jobs/{job.id}/frame?frame_index=3").status_code == 409
    link.unlink()
    assert test_client.get(f"/jobs/{job.id}/frame?frame_index=3").status_code == 200


def test_decode_failure_before_the_requested_frame_is_422(evidence_job, monkeypatch):
    test_client, job, source = evidence_job

    class Failing(_CountingCapture):
        def grab(self):
            return False if type(self).grabs >= 2 else super().grab()

    monkeypatch.setattr(Failing, "grabs", 0, raising=False)

    monkeypatch.setattr(server_app.cv2, "VideoCapture", Failing)
    assert test_client.get(f"/jobs/{job.id}/frame?frame_index=3").status_code == 422
    # Frame 0 decoded before the failure and is a legitimate evidence frame; the rest are not cached.
    assert [p.name for p in Path(job.output_dir).glob("evidence-*")] == ["evidence-0.jpg"]


# ── P1-4: mask endpoint ──────────────────────────────────────────────────────

import threading  # noqa: E402

from fastapi import HTTPException  # noqa: E402
from fastapi.responses import Response  # noqa: E402

from test_server import _write_test_video  # noqa: E402
from videowipe.plan import execution_masks, load_wipe_plan  # noqa: E402
from videowipe.server.app import ConfirmRequest  # noqa: E402


def _original_mask(job, frame_index, revision):
    """Verbatim copy of the original actual_mask body (response, or status and detail)."""
    try:
        with job.lock:
            if revision != job.review_revision or not job.final_plan_ready:
                raise HTTPException(status_code=409, detail='mask revision is not ready')
            if job.reviewed_plan_path and job.compiled_review_revision == revision:
                plan = load_wipe_plan(str(server_app._job_file(job, job.reviewed_plan_path)))
            else:
                plan, _ = server_app._reviewed_plan(job, ConfirmRequest(**(job.overrides or {})), allow_empty=True)
                machine = load_wipe_plan(str(server_app._plan_path(job)))
                if {t.id for t in plan.remove_tracks} - {t.id for t in machine.remove_tracks}:
                    raise HTTPException(status_code=409, detail='新增目标需先试擦确认出现时间')
            if frame_index >= plan.source.frame_count:
                raise HTTPException(status_code=400, detail='frame outside source')
            union, temporal = execution_masks(plan, server_app._get_engine()._task_impl.feather_radius)
            alpha = temporal(frame_index) if temporal else union[:, :, 0]
            ok, png = cv2.imencode('.png', np.rint(alpha * 255).astype(np.uint8))
            if not ok:
                raise HTTPException(status_code=500, detail='mask encoding failed')
            return Response(png.tobytes(), media_type='image/png', headers={'Cache-Control': 'no-store', 'X-Has-Removal': str(bool(union.any())).lower()})
    except HTTPException as exc:
        return exc.status_code, exc.detail


def _compare_with_original(test_client, job, revision, frames=range(0, 10)):
    for frame in frames:
        expected = _original_mask(job, frame, revision)
        actual = test_client.get(f"/jobs/{job.id}/mask?frame_index={frame}&revision={revision}")
        if isinstance(expected, tuple):
            assert (actual.status_code, actual.json()["detail"]) == expected, frame
            continue
        assert actual.status_code == 200
        assert actual.content == expected.body, frame
        assert actual.headers["content-type"] == "image/png"
        assert actual.headers["cache-control"] == "no-store"
        assert actual.headers["x-has-removal"] == expected.headers["x-has-removal"]


@pytest.fixture()
def mask_job(client, tmp_path):
    test_client, fake = client
    video = tmp_path / "mask.mp4"
    _write_test_video(video)
    job_id = _post_video(test_client, video).json()["id"]
    ready = _wait_for_state(test_client, job_id, "preview_ready")
    return test_client, jobs.get_job(job_id), ready["review_revision"]


EDITS = {
    "selected_ids": ["c1"], "bbox_overrides": {},
    "segment_overrides": {"c1": [[4, 8], [0, 3], [2, 5]]},
    "protections": [{"id": "p_one", "bbox": [20, 45, 30, 63], "segments": [[0, 1], [7, 8]]}],
}


def test_mask_png_and_headers_match_original_across_revisions(mask_job):
    test_client, job, revision = mask_job
    _compare_with_original(test_client, job, revision)
    cases = [
        EDITS,
        {"selected_ids": ["c1"], "segment_overrides": {"c1": [[2, 5]]}},
        {"selected_ids": ["c1"], "bbox_overrides": {"c1": [12, 48, 60, 58]}},
        {"selected_ids": []},  # nothing to remove: X-Has-Removal is false
        {"selected_ids": ["c2"]},  # a target the machine plan did not remove: 409
        {"selected_ids": ["c1", "c2"], "bbox_overrides": {"c2": [4, 4, 28, 16]}},
    ]
    seen = set()
    for edits in cases:
        saved = test_client.patch(f"/jobs/{job.id}/review", json={"expected_revision": revision, "overrides": edits})
        assert saved.status_code == 200, saved.text
        revision = saved.json()["review_revision"]
        _compare_with_original(test_client, job, revision)
        expected = _original_mask(job, 0, revision)
        # The review response carries the same boolean the mask header would.
        if isinstance(expected, tuple):
            assert saved.json()["has_removal"] is None and expected[0] == 409
        else:
            assert saved.json()["has_removal"] == (expected.headers["x-has-removal"] == "true")
            seen.add(saved.json()["has_removal"])
        assert test_client.get(f"/jobs/{job.id}/mask?frame_index=0&revision={revision - 1}").status_code == 409
    assert seen == {True, False}


def test_mask_matches_original_for_the_reviewed_plan_branch(mask_job):
    test_client, job, revision = mask_job
    saved = test_client.patch(f"/jobs/{job.id}/review", json={"expected_revision": revision, "overrides": EDITS})
    revision = saved.json()["review_revision"]
    assert test_client.post(f"/jobs/{job.id}/confirm", json={**EDITS, "expected_revision": revision}).status_code == 200
    _wait_for_state(test_client, job.id, "done")
    assert job.reviewed_plan_path and job.compiled_review_revision == job.review_revision
    _compare_with_original(test_client, job, job.review_revision)


def test_mask_error_paths_match_original(mask_job, monkeypatch):
    test_client, job, revision = mask_job
    _compare_with_original(test_client, job, revision, frames=[8, 9, 5000])
    assert test_client.get(f"/jobs/{job.id}/mask?frame_index=8&revision={revision}").status_code == 400
    assert test_client.get("/jobs/missing/mask?frame_index=0&revision=0").status_code == 404
    job.final_plan_ready = False
    assert test_client.get(f"/jobs/{job.id}/mask?frame_index=0&revision={revision}").json()["detail"] == "mask revision is not ready"
    job.final_plan_ready = True
    monkeypatch.setattr(server_app.cv2, "imencode", lambda *args: (False, None))
    failed = test_client.get(f"/jobs/{job.id}/mask?frame_index=0&revision={revision}")
    assert (failed.status_code, failed.json()["detail"]) == (500, "mask encoding failed")


def test_mask_plan_and_projection_are_cached_per_revision(mask_job, monkeypatch):
    test_client, job, revision = mask_job
    calls = {"plans": 0, "masks": 0}
    real_load, real_masks = server_app.load_wipe_plan, server_app.execution_masks
    monkeypatch.setattr(server_app, "load_wipe_plan", lambda *a, **k: calls.__setitem__("plans", calls["plans"] + 1) or real_load(*a, **k))
    monkeypatch.setattr(server_app, "execution_masks", lambda *a, **k: calls.__setitem__("masks", calls["masks"] + 1) or real_masks(*a, **k))
    for frame in range(8):
        assert test_client.get(f"/jobs/{job.id}/mask?frame_index={frame}&revision={revision}").status_code == 200
    assert calls == {"plans": 2, "masks": 1}  # compiled plan and machine plan, once
    saved = test_client.patch(f"/jobs/{job.id}/review", json={"expected_revision": revision, "overrides": EDITS})
    revision = saved.json()["review_revision"]
    assert calls["masks"] == 2  # the review response primed the new revision
    for frame in range(8):
        assert test_client.get(f"/jobs/{job.id}/mask?frame_index={frame}&revision={revision}").status_code == 200
    assert calls["masks"] == 2 and server_app._MASK_CACHE[job.id].key[0] == revision
    # A changed review is never answered from the old entry.
    again = test_client.patch(f"/jobs/{job.id}/review", json={"expected_revision": revision, "overrides": {"selected_ids": []}})
    assert again.json()["has_removal"] is False and calls["masks"] == 3
    assert len(server_app._MASK_CACHE) <= server_app._MASK_CACHE_LIMIT


def test_mask_cache_is_bounded_and_keeps_one_entry_per_job():
    server_app._MASK_CACHE.clear()
    for number in range(server_app._MASK_CACHE_LIMIT + 3):
        server_app._mask_entry(f"job-{number}", (number,))
    assert len(server_app._MASK_CACHE) == server_app._MASK_CACHE_LIMIT
    first = server_app._mask_entry("job-last", (1,))
    assert server_app._mask_entry("job-last", (1,)) is first
    assert server_app._mask_entry("job-last", (2,)) is not first
    assert list(server_app._MASK_CACHE).count("job-last") == 1
    server_app._MASK_CACHE.clear()


def test_mask_rendering_happens_outside_the_job_lock(mask_job, monkeypatch):
    test_client, job, revision = mask_job
    assert type(job.lock) is type(threading.RLock())  # re-entrant, so probe from another thread
    held = []

    def lock_is_free():
        result = []

        def probe():
            acquired = job.lock.acquire(blocking=False)
            result.append(acquired)
            if acquired:
                job.lock.release()

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()
        return result[0]

    real_encode, real_masks, real_load = cv2.imencode, server_app.execution_masks, server_app.load_wipe_plan
    monkeypatch.setattr(server_app.cv2, "imencode", lambda *a, **k: held.append(("encode", lock_is_free())) or real_encode(*a, **k))
    monkeypatch.setattr(server_app, "execution_masks", lambda *a, **k: held.append(("project", lock_is_free())) or real_masks(*a, **k))
    monkeypatch.setattr(server_app, "load_wipe_plan", lambda *a, **k: held.append(("load", lock_is_free())) or real_load(*a, **k))
    assert test_client.get(f"/jobs/{job.id}/mask?frame_index=1&revision={revision}").status_code == 200
    assert {name for name, _ in held} == {"load", "project", "encode"}
    assert all(free for _, free in held), held
    # The primed review response renders outside the lock too.
    held.clear()
    saved = test_client.patch(f"/jobs/{job.id}/review", json={"expected_revision": revision, "overrides": EDITS})
    # (_save_review itself still loads the plan under the lock, as before.)
    assert saved.status_code == 200 and ("project", True) in held
    assert all(free for name, free in held if name in {"project", "encode"})


@pytest.mark.skipif(not shutil.which("node"), reason="requires node")
def test_page_skips_mask_requests_when_overlay_is_off():
    html = server_app._web_index().read_text(encoding="utf-8")
    body = html.split("    async function refreshMask(){", 1)[1].split("    $('showMask').addEventListener", 1)[0]
    harness = r'''
const assert=require("node:assert/strict");
let currentJobId="j",finalPlanReady=true,saving=0,view="source",reviewRevision=3,maskEpoch=0,maskHasRemoval=true,knownRemoval=null;
let tracks=[{id:"c1"}],source={frame_count:10,fps:25},checked=false,fetches=[],renders=0,hasRemoval="false";
const busy=()=>false,position=()=>0.2,renderActions=()=>{renders++;};
const nodes={showMask:{get checked(){return checked;}},maskCanvas:{},maskState:{}};
const $=id=>nodes[id];
global.createImageBitmap=async()=>({close(){}});
global.fetch=async url=>{fetches.push(url);return {ok:true,headers:{get:()=>hasRemoval},blob:async()=>({})};};
async function refreshMask(){''' + body + r'''
(async()=>{
  // Unknown revision, overlay off: one request, then the answer is remembered.
  await refreshMask();
  assert.equal(fetches.length,1);assert.equal(maskHasRemoval,false);
  await refreshMask();await refreshMask();
  assert.equal(fetches.length,1);assert.equal(maskHasRemoval,false);assert.equal(nodes.maskState.textContent,"");
  // Value supplied by the review response: no request at all.
  knownRemoval={jobId:"j",revision:4,tracks,value:true};reviewRevision=4;maskHasRemoval=false;
  await refreshMask();
  assert.equal(fetches.length,1);assert.equal(maskHasRemoval,true);
  // A different revision, job or target list is not trusted.
  reviewRevision=5;await refreshMask();assert.equal(fetches.length,2);
  knownRemoval={jobId:"j",revision:5,tracks,value:true};tracks=[{id:"c1"}];await refreshMask();assert.equal(fetches.length,3);
  currentJobId="k";await refreshMask();assert.equal(fetches.length,4);
  // Overlay on: behaves as before and requests the mask every time.
  currentJobId="j";checked=true;knownRemoval={jobId:"j",revision:5,tracks,value:false};
  await refreshMask();await refreshMask();
  assert.equal(fetches.length,6);assert.ok(fetches.every(url=>url.includes("/mask?frame_index=")));
  // The same guards as before: no request while saving.
  saving=1;checked=false;knownRemoval=null;await refreshMask();assert.equal(fetches.length,6);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    subprocess.run(["node", "-e", harness], check=True)
