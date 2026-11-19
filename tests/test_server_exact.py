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
