"""The decode / compose / write overlap (videowipe.pipeline_overlap).

Three groups: the queue and the stages in isolation (ordering, error
propagation, cancellation, the bound), the decision (``off``/``on``/``auto``),
and the whole inpainting loop with the overlap off and on, which must write the
same bytes in the same order.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import videowipe.timing  # noqa: F401  (binds the real Popen before tests replace it)
from videowipe import pipeline_overlap as po
from videowipe.errors import ProcessingCancelledError
from videowipe.inpainters import sttn
from videowipe.inpainters.base import InpaintJob

from test_blend_box_exact import (
    DEFAULT_FEATHER, E2E_FRAMES, E2E_GAP, E2E_H, E2E_W, FakePipe, ListReader,
    _static_masks, _temporal, make_clip, stub_predict,
)

TIMEOUT = 20


def stage_threads():
    return [t for t in threading.enumerate() if t.name.startswith("videowipe-")]


@pytest.fixture(autouse=True)
def no_stage_threads_left():
    yield
    deadline = time.time() + 5
    while stage_threads() and time.time() < deadline:
        time.sleep(0.01)
    assert stage_threads() == []


# ---------------------------------------------------------------------------
# 1. The bounded channel
# ---------------------------------------------------------------------------

def run_in_thread(fn):
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, box


def test_channel_is_first_in_first_out_and_drains_after_close():
    channel = po.Channel(8)
    for item in range(5):
        channel.put(item)
    channel.close()
    assert [channel.get() for _ in range(5)] == [0, 1, 2, 3, 4]
    with pytest.raises(po._Closed):
        channel.get()


def test_channel_never_holds_more_than_its_capacity():
    channel = po.Channel(3)
    produced = []

    def producer():
        for item in range(50):
            channel.put(item)
            produced.append(item)
        channel.close()

    thread, _ = run_in_thread(producer)
    got = []
    while True:
        time.sleep(0.001)  # a slow consumer keeps the producer pressed against the bound
        try:
            got.append(channel.get())
        except po._Closed:
            break
    thread.join(TIMEOUT)
    assert got == list(range(50))
    assert channel.high_water == 3
    assert channel.put_wait_s > 0


def test_a_full_channel_blocks_the_producer_until_the_consumer_takes_one():
    channel = po.Channel(2)
    channel.put("a")
    channel.put("b")
    thread, box = run_in_thread(lambda: channel.put("c"))
    time.sleep(0.1)
    assert thread.is_alive() and len(channel) == 2
    assert channel.get() == "a"
    thread.join(TIMEOUT)
    assert not thread.is_alive() and "error" not in box
    assert len(channel) == 2


def test_abort_wakes_a_blocked_producer_and_a_blocked_consumer():
    full = po.Channel(1)
    full.put(1)
    producer, produced = run_in_thread(lambda: full.put(2))
    empty = po.Channel(1)
    consumer, consumed = run_in_thread(empty.get)
    time.sleep(0.1)
    assert producer.is_alive() and consumer.is_alive()
    full.abort()
    empty.abort()
    producer.join(TIMEOUT)
    consumer.join(TIMEOUT)
    assert isinstance(produced["error"], po.Aborted)
    assert isinstance(consumed["error"], po.Aborted)
    assert len(full) == 0


# ---------------------------------------------------------------------------
# 2. The read-ahead reader
# ---------------------------------------------------------------------------

class CountingCapture:
    """Frames are small arrays whose value is their ordinal."""

    videowipe_prefetch_safe = True

    def __init__(self, count, fail_at=None, delay=0.0):
        self.count = count
        self.fail_at = fail_at
        self.delay = delay
        self.reads = 0
        self.released = False

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if self.fail_at is not None and self.reads == self.fail_at:
            raise RuntimeError("decoder blew up")
        if self.reads >= self.count:
            return False, None
        frame = np.full((2, 2, 3), self.reads, dtype=np.uint8)
        self.reads += 1
        return True, frame

    def grab(self):
        return self.read()[0]

    def release(self):
        self.released = True

    def isOpened(self):
        return True


def test_prefetch_reader_returns_the_same_frames_in_the_same_order():
    reader = po.PrefetchReader(CountingCapture(40), depth=4)
    try:
        got = []
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            got.append(int(frame[0, 0, 0]))
        assert got == list(range(40))
        # Like a capture at its end: it keeps saying so.
        assert reader.read() == (False, None)
        assert reader.grab() is False
        assert reader.read() == (False, None)
    finally:
        reader.release()


def test_prefetch_grab_advances_exactly_one_frame():
    reader = po.PrefetchReader(CountingCapture(10), depth=3)
    try:
        assert reader.grab() is True
        assert reader.grab() is True
        ok, frame = reader.read()
        assert ok and int(frame[0, 0, 0]) == 2
    finally:
        reader.release()


def test_prefetch_offers_grab_only_when_the_capture_does():
    class NoGrab:
        def read(self):
            return False, None

    plain = po.PrefetchReader(NoGrab(), depth=1)
    grabbing = po.PrefetchReader(CountingCapture(1), depth=1)
    try:
        assert not hasattr(plain, "grab")
        assert hasattr(grabbing, "grab")
    finally:
        plain.stop()
        grabbing.stop()


def test_prefetch_reads_ahead_only_as_far_as_its_queue_allows():
    capture = CountingCapture(1000)
    reader = po.PrefetchReader(capture, depth=4)
    try:
        deadline = time.time() + TIMEOUT
        while reader.channel.high_water < 4 and time.time() < deadline:
            time.sleep(0.005)
        time.sleep(0.1)  # let it run into the bound
        # depth queued plus the one frame it is holding while it waits.
        assert capture.reads <= 4 + 1
        for _ in range(6):
            reader.read()
        time.sleep(0.1)
        assert capture.reads <= 6 + 4 + 1
        assert reader.channel.high_water <= 4
    finally:
        reader.release()


def test_prefetch_decoder_error_arrives_after_the_frames_before_it():
    reader = po.PrefetchReader(CountingCapture(40, fail_at=5), depth=2)
    try:
        for expected in range(5):
            ok, frame = reader.read()
            assert ok and int(frame[0, 0, 0]) == expected
        with pytest.raises(RuntimeError, match="decoder blew up"):
            reader.read()
    finally:
        reader.release()


def test_prefetch_stop_unblocks_a_reader_that_is_waiting_on_a_full_queue():
    reader = po.PrefetchReader(CountingCapture(1000), depth=2)
    time.sleep(0.05)
    started = time.time()
    reader.stop()
    assert time.time() - started < 5
    assert not stage_threads()


def test_prefetch_releases_only_a_capture_it_owns():
    owned, borrowed = CountingCapture(3), CountingCapture(3)
    po.PrefetchReader(owned, depth=2, owns=True).release()
    po.PrefetchReader(borrowed, depth=2, owns=False).release()
    assert owned.released and not borrowed.released


# ---------------------------------------------------------------------------
# 3. The overlap: ordering, errors, cancellation
# ---------------------------------------------------------------------------

def make_overlap(depth, compose=None, write=None, frames=None):
    written = []
    names = []

    def default_compose(index, payload):
        names.append(threading.current_thread().name)
        return ("composed", index, payload)

    def default_write(stream, frame):
        written.append((stream, frame))

    overlap = po.Overlap(
        po.Plan(True, depth, "test"),
        CountingCapture(frames) if frames is not None else None,
        None,
        compose or default_compose,
        write or default_write,
    )
    return overlap, written, names


def test_frames_are_composed_and_written_in_submission_order_on_the_writer_thread():
    overlap, written, names = make_overlap(3)
    try:
        for index in range(60):
            overlap.write("sink", overlap.compose(index, f"p{index}"))
        overlap.finish()
    finally:
        overlap.abort()
    assert written == [("sink", ("composed", i, f"p{i}")) for i in range(60)]
    assert set(names) == {"videowipe-writer"}
    assert overlap.writer.frames == 60


def test_the_writer_queue_is_bounded_and_the_producer_waits_for_a_slow_writer():
    gate = threading.Event()
    written = []

    def slow_write(stream, frame):
        gate.wait(TIMEOUT)
        written.append(frame)

    overlap, _, _ = make_overlap(2, write=slow_write)
    try:
        submitted = []

        def producer():
            for index in range(10):
                overlap.write("s", overlap.compose(index, None))
                submitted.append(index)

        thread, _box = run_in_thread(producer)
        time.sleep(0.3)
        # queue (2) + the frame the writer holds (1) + at most the one being queued
        assert len(submitted) <= 2 + 1 + 1
        assert overlap.writer.channel.high_water <= 2
        gate.set()
        thread.join(TIMEOUT)
        overlap.finish()
    finally:
        gate.set()
        overlap.abort()
    assert len(written) == 10
    assert overlap.writer.channel.high_water <= 2


def test_a_compose_error_reaches_the_main_thread_and_does_not_hang_a_blocked_producer():
    def compose(index, payload):
        if index == 3:
            raise ValueError("compose failed")
        return index

    overlap, written, _ = make_overlap(1, compose=compose)
    try:
        with pytest.raises(ValueError, match="compose failed"):
            for index in range(100):
                overlap.write("s", overlap.compose(index, None))
                time.sleep(0.005)
        assert written == [("s", 0), ("s", 1), ("s", 2)]
    finally:
        overlap.abort()


def test_a_blocked_producer_gets_the_stage_error_not_a_bare_abort():
    gate = threading.Event()

    def compose(index, payload):
        gate.wait(TIMEOUT)
        raise ValueError("compose failed")

    overlap, _, _ = make_overlap(1, compose=compose)
    try:
        overlap.write("s", overlap.compose(0, None))  # the writer takes it and waits in compose
        deadline = time.time() + TIMEOUT
        while len(overlap.writer.channel) and time.time() < deadline:
            time.sleep(0.005)
        overlap.write("s", overlap.compose(1, None))  # fills the one-frame queue
        thread, box = run_in_thread(lambda: overlap.write("s", overlap.compose(2, None)))
        time.sleep(0.1)
        assert thread.is_alive()  # blocked on the full queue
        gate.set()
        thread.join(TIMEOUT)
        assert isinstance(box.get("error"), ValueError)
        assert "compose failed" in str(box["error"])
    finally:
        gate.set()
        overlap.abort()


def test_a_write_error_is_raised_by_finish_when_nothing_else_is_submitted():
    def write(stream, frame):
        raise BrokenPipeError("ffmpeg went away")

    overlap, _, _ = make_overlap(4, write=write)
    try:
        overlap.write("s", overlap.compose(0, None))
        with pytest.raises(BrokenPipeError):
            overlap.finish()
    finally:
        overlap.abort()


def test_a_stage_error_surfaces_at_the_next_frame_read():
    def write(stream, frame):
        raise OSError("disk full")

    overlap, _, _ = make_overlap(4, write=write, frames=50)
    try:
        overlap.write("s", overlap.compose(0, None))
        with pytest.raises(OSError, match="disk full"):
            deadline = time.time() + TIMEOUT
            while time.time() < deadline:
                overlap.reader.read()
                time.sleep(0.005)
    finally:
        overlap.abort()


def test_abort_discards_queued_frames_without_writing_them():
    gate = threading.Event()
    written = []

    def blocking_write(stream, frame):
        gate.wait(TIMEOUT)
        written.append(frame)

    overlap, _, _ = make_overlap(8, write=blocking_write)
    for index in range(6):
        overlap.write("s", overlap.compose(index, None))
    time.sleep(0.1)
    thread, _ = run_in_thread(overlap.abort)  # the writer is stuck inside its first write
    time.sleep(0.1)
    gate.set()
    thread.join(TIMEOUT)
    assert not thread.is_alive()
    assert len(written) <= 1  # only the frame that was already being written


def test_abort_is_prompt_and_idempotent_with_everything_blocked():
    overlap, _, _ = make_overlap(1, frames=1000)
    overlap.write("s", overlap.compose(0, None))
    started = time.time()
    overlap.abort()
    overlap.abort()
    assert time.time() - started < 5


def test_off_hands_back_the_original_functions_and_readers():
    reader, output = CountingCapture(1), CountingCapture(1)
    compose, write = object(), object()
    job = SimpleNamespace(gap=25)
    off = po.attach(job, reader, output, compose, write, frame_bytes=1, mode="off")
    assert off.reader is reader and off.output_reader is output
    assert off.compose is compose and off.write is write
    assert off.metrics() == {}
    off.finish()
    off.abort()
    assert not stage_threads()


# ---------------------------------------------------------------------------
# 4. The decision
# ---------------------------------------------------------------------------

GIB = 1024 ** 3
FRAME_1080P = 1920 * 1080 * 3


def test_mode_values_and_unknown_values(monkeypatch):
    for value, expected in [("off", "off"), ("OFF", "off"), ("0", "off"), ("false", "off"),
                            ("on", "on"), ("1", "on"), ("auto", "auto"), ("", "auto"),
                            ("whatever", "auto")]:
        monkeypatch.setenv(po.ENV_MODE, value)
        assert po.requested_mode() == expected
    monkeypatch.delenv(po.ENV_MODE)
    assert po.requested_mode() == "auto"


def test_off_is_never_overridden_by_the_machine():
    assert not po.decide(FRAME_1080P, 25, 64 * GIB, cpus=64, mode="off").enabled


def test_auto_follows_the_cores_and_memory_the_machine_reports():
    assert po.decide(FRAME_1080P, 25, 16 * GIB, cpus=4, mode="auto").enabled
    assert po.decide(FRAME_1080P, 25, 16 * GIB, cpus=12, mode="auto").enabled
    assert not po.decide(FRAME_1080P, 25, 16 * GIB, cpus=2, mode="auto").enabled
    assert not po.decide(FRAME_1080P, 25, 16 * GIB, cpus=1, mode="auto").enabled
    starved = po.decide(FRAME_1080P, 25, 64 * 1024 ** 2, cpus=8, mode="auto")
    assert not starved.enabled and "memory" in starved.reason


def test_on_forces_the_overlap_even_on_one_core_and_little_memory():
    plan = po.decide(FRAME_1080P, 25, 1024, cpus=1, mode="on")
    assert plan.enabled and plan.depth == 1


def test_queue_depth_is_a_segment_scaled_down_to_the_memory_budget(monkeypatch):
    monkeypatch.delenv(po.ENV_DEPTH, raising=False)
    assert po.queue_depth(FRAME_1080P, 25, 16 * GIB) == 25
    assert po.queue_depth(FRAME_1080P, 500, 16 * GIB) == po.DEFAULT_DEPTH_CAP
    assert po.queue_depth(FRAME_1080P, 1, 16 * GIB) == 1
    # 4K on a 4 GiB machine: a quarter-gigabyte budget is only a few frames
    four_k = 3840 * 2160 * 3
    depth = po.queue_depth(four_k, 25, 4 * GIB)
    assert 1 <= depth < 25
    assert 3 * depth * four_k <= (4 * GIB) // po.MEMORY_FRACTION
    # more memory never lowers it
    assert po.queue_depth(four_k, 25, 64 * GIB) >= depth
    # no figure: the default
    assert po.queue_depth(FRAME_1080P, 25, 0) == 25


def test_queue_depth_override(monkeypatch):
    monkeypatch.setenv(po.ENV_DEPTH, "3")
    assert po.queue_depth(FRAME_1080P, 25, 16 * GIB) == 3
    monkeypatch.setenv(po.ENV_DEPTH, "junk")
    assert po.queue_depth(FRAME_1080P, 25, 16 * GIB) == 25


# ---------------------------------------------------------------------------
# 5. The whole inpainting loop, off against on
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("overlapclip") / "clip.avi"
    return str(path), make_clip(path)


def run(tmp_path, monkeypatch, frames, video, mode, *, static_mask, frame_mask=None,
        dual=False, trial=None, depth=None, progress=None, phase_progress=None,
        reader=None, write_frame=None, metrics=None):
    monkeypatch.setenv(po.ENV_MODE, mode)
    if depth is None:
        monkeypatch.delenv(po.ENV_DEPTH, raising=False)
    else:
        monkeypatch.setenv(po.ENV_DEPTH, str(depth))
    pipe = FakePipe()
    monkeypatch.setattr(sttn.subprocess, "Popen", lambda *a, **k: pipe)
    monkeypatch.setattr(sttn, "_process_segment", lambda fr, *args: stub_predict(fr, None))
    if write_frame is not None:
        monkeypatch.setattr(sttn, "_write_frame", write_frame)
    painter = sttn.STTNInpainter()
    painter.backend = SimpleNamespace(benchmark_metadata=lambda: {"device": "cpu", "weight_sha256": "t"})
    if reader is None:
        reader = sttn.cv2.VideoCapture(str(video)) if video else ListReader(frames)
    metrics = {} if metrics is None else metrics
    try:
        painter.inpaint(InpaintJob(
            video_path=str(video) if video else "", mask=static_mask, output_dir=str(tmp_path),
            fps=8, frame_count=len(frames), width=E2E_W, height=E2E_H, reader=reader,
            gap=E2E_GAP, frame_mask=frame_mask, dual=dual, trial_range=trial, metrics=metrics,
            progress=progress, phase_progress=phase_progress,
        ))
    finally:
        if hasattr(reader, "release"):
            reader.release()
    return pipe.stdin.writes, metrics


class SafeListReader(ListReader):
    """A fake that declares it can be read ahead on another thread."""
    videowipe_prefetch_safe = True


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("source", ["video", "fake_reader", "prefetched_fake_reader"])
@pytest.mark.parametrize("trial", [None, (3, 9)])
def test_static_mask_output_is_identical_with_the_overlap_off_and_on(
        tmp_path, monkeypatch, clip, dual, source, trial):
    video, frames = clip
    mask = _static_masks(DEFAULT_FEATHER)
    src = video if source == "video" else None
    kind = SafeListReader if source == "prefetched_fake_reader" else ListReader
    off, off_metrics = run(tmp_path, monkeypatch, frames, src, "off", static_mask=mask,
                           dual=dual, trial=trial, reader=None if src else kind(frames))
    on, on_metrics = run(tmp_path, monkeypatch, frames, src, "on", static_mask=mask,
                         dual=dual, trial=trial, reader=None if src else kind(frames))
    assert len(off) == (trial[1] - trial[0] if trial else E2E_FRAMES)
    assert on == off
    assert "pipeline_overlap" not in off_metrics
    assert on_metrics["pipeline_overlap"] == 1
    # only a real capture, or a reader that declares itself safe, is read ahead
    assert ("overlap_reader_high_water" in on_metrics) == (source != "fake_reader")
    assert on_metrics["frames_written"] == off_metrics["frames_written"]
    assert on_metrics["frames_read"] == off_metrics["frames_read"]


@pytest.mark.parametrize("kind", ["stable", "stable3d", "unmarked"])
@pytest.mark.parametrize("depth", [1, 2, None])
def test_temporal_mask_output_is_identical_with_the_overlap_off_and_on(
        tmp_path, monkeypatch, clip, kind, depth):
    video, frames = clip
    static = _static_masks(DEFAULT_FEATHER)
    off, _ = run(tmp_path, monkeypatch, frames, video, "off", static_mask=static,
                 frame_mask=_temporal(DEFAULT_FEATHER, kind != "unmarked", kind))
    on, _ = run(tmp_path, monkeypatch, frames, video, "on", static_mask=static, depth=depth,
                frame_mask=_temporal(DEFAULT_FEATHER, kind != "unmarked", kind))
    assert len(off) == E2E_FRAMES
    assert on == off


def test_progress_calls_are_the_same_with_the_overlap_on(tmp_path, monkeypatch, clip):
    video, frames = clip
    mask = _static_masks(0)
    calls = {}
    for mode in ("off", "on"):
        seen = []
        run(tmp_path, monkeypatch, frames, video, mode, static_mask=mask,
            phase_progress=lambda *a, _s=seen: _s.append(a))
        calls[mode] = seen
    assert calls["on"] == calls["off"]
    assert calls["off"][-1][0] == "encode"


def test_the_queues_stay_within_the_stated_bound_in_a_real_run(tmp_path, monkeypatch, clip):
    video, frames = clip
    metrics = {}

    def slow_write(stream, frame):
        time.sleep(0.01)
        stream.write(memoryview(frame).cast("B") if frame.flags.c_contiguous else frame.tobytes())

    run(tmp_path, monkeypatch, frames, video, "on", static_mask=_static_masks(0), depth=2,
        write_frame=slow_write, metrics=metrics)
    assert metrics["pipeline_depth"] == 2
    for key in ("overlap_writer_high_water", "overlap_reader_high_water",
                "overlap_output_reader_high_water"):
        assert 1 <= metrics[key] <= 2


def test_a_write_failure_stops_the_run_with_that_error(tmp_path, monkeypatch, clip):
    video, frames = clip
    count = {"n": 0}

    def failing_write(stream, frame):
        count["n"] += 1
        if count["n"] == 4:
            raise BrokenPipeError("ffmpeg exited")
        stream.write(frame.tobytes())

    with pytest.raises(BrokenPipeError, match="ffmpeg exited"):
        run(tmp_path, monkeypatch, frames, video, "on", static_mask=_static_masks(0),
            write_frame=failing_write)
    assert count["n"] == 4  # nothing was written after the failure


@pytest.mark.parametrize("safe", [False, True])
def test_a_decode_failure_in_the_source_reader_stops_the_run(tmp_path, monkeypatch, clip, safe):
    video, frames = clip

    class BrokenReader(SafeListReader if safe else ListReader):
        def read(self):
            if self.index == 6:
                raise RuntimeError("reader exploded")
            return super().read()

    with pytest.raises(RuntimeError, match="reader exploded"):
        run(tmp_path, monkeypatch, frames, None, "on", static_mask=_static_masks(0),
            reader=BrokenReader(frames))


@pytest.mark.parametrize("safe", [False, True])
def test_a_short_source_reports_the_same_error_as_before(tmp_path, monkeypatch, clip, safe):
    video, frames = clip
    kind = SafeListReader if safe else ListReader
    messages = {}
    for mode in ("off", "on"):
        with pytest.raises(ValueError) as caught:
            run(tmp_path, monkeypatch, frames, None, mode, static_mask=_static_masks(0),
                reader=kind(frames[:6]))
        messages[mode] = str(caught.value)
    assert messages["on"] == messages["off"]


def test_cancellation_from_a_progress_callback_aborts_promptly(tmp_path, monkeypatch, clip):
    video, frames = clip
    calls = {"n": 0}

    def cancelling(*args):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ProcessingCancelledError("Video cleanup was cancelled")

    started = time.time()
    with pytest.raises(ProcessingCancelledError):
        run(tmp_path, monkeypatch, frames, video, "on", static_mask=_static_masks(0),
            progress=cancelling)
    assert time.time() - started < 20
    assert calls["n"] == 2  # no segment was started after the cancellation


def test_cancellation_while_the_writer_is_busy_drops_the_queue_instead_of_draining_it(
        tmp_path, monkeypatch, clip):
    video, frames = clip
    gate = threading.Event()
    written = []

    def stuck_write(stream, frame):
        gate.wait(60)  # a pipe whose reader has stopped taking frames
        written.append(1)

    calls = {"n": 0}

    def cancelling(*args):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProcessingCancelledError("cancelled")

    try:
        with pytest.raises(ProcessingCancelledError):
            run(tmp_path, monkeypatch, frames, video, "on", static_mask=_static_masks(0), depth=8,
                write_frame=stuck_write, progress=cancelling)
        assert written == []  # the run ended while the first write was still stuck
    finally:
        gate.set()
    deadline = time.time() + TIMEOUT
    while stage_threads() and time.time() < deadline:
        time.sleep(0.01)
    # the one write that was under way finishes; the queued frames never are
    assert len(written) <= 1
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 6. The mask cache is shared by the main thread and the writer thread
# ---------------------------------------------------------------------------

def test_mask_cache_survives_two_threads_hammering_it():
    """Mask evaluation (main thread) and composition (writer thread) both use
    the cache. With tiny limits every call evicts, which is where an unguarded
    cache breaks (dict changed size during iteration, lost accounting)."""
    modes = [(0, 60), (60, 120)]
    rng = np.random.RandomState(5)
    masks = []
    for _ in range(12):
        alpha = np.zeros((120, 200, 3), np.float32)
        y, x = rng.randint(0, 50), rng.randint(0, 100)
        alpha[y:y + 40, x:x + 60] = 1.0
        alpha.setflags(write=False)
        masks.append(alpha)
    one_pair = masks[0][:60].astype(np.float32).nbytes * 2
    cache = sttn._MaskCache(modes, max_entries=2, budget=one_pair)
    errors = []
    stop = threading.Event()
    sys_interval = __import__("sys").getswitchinterval()
    __import__("sys").setswitchinterval(1e-6)

    def main_side():  # mask evaluation
        try:
            for round_ in range(400):
                cache.get(masks[round_ % len(masks)])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            stop.set()

    def writer_side():  # composition
        try:
            round_ = 0
            while not stop.is_set():
                facts = cache.get(masks[(round_ * 5) % len(masks)])
                for k in facts.active:
                    alpha, remainder = facts.float_alpha(k)
                    box = facts.boxes[k]
                    expected = facts.alpha[box[0]:box[1], box[2]:box[3]].astype(np.float32)
                    assert np.array_equal(alpha, expected)
                    assert np.array_equal(remainder, 1.0 - expected)
                round_ += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    try:
        threads = [threading.Thread(target=main_side), threading.Thread(target=writer_side)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(TIMEOUT * 3)
    finally:
        __import__("sys").setswitchinterval(sys_interval)
    assert errors == []
    assert cache.derived_bytes >= 0
    assert len(cache) <= 2


def test_mask_cache_operations_exclude_each_other():
    """Holding the cache's lock keeps every operation that changes it out."""
    modes = [(0, 60), (60, 120)]
    alpha = np.zeros((120, 200, 3), np.float32)
    alpha[10:40, 20:80] = 1.0
    other = alpha.copy()
    cache = sttn._MaskCache(modes, max_entries=4, budget=10 ** 7)
    facts = cache.get(alpha)
    k = facts.active[0]
    operations = {
        "get": lambda: cache.get(other),
        "admit": lambda: cache.admit(facts, 8),
        "float_alpha": lambda: facts.float_alpha(k),
    }
    for name, operation in operations.items():
        with cache.lock:
            thread, box = run_in_thread(operation)
            time.sleep(0.2)
            assert thread.is_alive(), f"{name} ran while the cache was locked"
        thread.join(TIMEOUT)
        assert not thread.is_alive() and "error" not in box
