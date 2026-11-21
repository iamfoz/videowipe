"""The progress stream sends a snapshot only when it changes, and keeps the connection alive between changes."""
import asyncio
import json
import threading
import time

import pytest

from videowipe.server import app as server_app
from videowipe.server import jobs


FINAL_STATES = {"done", "error", "cancelled", "preview_ready", "interrupted"}


@pytest.fixture()
def job(tmp_path, monkeypatch):
    jobs.reset_jobs()
    monkeypatch.setattr(server_app, "SSE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(server_app, "SSE_HEARTBEAT_SECONDS", 0.1)
    job = jobs.Job(id="a" * 32, video_path="", output_dir=str(tmp_path), state="running", phase="inpaint")
    jobs.JOBS[job.id] = job
    yield job
    jobs.reset_jobs()


class Consumer:
    """Reads the stream on a thread, as a browser would, recording every chunk."""

    def __init__(self, job_id):
        self.chunks = []
        self.cond = threading.Condition()
        self.finished = False
        self.thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
        self.thread.start()

    def _run(self, job_id):
        async def consume():
            response = server_app.progress_sse(job_id)
            async for chunk in response.body_iterator:
                with self.cond:
                    self.chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
                    self.cond.notify_all()

        asyncio.run(consume())
        with self.cond:
            self.finished = True
            self.cond.notify_all()

    @property
    def events(self):
        return [json.loads(c[len("data: "):]) for c in self.chunks if c.startswith("data: ")]

    @property
    def comments(self):
        return [c for c in self.chunks if c.startswith(":")]

    def wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        with self.cond:
            while not predicate():
                remaining = deadline - time.monotonic()
                assert remaining > 0, f"timed out; chunks so far: {self.chunks!r}"
                self.cond.wait(remaining)


def _set(job, **changes):
    with job.lock:
        for key, value in changes.items():
            setattr(job, key, value)


def test_every_state_change_is_delivered_in_order_and_the_final_state_ends_the_stream(job):
    consumer = Consumer(job.id)
    expected = [json.loads(json.dumps(job.snapshot()))]
    consumer.wait_for(lambda: len(consumer.events) == 1)
    steps = [
        dict(phase="inpaint", progress=0.1, completed=10, total=100),
        dict(progress=0.4, completed=40),
        dict(phase="encode", progress=0.0, completed=0, total=1),
        dict(progress=0.9, completed=1),
        dict(remaining_seconds=[3.0, 4.0]),
        dict(state="done", phase="complete", progress=1.0, result_path="out.mp4"),
    ]
    for step in steps:
        # Wait for the previous change to arrive before making the next: nothing may be coalesced away.
        _set(job, **step)
        expected.append(json.loads(json.dumps(job.snapshot())))
        consumer.wait_for(lambda: len(consumer.events) == len(expected))
    consumer.wait_for(lambda: consumer.finished)
    assert consumer.events == expected
    assert consumer.events[-1]["state"] == "done"
    assert consumer.events[-1]["progress"] == 1.0
    # Never two identical snapshots in a row.
    assert all(a != b for a, b in zip(consumer.events, consumer.events[1:]))


def test_an_idle_job_sends_one_snapshot_then_only_keep_alive_comments(job):
    consumer = Consumer(job.id)
    consumer.wait_for(lambda: len(consumer.comments) >= 3)
    assert len(consumer.events) == 1
    assert all(c == ": keep-alive\n\n" for c in consumer.comments)
    # Comments carry no data line, so an EventSource never turns them into messages.
    assert not any(c.startswith("data:") for c in consumer.comments)
    _set(job, state="cancelled")
    consumer.wait_for(lambda: consumer.finished)
    assert [e["state"] for e in consumer.events] == ["running", "cancelled"]
    assert consumer.chunks[-1].startswith("data: ")


@pytest.mark.parametrize("state", sorted(FINAL_STATES))
def test_a_job_already_in_a_final_state_gets_that_snapshot_and_the_stream_closes(job, state):
    _set(job, state=state)
    consumer = Consumer(job.id)
    consumer.wait_for(lambda: consumer.finished)
    assert [e["state"] for e in consumer.events] == [state]
    assert consumer.comments == []


def test_a_change_after_a_long_quiet_spell_still_arrives(job, monkeypatch):
    monkeypatch.setattr(server_app, "SSE_HEARTBEAT_SECONDS", 0.05)
    consumer = Consumer(job.id)
    consumer.wait_for(lambda: len(consumer.comments) >= 2)
    _set(job, progress=0.5, completed=5, total=10)
    consumer.wait_for(lambda: len(consumer.events) == 2)
    assert consumer.events[1]["progress"] == 0.5
    _set(job, state="error", error="boom")
    consumer.wait_for(lambda: consumer.finished)
    assert consumer.events[-1]["error"] == "boom"


def test_default_timings_are_unchanged():
    assert server_app.SSE_POLL_SECONDS == 0.5
    assert server_app.SSE_HEARTBEAT_SECONDS == 15.0


def test_unknown_job_is_still_404():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as info:
        server_app.progress_sse("missing")
    assert info.value.status_code == 404
