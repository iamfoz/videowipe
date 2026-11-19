"""TorchBackend.cleanup releases device allocator caches."""
from types import SimpleNamespace

from videowipe.backends import TorchBackend


def _backend(device_type, torch):
    backend = object.__new__(TorchBackend)
    backend._torch = torch
    backend.device = SimpleNamespace(type=device_type)
    backend.model = object()
    return backend


def _fake_torch(calls, with_mps=True):
    torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: calls.append("cuda")))
    if with_mps:
        torch.mps = SimpleNamespace(empty_cache=lambda: calls.append("mps"))
    return torch


def test_cleanup_empties_mps_cache():
    calls = []
    backend = _backend("mps", _fake_torch(calls))
    backend.cleanup()
    assert calls == ["mps"] and backend.model is None


def test_cleanup_tolerates_torch_without_mps_empty_cache():
    calls = []
    backend = _backend("mps", _fake_torch(calls, with_mps=False))
    backend.cleanup()
    assert calls == [] and backend.model is None


def test_cleanup_cuda_and_cpu_unchanged():
    calls = []
    _backend("cuda", _fake_torch(calls)).cleanup()
    _backend("cpu", _fake_torch(calls)).cleanup()
    assert calls == ["cuda"]
