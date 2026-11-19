"""Precision selection for the PyTorch backend (VIDEOWIPE_PRECISION).

Apple's MPS cannot run here, so path selection is tested with fake torch and
device objects; the numerics are tested with real CPU torch where available.
"""
import contextlib
import logging
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from videowipe import backends
from videowipe.backends import (
    LABEL_CUDA_AUTOCAST, LABEL_FLOAT32, LABEL_MPS_AUTOCAST, LABEL_MPS_FALLBACK,
    LABEL_MPS_HALF, MODE_CUDA_AUTOCAST, MODE_FLOAT32, MODE_MPS_AUTOCAST, MODE_MPS_HALF,
    InpaintBackend, ONNXBackend, TorchBackend, parse_precision, wants_reduced_precision,
)


# --------------------------------------------------------------------------
# Setting parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("auto", "auto"), ("float32", "float32"), ("float16", "float16"),
    ("FLOAT16", "float16"), ("  float32 ", "float32"),
    ("", "auto"), ("half", "auto"), ("bfloat16", "auto"), ("fp16", "auto"), ("1", "auto"),
])
def test_parse_precision_values(value, expected):
    assert parse_precision(value) == expected


def test_parse_precision_reads_environment(monkeypatch):
    monkeypatch.delenv("VIDEOWIPE_PRECISION", raising=False)
    assert parse_precision() == "auto"
    monkeypatch.setenv("VIDEOWIPE_PRECISION", "float16")
    assert parse_precision() == "float16"
    monkeypatch.setenv("VIDEOWIPE_PRECISION", "nonsense")
    assert parse_precision() == "auto"
    # An explicit argument wins over the environment.
    assert parse_precision("float32") == "float32"


@pytest.mark.parametrize("setting,device,expected", [
    ("auto", "cuda", True), ("auto", "mps", False), ("auto", "cpu", False),
    ("float32", "cuda", False), ("float32", "mps", False), ("float32", "cpu", False),
    ("float16", "cuda", True), ("float16", "mps", True), ("float16", "cpu", False),
])
def test_wants_reduced_precision_matrix(setting, device, expected):
    assert wants_reduced_precision(setting, device) is expected


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeTensor:
    def __init__(self, array, device="cpu"):
        self.array = np.asarray(array)
        self.device = device

    dtype = property(lambda self: self.array.dtype)
    is_cuda = property(lambda self: self.device == "cuda")

    def to(self, *args, **kwargs):
        if args and isinstance(args[0], (str, FakeDevice)):
            self.device = args[0] if isinstance(args[0], str) else args[0].type
        return self

    def half(self):
        return FakeTensor(self.array.astype(np.float16), self.device)

    def float(self):
        return FakeTensor(self.array.astype(np.float32), self.device)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeDevice:
    def __init__(self, type):
        self.type = type

    def __str__(self):
        return self.type


class FakeTorch:
    """Just enough of torch to drive TorchBackend.transform and the detection."""

    float16 = np.float16
    channels_last = "channels_last"
    __version__ = "fake"

    def __init__(self, mps_autocast="works"):
        self.mps_autocast = mps_autocast  # "works", "missing" or "inert"
        self.autocast_calls = []
        self._active = []

    @contextlib.contextmanager
    def no_grad(self):
        yield

    def from_numpy(self, array):
        return FakeTensor(array)

    def ones(self, shape, device=None):
        return FakeTensor(np.ones(shape, dtype=np.float32))

    def matmul(self, a, b):
        reduced = self.mps_autocast == "works" and ("mps", np.float16) in self._active
        return FakeTensor(np.ones((2, 2), dtype=np.float16 if reduced else np.float32))

    @contextlib.contextmanager
    def autocast(self, device_type, dtype=None, enabled=True):
        self.autocast_calls.append((device_type, dtype, enabled))
        if device_type == "mps" and self.mps_autocast == "missing":
            raise RuntimeError("User specified an unsupported autocast device_type 'mps'")
        self._active.append((device_type, dtype))
        try:
            yield
        finally:
            self._active.pop()


class FakeModel:
    """Records how infer was reached; transformer is the part that is halved."""

    class Transformer:
        def __init__(self, half=False):
            self.is_half = half

        def half(self):
            return FakeModel.Transformer(half=True)

        def eval(self):
            return self

        def __deepcopy__(self, memo):
            return FakeModel.Transformer(self.is_half)

    _log = []

    def __init__(self):
        self.transformer = FakeModel.Transformer()
        FakeModel._log = []

    infer_calls = property(lambda self: FakeModel._log)

    def infer(self, feat):
        # Also reached unbound with a stand-in ``self`` that only has ``transformer``.
        FakeModel._log.append({"dtype": feat.dtype, "half_transformer": self.transformer.is_half})
        return FakeTensor(feat.array.astype(np.float16) + 1)  # deliberately not float32


def make_backend(device_type, setting="auto", torch=None, selftest=None):
    """A TorchBackend with fakes: no model file, no real torch."""
    backend = object.__new__(TorchBackend)
    backend._torch = torch or FakeTorch()
    backend.device = FakeDevice(device_type)
    backend.model = FakeModel()
    backend.weight_path = "unused"
    backend._weight_sha256 = "weights"
    backend._init_precision(setting)
    backend.selftest_calls = []
    if selftest is not None:
        def run(mode):
            backend.selftest_calls.append(mode)
            if isinstance(selftest, Exception):
                raise selftest
            return dict(selftest, mode=mode)
        backend._reduced_precision_selftest = run
    backend._torch.get_num_threads = lambda: 4
    return backend


PASSING = {"passed": True, "finite": True, "mean_abs_diff": 0.02, "max_abs_diff": 0.3, "seconds": 0.1}
FAILING = {"passed": False, "finite": True, "mean_abs_diff": 2.5, "max_abs_diff": 40.0, "seconds": 0.1}


# --------------------------------------------------------------------------
# Path selection per device and setting
# --------------------------------------------------------------------------

def test_auto_cuda_is_unchanged_autocast_without_selftest():
    backend = make_backend("cuda", "auto", selftest=PASSING)
    assert backend._resolve_precision()["mode"] == MODE_CUDA_AUTOCAST
    assert backend.precision_label() == "float16-autocast" == LABEL_CUDA_AUTOCAST
    assert backend.selftest_calls == []


def test_cuda_transform_uses_autocast_cuda_float16():
    backend = make_backend("cuda", "auto")
    out = backend.transform(np.zeros((2, 4, 1, 1), np.float32))
    assert backend._torch.autocast_calls == [("cuda", np.float16, True)]
    assert backend.model.infer_calls[0]["dtype"] == np.float32
    assert out.dtype == np.float16  # CUDA path hands back what autocast produced, as before


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_float32_setting_never_builds_an_autocast_context(device):
    backend = make_backend(device, "float32", selftest=PASSING)
    backend.transform(np.zeros((2, 4, 1, 1), np.float32))
    assert backend._torch.autocast_calls == []
    assert backend.precision_label() == "float32"
    assert backend.selftest_calls == []
    assert backend.model.infer_calls[0]["half_transformer"] is False


@pytest.mark.parametrize("setting", ["auto", "float32"])
@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_no_autocast_context_when_disabled(device, setting):
    backend = make_backend(device, setting, selftest=PASSING)
    backend.transform(np.zeros((2, 4, 1, 1), np.float32))
    assert backend._torch.autocast_calls == []
    assert backend.selftest_calls == []
    assert backend.precision_label() == "float32"


def test_cpu_float16_stays_float32_and_says_so():
    backend = make_backend("cpu", "float16", selftest=PASSING)
    backend.transform(np.zeros((2, 4, 1, 1), np.float32))
    assert backend._torch.autocast_calls == []
    assert backend.precision_label() == "float32"
    assert "not supported on cpu" in backend.benchmark_metadata()["precision_note"]
    assert backend.selftest_calls == []


def test_mps_float16_uses_autocast_when_supported():
    backend = make_backend("mps", "float16", selftest=PASSING)
    out = backend.transform(np.zeros((2, 4, 1, 1), np.float32))
    assert backend.selftest_calls == [MODE_MPS_AUTOCAST]
    assert backend.precision_label() == "float16-autocast-mps" == LABEL_MPS_AUTOCAST
    # one detection probe plus one real call, both for mps/float16
    assert backend._torch.autocast_calls == [("mps", np.float16, True)] * 2
    assert backend.model.infer_calls[0]["half_transformer"] is False
    assert out.dtype == np.float32  # always handed back to the float32 decoder as float32


def test_mps_float16_falls_back_to_explicit_half_when_autocast_unsupported():
    backend = make_backend("mps", "float16", torch=FakeTorch("missing"), selftest=PASSING)
    out = backend.transform(np.zeros((2, 4, 1, 1), np.float32))
    assert backend.selftest_calls == [MODE_MPS_HALF]
    assert backend.precision_label() == "float16-half-mps" == LABEL_MPS_HALF
    call = backend.model.infer_calls[0]
    assert call["dtype"] == np.float16 and call["half_transformer"] is True
    assert backend.model.transformer.is_half is False  # the original stays float32
    assert out.dtype == np.float32


def test_mps_autocast_that_does_not_change_dtype_counts_as_unsupported():
    backend = make_backend("mps", "float16", torch=FakeTorch("inert"), selftest=PASSING)
    assert backend._resolve_precision()["mode"] == MODE_MPS_HALF


def test_selftest_failure_falls_back_to_float32_with_one_warning(caplog):
    backend = make_backend("mps", "float16", selftest=FAILING)
    with caplog.at_level(logging.WARNING, logger="videowipe.backends"):
        backend.transform(np.zeros((2, 4, 1, 1), np.float32))
        backend.transform(np.zeros((2, 4, 1, 1), np.float32))
        backend.precision_label()
    warnings_logged = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings_logged) == 1 and "self-test" in warnings_logged[0].getMessage()
    assert backend.precision_label() == "float32-mps-fallback" == LABEL_MPS_FALLBACK
    assert backend._resolve_precision()["mode"] == MODE_FLOAT32
    assert backend.model.infer_calls[0]["half_transformer"] is False
    assert backend.selftest_calls == [MODE_MPS_AUTOCAST]  # not repeated
    note = backend.benchmark_metadata()["precision_note"]
    assert "self-test" in note and "float32" in note


def test_selftest_exception_falls_back_to_float32(caplog):
    backend = make_backend("mps", "float16", selftest=RuntimeError("Metal kernel missing"))
    with caplog.at_level(logging.WARNING, logger="videowipe.backends"):
        assert backend.precision_label() == LABEL_MPS_FALLBACK
    assert "Metal kernel missing" in backend.benchmark_metadata()["precision_note"]
    assert backend._precision_state["selftest"]["passed"] is False


def test_selftest_is_lazy_and_cached_per_instance():
    backend = make_backend("mps", "float16", selftest=PASSING)
    assert backend._precision_state is None and backend.selftest_calls == []
    backend._runtime_metadata()  # what __init__ uses for the weight hash
    assert backend.selftest_calls == []
    backend.benchmark_metadata()
    backend.precision_label()
    backend.transform(np.zeros((2, 4, 1, 1), np.float32))
    assert len(backend.selftest_calls) == 1
    other = make_backend("mps", "float16", selftest=PASSING)
    other.precision_label()
    assert len(other.selftest_calls) == 1  # a new instance tests again


def test_auto_on_mps_is_float32_today_and_never_selftests():
    backend = make_backend("mps", "auto", selftest=PASSING)
    backend.transform(np.zeros((2, 4, 1, 1), np.float32))
    assert backend.precision_label() == "float32"
    assert backend.selftest_calls == [] and backend._torch.autocast_calls == []


def test_half_copy_is_dropped_on_cleanup():
    backend = make_backend("mps", "float16", torch=FakeTorch("missing"), selftest=PASSING)
    backend._torch.mps = SimpleNamespace(empty_cache=lambda: None)
    backend.transform(np.zeros((2, 4, 1, 1), np.float32))
    assert backend._half_transformer is not None
    backend.cleanup()
    assert backend._half_transformer is None and backend.model is None


# --------------------------------------------------------------------------
# Labels in metadata and identities
# --------------------------------------------------------------------------

def test_benchmark_metadata_carries_precision():
    backend = make_backend("mps", "float16", selftest=PASSING)
    metadata = backend.benchmark_metadata()
    assert metadata["precision"] == "float16-autocast-mps"
    assert "precision_note" not in metadata
    assert metadata["device"] == "mps" and metadata["weight_sha256"] == "weights"


def test_default_label_for_backends_without_precision_support():
    onnx = object.__new__(ONNXBackend)
    onnx.encoder_session = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    onnx._runtime_version = "x"
    onnx._weight_sha256 = {}
    assert onnx.precision_label() == "float32"  # as before: the device is a provider list

    class CudaLike(InpaintBackend):
        def benchmark_metadata(self):
            return {"device": "cuda:0"}

    class CpuLike(CudaLike):
        def benchmark_metadata(self):
            return {"device": "cpu"}

    assert CudaLike().precision_label() == "float16-autocast"
    assert CpuLike().precision_label() == "float32"


def test_trial_identity_reports_backend_precision():
    from videowipe.engine import WipeEngine

    def identity(backend):
        task = SimpleNamespace(
            backend=backend, gap=25, feather_radius=4,
            inpainter=SimpleNamespace(ref_length=5, neighbor_stride=5))
        engine = SimpleNamespace(
            _run_lock=SimpleNamespace(acquire=lambda blocking=False: True, release=lambda: None),
            _ensure_model=lambda: None, _task_impl=task)
        return WipeEngine._trial_identity(engine)

    half = make_backend("mps", "float16", selftest=PASSING)
    fallback = make_backend("mps", "float16", selftest=FAILING)
    plain = make_backend("mps", "auto")
    legacy_cuda = SimpleNamespace(benchmark_metadata=lambda: {"device": "cuda:0", "weight_sha256": "w"})
    legacy_cpu = SimpleNamespace(benchmark_metadata=lambda: {"device": "cpu", "weight_sha256": "w"})
    assert identity(half)["precision"] == "float16-autocast-mps"
    assert identity(fallback)["precision"] == "float32-mps-fallback"
    assert identity(plain)["precision"] == "float32"
    assert identity(legacy_cuda)["precision"] == "float16-autocast"
    assert identity(legacy_cpu)["precision"] == "float32"
    assert len({identity(b)["precision"] for b in (half, fallback, plain)}) == 3


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")
def test_prediction_cache_never_reused_across_precisions(tmp_path, monkeypatch):
    from videowipe.inpainters import prediction_cache, sttn
    from videowipe.inpainters.base import InpaintJob
    import cv2

    video = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=96x64:rate=4:duration=3", "-c:v", "libx264", str(video)], check=True)
    calls = []

    def predict(frames, *args):
        calls.append(len(frames))
        return [np.mean(frames, axis=0).astype(np.uint8) for _ in frames]

    monkeypatch.setattr(sttn, "_process_segment", predict)
    identities = []
    real_cache = prediction_cache.PredictionCache

    def recording_cache(directory, identity):
        identities.append(dict(identity))
        return real_cache(directory, identity)

    monkeypatch.setattr(prediction_cache, "PredictionCache", recording_cache)
    state = {"label": "float32"}
    backend = SimpleNamespace(
        benchmark_metadata=lambda: {"device": "mps", "weight_sha256": "w"},
        precision_label=lambda: state["label"])
    painter = sttn.STTNInpainter()
    painter.backend = backend
    mask = np.zeros((64, 96, 1), dtype=np.uint8)
    mask[52:60, 15:85] = 1
    cache = tmp_path / "task" / "predictions"

    def run():
        calls.clear()
        reader = cv2.VideoCapture(str(video))
        try:
            painter.inpaint(InpaintJob(str(video), mask, str(tmp_path), 4, 12, 96, 64,
                                       reader=reader, gap=4, prediction_cache_dir=str(cache)))
        finally:
            reader.release()
        return len(calls)

    first = run()
    assert first > 0 and run() == 0                     # same precision: served from the cache
    state["label"] = "float16-autocast-mps"
    assert run() == first                               # different precision: recomputed
    assert run() == 0
    state["label"] = "float32"
    assert run() == 0                                   # the float32 entries are still valid
    assert [i["precision"] for i in identities[:3]] == ["float32", "float32", "float16-autocast-mps"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")
def test_cache_identity_for_backend_without_precision_label_is_legacy(tmp_path, monkeypatch):
    """Fakes such as the one in test_trial keep their historic float32/float16-autocast labels."""
    from videowipe.inpainters import prediction_cache, sttn
    from videowipe.inpainters.base import InpaintJob
    import cv2

    video = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=96x64:rate=4:duration=2", "-c:v", "libx264", str(video)], check=True)
    monkeypatch.setattr(sttn, "_process_segment",
                        lambda frames, *a: [np.mean(frames, axis=0).astype(np.uint8) for _ in frames])
    identities = []
    real_cache = prediction_cache.PredictionCache
    monkeypatch.setattr(prediction_cache, "PredictionCache",
                        lambda directory, identity: (identities.append(dict(identity)),
                                                     real_cache(directory, identity))[1])
    painter = sttn.STTNInpainter()
    mask = np.zeros((64, 96, 1), dtype=np.uint8)
    mask[52:60, 15:85] = 1
    for device in ("cpu", "cuda:0"):
        painter.backend = SimpleNamespace(benchmark_metadata=lambda device=device: {"device": device, "weight_sha256": "w"})
        reader = cv2.VideoCapture(str(video))
        try:
            painter.inpaint(InpaintJob(str(video), mask, str(tmp_path), 4, 8, 96, 64, reader=reader,
                                       gap=4, prediction_cache_dir=str(tmp_path / "p")))
        finally:
            reader.release()
    assert [i["precision"] for i in identities] == ["float32", "float16-autocast"]
