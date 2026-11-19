"""Real-torch checks of the precision paths on CPU (skipped without torch).

Apple's MPS cannot run here; these prove that ``auto`` is bit-identical to the
original implementation on CPU, that the explicit half path runs and stays
within the quality gate, and that the self-test catches bad numerics.
"""
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from videowipe import backends  # noqa: E402
from videowipe.backends import MODE_FLOAT32, MODE_MPS_HALF, TorchBackend  # noqa: E402

FRAMES = 2  # features must be 160x30 (input 640x120); two frames keep CPU time short


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    """Random weights, scaled so activations are O(1) (the default init is near zero,
    and larger scales overflow even float32 through eight residual blocks)."""
    from videowipe.models.sttn import InpaintGenerator

    torch.manual_seed(0)
    model = InpaintGenerator(init_weights=False)
    model.init_weights("normal", 0.03)
    path = tmp_path_factory.mktemp("weights") / "random.pth"
    torch.save({"netG": model.state_dict()}, path)
    return str(path)


@pytest.fixture(scope="module")
def features(weights):
    backend = TorchBackend(weights, device="cpu", precision="float32")
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 256, (120, 640, 3), dtype=np.uint8) for _ in range(FRAMES)]
    feats = backend.encode(backend.preprocess(frames))
    assert feats.shape == (FRAMES, 256, 30, 160)
    return feats


def _original_transform(model, feats):
    """The pre-change TorchBackend.transform, verbatim."""
    with torch.no_grad():
        t = torch.from_numpy(feats).to("cpu")
        if t.is_cuda:
            t = t.to(memory_format=torch.channels_last)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.cuda.amp.autocast(enabled=t.is_cuda):
                result = model.infer(t)
    return result.cpu().numpy()


def _original_decode(model, feats):
    with torch.no_grad():
        t = torch.from_numpy(feats).to("cpu")
        decoded = model.decoder(t)
        pred_img = torch.tanh(decoded)
        pred_img = ((pred_img + 1.0) * 127.5).clamp(0, 255)
        pred_img = pred_img.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return pred_img.astype(np.uint8)


@pytest.mark.parametrize("setting", ["auto", "float16"])
def test_cpu_transform_and_decode_are_bit_identical_to_the_original(weights, features, setting):
    backend = TorchBackend(weights, device="cpu", precision=setting)
    expected = _original_transform(backend.model, features)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the deprecated torch.cuda.amp.autocast warning is gone
        actual = backend.transform(features)
    assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
    assert np.array_equal(backend.decode(actual), _original_decode(backend.model, expected))
    assert backend.precision_label() == "float32"


def test_backend_reads_precision_from_environment(weights, monkeypatch):
    monkeypatch.setenv("VIDEOWIPE_PRECISION", "float16")
    assert TorchBackend(weights, device="cpu").precision_setting == "float16"
    assert TorchBackend(weights, device="cpu", precision="float32").precision_setting == "float32"
    monkeypatch.setenv("VIDEOWIPE_PRECISION", "garbage")
    assert TorchBackend(weights, device="cpu").precision_setting == "auto"


def _half_supported_on_cpu():
    try:
        torch.nn.Conv2d(2, 2, 1).half()(torch.zeros(1, 2, 2, 2, dtype=torch.float16))
        torch.zeros(2, 2, dtype=torch.float16) @ torch.zeros(2, 2, dtype=torch.float16)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _half_supported_on_cpu(), reason="this torch build has no CPU float16 kernels")
def test_explicit_half_path_runs_on_cpu_and_stays_within_the_gate(weights, features):
    backend = TorchBackend(weights, device="cpu", precision="float32")
    reference = backend.transform(features)
    before = {k: v.clone() for k, v in backend.model.state_dict().items()}
    half = backend._infer(torch.from_numpy(features), MODE_MPS_HALF).detach().numpy()
    assert half.dtype == np.float32 and np.isfinite(half).all()
    assert all(p.dtype == torch.float32 for p in backend.model.parameters())
    assert all(torch.equal(v, before[k]) for k, v in backend.model.state_dict().items())
    assert backend._half_transformer is not None
    pixels = backend.decode(reference).astype(np.float32)
    diff = np.abs(pixels - backend.decode(half).astype(np.float32))
    assert diff.mean() <= 0.5 and diff.max() <= backends.SELFTEST_MAX_ABS_DIFF


@pytest.mark.skipif(not _half_supported_on_cpu(), reason="this torch build has no CPU float16 kernels")
def test_selftest_passes_on_a_healthy_model_and_leaves_it_untouched(weights, monkeypatch):
    monkeypatch.setattr(backends, "SELFTEST_FRAMES", FRAMES)
    backend = TorchBackend(weights, device="cpu", precision="float32")
    before = {k: v.clone() for k, v in backend.model.state_dict().items()}
    report = backend._reduced_precision_selftest(MODE_MPS_HALF)
    assert report["passed"] and report["finite"]
    assert report["mean_abs_diff"] <= backends.SELFTEST_MAX_MEAN_ABS_DIFF
    assert report["max_abs_diff"] <= backends.SELFTEST_MAX_ABS_DIFF
    assert report["seconds"] > 0
    assert not backend.model.training
    assert all(torch.equal(v, before[k]) for k, v in backend.model.state_dict().items())
    # deterministic: fixed seed, same numbers twice
    again = backend._reduced_precision_selftest(MODE_MPS_HALF)
    assert (again["mean_abs_diff"], again["max_abs_diff"]) == (report["mean_abs_diff"], report["max_abs_diff"])


def test_selftest_rejects_non_finite_output(weights, monkeypatch):
    monkeypatch.setattr(backends, "SELFTEST_FRAMES", FRAMES)
    backend = TorchBackend(weights, device="cpu", precision="float32")
    real_infer = backend._infer

    def broken(t, mode):
        out = real_infer(t, MODE_FLOAT32).clone()
        if mode != MODE_FLOAT32:
            out[0, 0, 0, 0] = float("nan")
        return out

    backend._infer = broken
    report = backend._reduced_precision_selftest(MODE_MPS_HALF)
    assert report["passed"] is False and report["finite"] is False


def test_selftest_rejects_large_error(weights, monkeypatch):
    monkeypatch.setattr(backends, "SELFTEST_FRAMES", FRAMES)
    backend = TorchBackend(weights, device="cpu", precision="float32")
    real_infer = backend._infer
    backend._infer = lambda t, mode: real_infer(t, MODE_FLOAT32) * (1.0 if mode == MODE_FLOAT32 else 3.0) + (
        0.0 if mode == MODE_FLOAT32 else 1.0)
    report = backend._reduced_precision_selftest(MODE_MPS_HALF)
    assert report["passed"] is False and report["finite"] is True
    assert report["mean_abs_diff"] > backends.SELFTEST_MAX_MEAN_ABS_DIFF


def test_overflowing_half_precision_is_caught_by_the_selftest():
    """Activations beyond float16's range (65504) turn into inf/nan: the failure the PRD warns of."""
    from videowipe.models.sttn import InpaintGenerator

    torch.manual_seed(0)
    model = InpaintGenerator(init_weights=False)
    model.init_weights("normal", 0.04)  # activations of order 1e2-1e3 per block
    import tempfile, os
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "hot.pth")
        torch.save({"netG": model.state_dict()}, path)
        backend = TorchBackend(path, device="cpu", precision="float32")
    original = backends.SELFTEST_FRAMES
    backends.SELFTEST_FRAMES = FRAMES
    try:
        if not _half_supported_on_cpu():
            pytest.skip("no CPU float16 kernels")
        report = backend._reduced_precision_selftest(MODE_MPS_HALF)
    finally:
        backends.SELFTEST_FRAMES = original
    assert report["passed"] is False
