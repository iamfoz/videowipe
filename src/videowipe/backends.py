"""Inference backends: PyTorch or ONNX Runtime."""
from __future__ import annotations

import copy
import hashlib
import logging
import os
import threading
import time
from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np


def _detect_backend(weight_path: str) -> str:
    """Detect backend from weight file extension."""
    if weight_path.endswith(".onnx"):
        return "onnx"
    if weight_path.endswith(".pth") or weight_path.endswith(".pt"):
        return "torch"
    raise ValueError(
        f"Unsupported weight file: {weight_path}. Expected .onnx, .pth, or .pt."
    )


def load_backend(
    weight_path: str,
    device: str = "auto",
    precision: Optional[str] = None,
) -> "InpaintBackend":
    """Load the appropriate backend based on weight file type.

    If weight_path ends with .onnx, uses ONNX Runtime.
    Otherwise, uses PyTorch (requires torch installed). ``precision`` is only
    used by the PyTorch backend; ``None`` reads ``VIDEOWIPE_PRECISION``.
    """
    backend = _detect_backend(weight_path)
    if backend == "onnx":
        return ONNXBackend(weight_path)
    return TorchBackend(weight_path, device=device, precision=precision)


logger = logging.getLogger(__name__)

# --- Numeric precision --------------------------------------------------------
#
# ``VIDEOWIPE_PRECISION`` selects how the transformer (``model.infer``) runs:
#   auto     today's behaviour: float16 autocast on CUDA, float32 elsewhere
#   float32  never use reduced precision, on any device
#   float16  half precision where the device supports it (CUDA, MPS); CPU stays
#            float32 because it has no benefit there
# The encoder and decoder always run in float32.
PRECISION_ENV = "VIDEOWIPE_PRECISION"
PRECISIONS = ("auto", "float32", "float16")

# Precision labels. They feed ``benchmark_metadata()``, the trial identity and
# the prediction-cache identity, so a cache built in one precision is never
# reused in another. The first two are the historic strings.
LABEL_FLOAT32 = "float32"
LABEL_CUDA_AUTOCAST = "float16-autocast"
LABEL_MPS_AUTOCAST = "float16-autocast-mps"
LABEL_MPS_HALF = "float16-half-mps"
LABEL_MPS_FALLBACK = "float32-mps-fallback"

# Execution modes of ``TorchBackend.transform``.
MODE_FLOAT32 = "float32"
MODE_CUDA_AUTOCAST = "cuda-autocast"
MODE_MPS_AUTOCAST = "mps-autocast"
MODE_MPS_HALF = "mps-half"

# Self-test thresholds, on the 0-255 decoded-pixel scale. The mean bound is the
# PRD's quality gate (C4: 0.5/255 in the removal region, here applied to the
# whole synthetic frame, which is stricter). The max bound only catches
# localised blow-ups (overflow, a few pixels saturating); half precision on a
# healthy model differs by a few levels at most.
SELFTEST_MAX_MEAN_ABS_DIFF = 0.5
SELFTEST_MAX_ABS_DIFF = 16.0
SELFTEST_FRAMES = 12
SELFTEST_SIZE = (120, 640)  # (height, width) of the model input


def parse_precision(value: Optional[str] = None) -> str:
    """Return ``auto``, ``float32`` or ``float16``.

    ``None`` reads ``VIDEOWIPE_PRECISION``; anything unrecognised is ``auto``.
    """
    if value is None:
        value = os.environ.get(PRECISION_ENV)
    value = str(value or "").strip().lower()
    return value if value in PRECISIONS else "auto"


def wants_reduced_precision(setting: str, device_type: str) -> bool:
    """Whether ``setting`` asks for half precision on this kind of device."""
    if setting == "float32":
        return False
    if setting == "float16":
        return device_type in ("cuda", "mps")
    # auto: CUDA only. To make ``auto`` prefer half precision on MPS once it
    # has been measured on a Mac, change this line to
    # ``device_type in ("cuda", "mps")``; the self-test still guards MPS.
    return device_type == "cuda"


class InpaintBackend:
    """Unified interface for model inference.

    All methods take and return numpy arrays so that the
    calling code (task layer) stays backend-agnostic.
    """

    def preprocess(self, frames: List[np.ndarray]) -> np.ndarray:
        """Convert list of (H, W, 3) BGR uint8 frames to (T, 3, H, W) float32 in [-1, 1]."""
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        arr = np.stack(rgb, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
        arr = arr / 127.5 - 1.0
        return arr

    def encode(self, tensor: np.ndarray) -> np.ndarray:
        """(T, 3, H, W) → (T, C, H', W')"""
        raise NotImplementedError

    def transform(self, feats: np.ndarray) -> np.ndarray:
        """(T, C, H', W') → (T, C, H', W')"""
        raise NotImplementedError

    def decode(self, feats: np.ndarray) -> np.ndarray:
        """(T, C, H', W') → (T, H, W, 3) uint8 in [0, 255]."""
        raise NotImplementedError

    def precision_label(self) -> str:
        """Numeric precision in use, as recorded in cache identities."""
        device = self.benchmark_metadata().get("device")
        return LABEL_CUDA_AUTOCAST if str(device).startswith("cuda") else LABEL_FLOAT32

    def cleanup(self) -> None:
        pass


class TorchBackend(InpaintBackend):
    """PyTorch model backend. Requires torch and torchvision."""

    def __init__(self, weight_path: str, device: str = "auto",
                 precision: Optional[str] = None):
        import torch
        from videowipe.models.sttn import InpaintGenerator

        if device == "auto":
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable in this environment; use device=cpu or a supported macOS runtime")
        self.weight_path = weight_path
        self.model = InpaintGenerator().to(self.device)
        data = torch.load(weight_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(data["netG"])
        self.model.eval()
        self._torch = torch
        self._init_precision(precision)
        self._weight_sha256 = self._runtime_metadata()["weight_sha256"]

    # -- precision ---------------------------------------------------------

    def _init_precision(self, precision: Optional[str]) -> None:
        self.precision_setting = parse_precision(precision)
        self._precision_lock = threading.Lock()
        self._precision_state: Optional[dict] = None
        self._half_transformer = None

    def _resolve_precision(self) -> dict:
        """Decide (once per backend) how the transformer runs.

        Returns ``{"mode", "label", "note", "selftest"}``. The one-off MPS
        self-test runs here, the first time the answer is needed, never at
        construction.
        """
        state = self._precision_state
        if state is not None:
            return state
        with self._precision_lock:
            if self._precision_state is None:
                self._precision_state = self._decide_precision()
            return self._precision_state

    def _decide_precision(self) -> dict:
        device_type = self.device.type
        setting = self.precision_setting
        state = {"mode": MODE_FLOAT32, "label": LABEL_FLOAT32, "note": None, "selftest": None}
        if not wants_reduced_precision(setting, device_type):
            if setting == "float16":
                state["note"] = f"float16 requested but not supported on {device_type}; using float32"
            return state
        if device_type == "cuda":
            # Existing behaviour: autocast, no self-test.
            state.update(mode=MODE_CUDA_AUTOCAST, label=LABEL_CUDA_AUTOCAST)
            return state
        # MPS: new territory, so detect the mechanism and verify the numbers.
        mode = MODE_MPS_AUTOCAST if self._mps_autocast_supported() else MODE_MPS_HALF
        label = LABEL_MPS_AUTOCAST if mode == MODE_MPS_AUTOCAST else LABEL_MPS_HALF
        try:
            report = self._reduced_precision_selftest(mode)
        except Exception as exc:  # any failure means "do not trust half precision"
            report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        state["selftest"] = report
        if report["passed"]:
            state.update(mode=mode, label=label)
        else:
            reason = report.get("error") or (
                f"mean abs diff {report.get('mean_abs_diff'):.4f}, "
                f"max abs diff {report.get('max_abs_diff'):.4f}, "
                f"finite={report.get('finite')}")
            state.update(label=LABEL_MPS_FALLBACK,
                         note=f"half precision on mps failed its self-test ({reason}); using float32")
            logger.warning("VideoWipe: %s", state["note"])
        return state

    def precision_label(self) -> str:
        """Precision actually in use (resolves the self-test if one is due)."""
        return self._resolve_precision()["label"]

    def _mps_autocast_supported(self) -> bool:
        """Feature-detect MPS autocast by using it once on a tiny matmul."""
        torch = self._torch
        try:
            a = torch.ones((2, 2), device=self.device)
            with torch.autocast(device_type="mps", dtype=torch.float16):
                out = torch.matmul(a, a)
            return out.dtype == torch.float16
        except Exception:
            return False

    def _half_transformer_for_infer(self):
        """A float16 copy of the transformer only (explicit half path)."""
        if self._half_transformer is None:
            self._half_transformer = copy.deepcopy(self.model.transformer).half().eval()
        return self._half_transformer

    def _infer(self, t, mode: str):
        """Run the transformer on feature tensor ``t`` in the given mode."""
        torch = self._torch
        if mode == MODE_CUDA_AUTOCAST:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.model.infer(t)
        if mode == MODE_MPS_AUTOCAST:
            with torch.autocast(device_type="mps", dtype=torch.float16):
                result = self.model.infer(t)
            return result.float()
        if mode == MODE_MPS_HALF:
            holder = SimpleNamespace(transformer=self._half_transformer_for_infer())
            return type(self.model).infer(holder, t.half()).float()
        return self.model.infer(t)

    def _reduced_precision_selftest(self, mode: str) -> dict:
        """Compare float32 with the reduced-precision path on fixed inputs.

        A seeded random image batch goes through the real encoder (so the
        features have realistic scale), then the transformer in float32 and in
        ``mode``, then the decoder. The model is not modified.
        """
        torch = self._torch
        started = time.perf_counter()
        height, width = SELFTEST_SIZE
        gen = torch.Generator(device="cpu").manual_seed(20240607)
        images = torch.rand((SELFTEST_FRAMES, 3, height, width), generator=gen) * 2.0 - 1.0
        with torch.no_grad():
            feats = self.model.encoder(images.to(self.device))
            reference = self.model.infer(feats)
            candidate = self._infer(feats, mode)
            finite = bool(torch.isfinite(candidate).all().item())
            ref_px = self._decode_tensor(reference).float().cpu()
            cand_px = self._decode_tensor(candidate.float()).float().cpu()
            finite = finite and bool(torch.isfinite(cand_px).all().item())
            diff = (ref_px - cand_px).abs()
            mean_diff = float(diff.mean().item())
            max_diff = float(diff.max().item())
        passed = (finite and mean_diff <= SELFTEST_MAX_MEAN_ABS_DIFF
                  and max_diff <= SELFTEST_MAX_ABS_DIFF)
        return {"passed": bool(passed), "finite": finite, "mode": mode,
                "mean_abs_diff": mean_diff, "max_abs_diff": max_diff,
                "seconds": time.perf_counter() - started}

    # -- metadata ----------------------------------------------------------

    def _runtime_metadata(self) -> dict:
        if hasattr(self, "_weight_sha256"):
            return {"device": str(self.device), "torch": self._torch.__version__,
                    "threads": self._torch.get_num_threads(), "weight_sha256": self._weight_sha256}
        digest = hashlib.sha256()
        with open(self.weight_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return {"device": str(self.device), "torch": self._torch.__version__,
                "threads": self._torch.get_num_threads(), "weight_sha256": digest.hexdigest()}

    def benchmark_metadata(self) -> dict:
        metadata = self._runtime_metadata()
        state = self._resolve_precision()
        metadata["precision"] = state["label"]
        if state["note"]:
            metadata["precision_note"] = state["note"]
        return metadata

    def encode(self, tensor: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            t = torch.from_numpy(tensor).to(self.device)
            if t.is_cuda:
                t = t.to(memory_format=torch.channels_last)
            feats = self.model.encoder(t)
        return feats.cpu().numpy()

    def transform(self, feats: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            t = torch.from_numpy(feats).to(self.device)
            if t.is_cuda:
                t = t.to(memory_format=torch.channels_last)
            result = self._infer(t, self._resolve_precision()["mode"])
        return result.cpu().numpy()

    def decode(self, feats: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            t = torch.from_numpy(feats).to(self.device)
            pred_img = self._decode_tensor(t)
            pred_img = pred_img.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return pred_img.astype(np.uint8)

    def _decode_tensor(self, t):
        """Decoder + tanh + scale to [0, 255], still (T, 3, H, W) float32."""
        torch = self._torch
        decoded = self.model.decoder(t)
        pred_img = torch.tanh(decoded)
        return ((pred_img + 1.0) * 127.5).clamp(0, 255)

    def cleanup(self) -> None:
        torch = self._torch
        self._half_transformer = None
        if self.model is not None:
            del self.model
            self.model = None
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device is not None and self.device.type == "mps":
            # Apple Silicon memory is shared with the system: return the Metal
            # allocator's cached blocks once the model's tensors are collected.
            import gc

            gc.collect()
            empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
            if empty_cache is not None:
                empty_cache()


class ONNXBackend(InpaintBackend):
    """ONNX Runtime backend. Requires onnxruntime.

    Expects three ONNX model files:
      - <base>_encoder.onnx
      - <base>_transformer.onnx
      - <base>_decoder.onnx
    where <base> is weight_path with .onnx stripped.
    """

    def __init__(self, weight_path: str):
        import onnxruntime as ort

        base = weight_path[: -len(".onnx")]
        required = [
            f"{base}_encoder.onnx",
            f"{base}_transformer.onnx",
            f"{base}_decoder.onnx",
        ]
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise ValueError(
                "ONNX backend expects a prefix path ending in .onnx. Missing files: "
                + ", ".join(missing)
            )
        available = ort.get_available_providers()
        if "CPUExecutionProvider" not in available:
            raise RuntimeError(
                "ONNX Runtime CPUExecutionProvider is required for STTN"
            )
        providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in available
        ]

        self.encoder_session = ort.InferenceSession(
            f"{base}_encoder.onnx", providers=providers
        )
        self.transformer_session = ort.InferenceSession(
            f"{base}_transformer.onnx", providers=providers
        )
        self.decoder_session = ort.InferenceSession(
            f"{base}_decoder.onnx", providers=providers
        )

        self._enc_name = self.encoder_session.get_inputs()[0].name
        self._trans_name = self.transformer_session.get_inputs()[0].name
        self._dec_name = self.decoder_session.get_inputs()[0].name
        self._weight_sha256 = {}
        for path in required:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            self._weight_sha256[os.path.basename(path)] = digest.hexdigest()
        self._runtime_version = ort.__version__

    def benchmark_metadata(self) -> dict:
        return {"device": self.encoder_session.get_providers(),
                "onnxruntime": self._runtime_version,
                "weight_sha256": dict(self._weight_sha256)}

    def encode(self, tensor: np.ndarray) -> np.ndarray:
        return self.encoder_session.run(None, {self._enc_name: tensor})[0]

    def transform(self, feats: np.ndarray) -> np.ndarray:
        return self.transformer_session.run(None, {self._trans_name: feats})[0]

    def decode(self, feats: np.ndarray) -> np.ndarray:
        raw = self.decoder_session.run(None, {self._dec_name: feats})[0]
        raw = np.tanh(raw)
        raw = ((raw + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return raw.transpose(0, 2, 3, 1)
