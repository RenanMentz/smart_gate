"""Status of the hardware and of the deep learning models.

Shown on the Settings screen so the operator knows *what is really running*:

* **GPU** - whether PyTorch sees a CUDA device and, importantly, whether the
  installed PyTorch build contains kernels for that GPU's architecture. A
  build compiled for older architectures reports ``cuda.is_available() ==
  True`` on a newer card but fails on the first operation ("no kernel image
  is available for execution on the device"). For example, RTX 50xx cards
  (Blackwell, compute capability 12.0 / ``sm_120``) need a CUDA 12.8 build
  (PyTorch >= 2.7).
* **YOLO** - whether the weights file exists, whether it is the published
  file (SHA-256 of ``scripts/download_model.py``) and whether it is loaded.
* **EasyOCR** - whether the reader is loaded and if it runs on the GPU.

Nothing here loads a model: reading the status is cheap and side-effect free.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from modules.detection import PlateDetector
from modules.device import arch_supported
from modules.ocr import PlateOCR


@dataclass
class GPUStatus:
    torch_version: str | None        # None if PyTorch is not installed
    cuda_available: bool
    device_name: str | None = None
    capability: str | None = None    # e.g. "sm_120"
    cuda_version: str | None = None  # CUDA version PyTorch was built with
    arch_supported: bool | None = None  # build has kernels for this GPU (None = unknown)
    message: str = ""

    @property
    def usable(self) -> bool:
        return self.cuda_available and self.arch_supported is not False


@dataclass
class ModelStatus:
    name: str
    loaded: bool
    error: str | None
    detail: str                      # one line for the operator


def gpu_status() -> GPUStatus:
    """Inspect PyTorch and the first CUDA device (never raises)."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        return GPUStatus(None, False, message=f"PyTorch could not be imported: {exc}")
    status = GPUStatus(torch.__version__, False, cuda_version=torch.version.cuda)
    try:
        if not torch.cuda.is_available():
            status.message = ("No CUDA GPU available: the models run on the CPU (slower)."
                              if torch.version.cuda else "CPU-only PyTorch build: the models run on the CPU.")
            return status
        status.cuda_available = True
        status.device_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        status.capability = f"sm_{major}{minor}"
        status.arch_supported = arch_supported(status.capability, torch.cuda.get_arch_list())
        if status.arch_supported is False:
            status.message = (f"This PyTorch build (CUDA {torch.version.cuda}) has no kernels for "
                              f"{status.capability}. Install a build that supports this GPU.")
        else:
            status.message = f"GPU in use: {status.device_name}."
    except Exception as exc:
        status.message = f"Could not query the GPU: {exc}"
    return status


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def yolo_status(detector: PlateDetector, expected_sha256: str | None = None,
                file_hash: str | None = None) -> ModelStatus:
    """Status of the YOLO detector.

    Args:
        expected_sha256: hash of the published weights.
        file_hash: hash of the file on disk, if already computed (the caller
            may cache it; hashing reads the whole file).
    """
    path = detector.model_path
    if not path.is_file():
        return ModelStatus("YOLO11", False, detector.load_error,
                           f"Weights not found at {path}. Run: python -m scripts.download_model")
    size_mb = path.stat().st_size / 2**20
    detail = f"{path.name} ({size_mb:.1f} MB)"
    if expected_sha256:
        file_hash = file_hash or sha256_of(path)
        detail += (", published weights (SHA-256 verified)" if file_hash == expected_sha256
                   else ", weights differ from the published file (e.g. trained locally)")
    if detector.loaded:
        detail += f", loaded on {'GPU' if str(detector.device) not in ('cpu', 'None') else 'CPU'}"
    elif not detector.load_error:
        detail += ", not loaded yet (loads on the first reading)"
    return ModelStatus("YOLO11", detector.loaded, detector.load_error, detail)


def ocr_status(ocr: PlateOCR) -> ModelStatus:
    if ocr.loaded:
        detail = f"Loaded on {'GPU' if ocr.gpu else 'CPU'}, languages: {', '.join(ocr.cfg.languages)}"
    elif ocr.load_error:
        detail = "Failed to load."
    else:
        detail = "Not loaded yet (loads on the first reading)."
    return ModelStatus("EasyOCR", ocr.loaded, ocr.load_error, detail)
