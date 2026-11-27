"""Choose where the neural networks run (GPU or CPU).

``torch.cuda.is_available()`` is not enough: a PyTorch build compiled for older
GPU architectures reports ``True`` on a newer card and then fails on the first
operation with "no kernel image is available for execution on the device".
This happened with the RTX 50xx (Blackwell, ``sm_120``) and the cu124 build.

:func:`cuda_usable` therefore also checks that the build contains kernels for
the GPU's architecture. When it does not, the models run on the CPU (slower
but correct) instead of failing every reading.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def arch_supported(capability: str, arch_list: list[str]) -> bool | None:
    """Does the build contain kernels for ``capability`` (``sm_XY``)?

    PTX entries (``compute_XY``) can be JIT-compiled for newer GPUs of the
    same major version or later, so they also count. ``None`` = unknown.
    """
    if not arch_list:
        return None
    if capability in arch_list:
        return True
    major = int(capability.removeprefix("sm_")) // 10
    for arch in arch_list:
        if arch.startswith("compute_") and int(arch.removeprefix("compute_")) // 10 <= major:
            return True
    return False


@lru_cache(maxsize=1)
def cuda_usable() -> bool:
    """True if a CUDA GPU is present *and* the installed PyTorch can run on it."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability(0)
        capability = f"sm_{major}{minor}"
        if arch_supported(capability, torch.cuda.get_arch_list()) is False:
            logger.warning("PyTorch %s has no kernels for %s; running on the CPU.", torch.__version__, capability)
            return False
        return True
    except Exception:  # broken driver, etc.
        logger.exception("CUDA check failed; running on the CPU.")
        return False
