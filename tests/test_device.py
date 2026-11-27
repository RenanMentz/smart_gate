"""Tests for the GPU / CPU choice (PyTorch is faked, no GPU needed)."""

import sys
import types

import pytest

from modules import device


def fake_torch(available=True, capability=(12, 0), arch_list=("sm_80", "sm_90")):
    cuda = types.SimpleNamespace(is_available=lambda: available, get_device_capability=lambda i: capability,
                                 get_arch_list=lambda: list(arch_list))
    return types.SimpleNamespace(cuda=cuda, __version__="fake")


@pytest.fixture(autouse=True)
def clear_cache():
    device.cuda_usable.cache_clear()
    yield
    device.cuda_usable.cache_clear()


@pytest.mark.parametrize(("torch_module", "usable"), [
    (fake_torch(arch_list=("sm_80", "sm_90")), False),          # RTX 50xx with the cu124 build
    (fake_torch(arch_list=("sm_90", "sm_120")), True),          # cu128 build
    (fake_torch(available=False), False),                       # no GPU
    (fake_torch(capability=(8, 9), arch_list=("compute_80",)), True),   # PTX JIT
])
def test_cuda_usable(monkeypatch, torch_module, usable):
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    assert device.cuda_usable() is usable


def test_arch_supported():
    assert device.arch_supported("sm_120", ["sm_80", "sm_90", "sm_120"]) is True
    assert device.arch_supported("sm_120", ["sm_50", "sm_80", "sm_90"]) is False
    assert device.arch_supported("sm_89", ["sm_80", "compute_80"]) is True
    assert device.arch_supported("sm_89", []) is None
