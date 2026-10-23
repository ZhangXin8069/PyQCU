"""Device-normalization tests for batched null-vector construction."""

import torch

from pyqcu.torch.tools._multigrid import _normalise_nullvec_device


def test_index_free_cuda_device_uses_current_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert _normalise_nullvec_device("cuda") == torch.device("cuda", 2)


def test_indexed_cuda_device_is_preserved():
    assert _normalise_nullvec_device("cuda:1") == torch.device("cuda", 1)


def test_cpu_device_is_preserved():
    assert _normalise_nullvec_device("cpu") == torch.device("cpu")
