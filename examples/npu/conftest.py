import torch


def _enable_cpu_emulation() -> None:
    import pyqcu.cann as _cann
    from pyqcu.dslash import _wilson as _wilson
    from pyqcu.smear import _stout as _stout
    from pyqcu.tools import _define as _tools_define
    from pyqcu.tools import _multigrid as _tools_multigrid

    _cann.force_use_npu = True
    _wilson.force_use_npu = True
    _stout.force_use_npu = True
    _tools_define.force_use_npu = True
    _tools_multigrid.force_use_npu = True


def _select_device() -> torch.device:
    try:
        import torch_npu  # noqa: F401
        return torch.device("npu")
    except Exception as exc:
        _enable_cpu_emulation()
        print(
            "PYQCU::NPU::CONFTEST:\n"
            f" torch_npu unavailable or npu device unsupported: {exc}\n"
            "PYQCU::NPU::CONFTEST:\n"
            " using CPU emulation via force_use_npu=True"
        )
        return torch.device("cpu")


def main() -> int:
    from pyqcu.testing import test_solver

    device = _select_device()
    test_solver(
        kind="wilson",
        method="multigrid",
        dtype=torch.complex64,
        lat_size=[8, 8, 8, 16],
        device=device,
        support_parity=True,
    )
    return 0


if __name__ == "__main__":
    main()
