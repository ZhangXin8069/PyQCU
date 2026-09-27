import datetime
import os


def _enable_cpu_npu_emulation() -> None:
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


def _select_device():
    import torch
    try:
        import torch_npu  # noqa: F401
        return torch.device('npu')
    except Exception as exc:
        _enable_cpu_npu_emulation()
        print(
            "PYQCU::PROFILER::NPU:\n"
            f" torch_npu unavailable or npu device unsupported: {exc}\n"
            "PYQCU::PROFILER::NPU:\n"
            " using CPU emulation via force_use_npu=True"
        )
        return torch.device('cpu')


def main() -> None:
    import torch
    import mpi4py.MPI as MPI
    comm = MPI.COMM_WORLD
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    rank = comm.Get_rank()
    device = _select_device()
    from pyqcu.testing import test_solver
    prof = torch.profiler.profile(
        record_shapes=True,
        with_modules=True,
        with_flops=True,
        with_stack=True,
    )
    prof.start()
    test_solver(method='bistabcg', dtype=torch.complex64, device=device,
                lat_size=[8, 8, 16, 16], support_parity=True)
    prof.stop()
    prof.export_chrome_trace(
        f"{os.path.abspath(os.path.dirname(__file__))}/trace_{time}_{rank}.json")


if __name__ == '__main__':
    main()
