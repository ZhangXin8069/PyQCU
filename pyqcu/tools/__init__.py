"""Backend-neutral PyQCU utilities and Torch compatibility exports.

MPI/layout, HDF5, environment, resource-budget and plotting helpers are common
and implemented in this directory. Torch-only numerical helpers live in
``pyqcu.torch.tools`` and are loaded lazily here for old public imports.
"""

from __future__ import annotations

from importlib import import_module

from ._bench_plots import BENCH_GRID as BENCH_GRID
from ._bench_plots import BENCH_INK as BENCH_INK
from ._bench_plots import BENCH_INK2 as BENCH_INK2
from ._bench_plots import BENCH_MUTED as BENCH_MUTED
from ._bench_plots import BENCH_PALETTE as BENCH_PALETTE
from ._bench_plots import BENCH_SEQ_BLUE as BENCH_SEQ_BLUE
from ._bench_plots import BENCH_SURF as BENCH_SURF
from ._bench_plots import label_of as label_of
from ._bench_plots import load_json as load_json
from ._bench_plots import mpl_cjk_setup as mpl_cjk_setup
from ._bench_plots import plot_bench_report as plot_bench_report
from ._bench_plots import plot_convergence as plot_convergence
from ._bench_plots import plot_hotspot as plot_hotspot
from ._bench_plots import plot_rank_metric as plot_rank_metric
from ._bench_plots import plot_speedup as plot_speedup
from ._bench_plots import plot_sweep_curves as plot_sweep_curves
from ._bench_plots import plot_time as plot_time
from ._bench_plots import save_figure as save_figure
from ._bench_plots import save_figure_variants as save_figure_variants
from ._bench_plots import short_label as short_label
from ._bench_plots import style_axes as style_axes
from ._budget import CONST_PER_V as CONST_PER_V
from ._budget import budget_table as budget_table
from ._budget import disk_cache_bytes as disk_cache_bytes
from ._budget import fit_from_bench as fit_from_bench
from ._budget import rss_model as rss_model
from ._budget import vram_model as vram_model
from ._budget import vram_model_warm as vram_model_warm
from ._define import HAS_MPI_SUPPORT as HAS_MPI_SUPPORT
from ._define import ccdptzyx2ccdxyzt as ccdptzyx2ccdxyzt
from ._define import ccdxyzt2ccdptzyx as ccdxyzt2ccdptzyx
from ._define import check_mpi_support as check_mpi_support
from ._define import give_eo_mask as give_eo_mask
from ._define import give_grid_index as give_grid_index
from ._define import give_grid_size as give_grid_size
from ._define import give_rank_minus as give_rank_minus
from ._define import give_rank_minus_minus as give_rank_minus_minus
from ._define import give_rank_minus_plus as give_rank_minus_plus
from ._define import give_rank_plus as give_rank_plus
from ._define import give_rank_plus_minus as give_rank_plus_minus
from ._define import give_rank_plus_plus as give_rank_plus_plus
from ._define import local_xyzt2whole_xyzt as local_xyzt2whole_xyzt
from ._define import np2torch_dtype as np2torch_dtype
from ._define import oooxyzt2poooxyzt as oooxyzt2poooxyzt
from ._define import poooxyzt2oooxyzt as poooxyzt2oooxyzt
from ._define import prime_factorization as prime_factorization
from ._define import psctzyx2scxyzt as psctzyx2scxyzt
from ._define import scxyzt2psctzyx as scxyzt2psctzyx
from ._define import set_device as set_device
from ._define import slice_dim as slice_dim
from ._define import slice_dim_dim as slice_dim_dim
from ._define import slice_dim_none_dim as slice_dim_none_dim
from ._define import to_contiguous_real as to_contiguous_real
from ._define import torch2np_dtype as torch2np_dtype
from ._define import torch2tl_dtype as torch2tl_dtype
from ._define import warp_size as warp_size
from ._define import whole_xyzt2local_xyzt as whole_xyzt2local_xyzt
from ._env import cache_disk_mb as cache_disk_mb
from ._env import dump_env_h5 as dump_env_h5
from ._env import git_snapshot as git_snapshot
from ._env import gpu_snapshot as gpu_snapshot
from ._env import gpu_used_mb as gpu_used_mb
from ._env import nullvec_cache_dir as nullvec_cache_dir
from ._env import rss_kb as rss_kb
from ._io import gridoooxyzt2hdf5oooxyzt as gridoooxyzt2hdf5oooxyzt
from ._io import hdf5oooxyzt2gridoooxyzt as hdf5oooxyzt2gridoooxyzt
from ._io import load_dict_h5 as load_dict_h5
from ._io import load_tensor_h5 as load_tensor_h5
from ._io import save_dict_h5 as save_dict_h5
from ._io import save_tensor_h5 as save_tensor_h5

_TORCH_MODULES = {
    "_distributed_setup",
    "_einsum",
    "_hierarchical",
    "_linalg",
    "_matul",
    "_mpi_roll",
    "_multigrid",
    "_roll",
    "_strict_galerkin",
}

_TORCH_EXPORTS = {
    "Eexyzt_exyzt2Exyzt": "_einsum",
    "HierarchicalCache": "_hierarchical",
    "HierarchicalTensor": "_hierarchical",
    "PAIRS": "_multigrid",
    "SIGN": "_multigrid",
    "apply_stencil": "_multigrid",
    "build_stencil": "_multigrid",
    "build_stencil_mt": "_multigrid",
    "give_null_vecs": "_multigrid",
    "give_null_vecs_mt": "_multigrid",
    "local_orthogonalize": "_multigrid",
    "matmul_cpu": "_matul",
    "matmul_gpu": "_matul",
    "norm": "_linalg",
    "offload_to_disk": "_hierarchical",
    "offload_to_ram": "_hierarchical",
    "prolong": "_multigrid",
    "reload_to_vram": "_hierarchical",
    "restrict": "_multigrid",
    "vdot": "_linalg",
}


def __getattr__(name):
    module_name = name if name in _TORCH_MODULES else _TORCH_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'pyqcu.tools' has no attribute {name!r}")
    module = import_module(f"pyqcu.torch.tools.{module_name}")
    value = module if name in _TORCH_MODULES else getattr(module, name)
    globals()[name] = value
    return value


from argparse import Namespace

Namespace.__module__ = "pyqcu.tools"

__all__ = [
    "BENCH_GRID",
    "BENCH_INK",
    "BENCH_INK2",
    "BENCH_MUTED",
    "BENCH_PALETTE",
    "BENCH_SEQ_BLUE",
    "BENCH_SURF",
    "CONST_PER_V",
    "Eexyzt_exyzt2Exyzt",
    "HAS_MPI_SUPPORT",
    "HierarchicalCache",
    "HierarchicalTensor",
    "PAIRS",
    "SIGN",
    "apply_stencil",
    "budget_table",
    "build_stencil",
    "build_stencil_mt",
    "cache_disk_mb",
    "ccdptzyx2ccdxyzt",
    "ccdxyzt2ccdptzyx",
    "check_mpi_support",
    "disk_cache_bytes",
    "dump_env_h5",
    "fit_from_bench",
    "git_snapshot",
    "give_eo_mask",
    "give_grid_index",
    "give_grid_size",
    "give_null_vecs",
    "give_null_vecs_mt",
    "give_rank_minus",
    "give_rank_minus_minus",
    "give_rank_minus_plus",
    "give_rank_plus",
    "give_rank_plus_minus",
    "give_rank_plus_plus",
    "gpu_snapshot",
    "gpu_used_mb",
    "gridoooxyzt2hdf5oooxyzt",
    "hdf5oooxyzt2gridoooxyzt",
    "label_of",
    "load_dict_h5",
    "load_json",
    "load_tensor_h5",
    "local_orthogonalize",
    "local_xyzt2whole_xyzt",
    "matmul_cpu",
    "matmul_gpu",
    "mpl_cjk_setup",
    "norm",
    "np2torch_dtype",
    "nullvec_cache_dir",
    "offload_to_disk",
    "offload_to_ram",
    "oooxyzt2poooxyzt",
    "plot_bench_report",
    "plot_convergence",
    "plot_hotspot",
    "plot_rank_metric",
    "plot_speedup",
    "plot_sweep_curves",
    "plot_time",
    "poooxyzt2oooxyzt",
    "prime_factorization",
    "prolong",
    "psctzyx2scxyzt",
    "reload_to_vram",
    "restrict",
    "rss_kb",
    "rss_model",
    "save_dict_h5",
    "save_figure",
    "save_figure_variants",
    "save_tensor_h5",
    "scxyzt2psctzyx",
    "set_device",
    "short_label",
    "slice_dim",
    "slice_dim_dim",
    "slice_dim_none_dim",
    "style_axes",
    "to_contiguous_real",
    "torch2np_dtype",
    "torch2tl_dtype",
    "vdot",
    "vram_model",
    "vram_model_warm",
    "warp_size",
    "whole_xyzt2local_xyzt",
]
