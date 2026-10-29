"""Static contracts for the CUDA Legacy-MultiGrid convergence gate."""

from pathlib import Path


HEADER = (
    Path(__file__).resolve().parents[3]
    / "cpp/cuda/qcu/include/lattice_clover_multigrid.h"
)
STRICT_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "cpp/cuda/qcu/src/apply_multigrid_strict.cu"
)


def _function_body_in(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for offset in range(brace, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1:offset]
    raise AssertionError(f"unterminated function body for {signature!r}")


def _function_body(signature: str) -> str:
    return _function_body_in(HEADER.read_text(encoding="utf-8"), signature)


def test_full_residual_gate_is_collective_and_authoritative():
    full_residual = _function_body("void compute_full_residual()")
    assert "if(rank" not in full_residual

    helper = _function_body("T compute_full_relative_residual(")
    assert "compute_full_residual();" in helper
    assert "fine_dot_host(r_full, r_full" in helper
    assert "full_rhs_norm2" in helper
    assert "if(rank" not in helper

    full_dot = _function_body("std::complex<T> fine_dot_host(")
    assert "mpi_reduce_complex<T>" in full_dot
    assert "if(rank" not in full_dot

    reduction = _function_body("std::complex<U> mpi_reduce_complex(")
    assert "MPI_Allreduce" in reduction

    run = _function_body("void run()")
    gate_start = run.index("if(rn2<=atol2*bnorm2)")
    gate_end = run.index("// ---- V-cycle coarse correction ----", gate_start)
    gate = run[gate_start:gate_end]
    gate_call = gate.index("compute_full_relative_residual(")
    assert "if(rank==0" not in gate[:gate_call]
    assert "full_rel<=atol" in gate
    assert "break;" in gate
    assert "near_tolerance" in run
    assert "16 * atol2 * bnorm2" in run


def test_iteration_limit_cannot_log_false_convergence():
    run = _function_body("void run()")
    assert "post_correction_norm2<rn2" in run
    assert "corrections_enabled=false;" in run
    assert "rejected and rolled back" in run
    assert "mg_gate_factor" not in run
    assert "const bool converged=" in run
    assert "FINAL STOP: reason=" in run
    assert "corrections_accepted=" in run
    assert "corrections_rejected=" in run
    assert "if(!converged)" in run
    assert "NOT CONVERGED: reason=" in run


def test_mpi_dual_dot_uses_one_four_scalar_collective():
    reducer = _function_body("void mpi_reduce_complex_pair(")
    assert reducer.count("MPI_Allreduce(") == 1
    assert "values, 4, mpi_real_type<U>()" in reducer
    assert "result0 = std::complex<U>(values[0], values[1])" in reducer
    assert "result1 = std::complex<U>(values[2], values[3])" in reducer

    fine_pair = _function_body("void dot_mpi_pair(")
    assert fine_pair.count("_cublasDot<T>") == 2
    assert fine_pair.count("mpi_reduce_complex_pair<T>(") == 1
    assert "cudaStreamSynchronize(set_ptr->streams[si0])" in fine_pair
    assert "cudaStreamSynchronize(set_ptr->streams[si1])" in fine_pair

    coarse_pair = _function_body("void coarse_dot_slot_pair(")
    assert coarse_pair.count("coarse_dot_dest(") == 2
    assert coarse_pair.count("mpi_reduce_complex_pair<T>(") == 1

    run = _function_body("void bistabcg_iter(")
    assert "QCU_MPI_DUAL_DOT_BASELINE" in run
    assert "dot_mpi_pair(st.t, st.s, st.t" in run


def test_pair_fusion_keeps_reduction_barriers_by_default():
    barrier = _function_body("void mpi_reduction_barrier()")
    assert "QCU_MPI_SKIP_REDUCTION_BARRIERS" in barrier
    assert "checkMpiErrors(MPI_Barrier(MPI_COMM_WORLD))" in barrier

    for signature in ("void dot_mpi(", "void dot_full_mpi("):
        reduction = _function_body(signature)
        assert "mpi_reduction_barrier();" in reduction
        assert "MPI_Barrier(MPI_COMM_WORLD)" not in reduction

    coarse_reduction = _function_body("void coarse_dot_slot(")
    assert "QCU_MPI_SKIP_REDUCTION_BARRIERS" in coarse_reduction
    assert "cudaStreamSynchronize(set_ptr->stream)" in coarse_reduction


def test_strict_pair_global_copy_keeps_blocking_baseline():
    source = STRICT_SOURCE.read_text(encoding="utf-8")
    body = _function_body_in(source, "void dot_pair_global_device(")
    assert "cudaMemcpy(" in body
    assert "cudaMemcpyAsync(" not in body
