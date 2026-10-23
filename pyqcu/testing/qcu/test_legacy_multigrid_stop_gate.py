"""Static contracts for the CUDA Legacy-MultiGrid convergence gate."""

from pathlib import Path


HEADER = (
    Path(__file__).resolve().parents[3]
    / "cpp/cuda/qcu/include/lattice_clover_multigrid.h"
)


def _function_body(signature: str) -> str:
    source = HEADER.read_text(encoding="utf-8")
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


def test_full_residual_gate_is_collective_and_authoritative():
    full_residual = _function_body("void compute_full_residual()")
    assert "if(rank" not in full_residual

    helper = _function_body("T compute_full_relative_residual(")
    assert "compute_full_residual();" in helper
    assert "fine_dot_host(r_full, r_full" in helper
    assert "full_rhs_norm2" in helper
    assert "if(rank" not in helper

    full_dot = _function_body("std::complex<T> fine_dot_host(")
    assert "MPI_Allreduce" in full_dot
    assert "if(rank" not in full_dot

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
