import pytest
import torch
from pyqcu.tools import Eexyzt_exyzt2Exyzt, torch2np_dtype, torch2tl_dtype, torch_complex2real_dtype

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inputs(E, e, x, y, z, t, dtype, device, seed=42):
    """
    Create reproducible random tensors for Eexyzt and exyzt.

    For complex dtypes, real and imaginary parts are sampled independently
    from a standard normal distribution.

    Parameters
    ----------
    E, e, x, y, z, t : int          — dimension sizes
    dtype             : torch.dtype  — real or complex dtype
    device            : str          — 'cpu' or 'cuda'
    seed              : int          — RNG seed for reproducibility
    """
    torch.manual_seed(seed)
    if dtype in (torch.complex32, torch.complex64, torch.complex128):
        # Determine the matching real dtype for sampling
        real_dtype = torch_complex2real_dtype[dtype]
        Eexyzt = torch.randn(E, e, x, y, z, t, dtype=real_dtype, device=device)
        exyzt = torch.randn(e, x, y, z, t, dtype=real_dtype, device=device)
        Eexyzt = torch.view_as_complex(
            torch.stack([Eexyzt,
                         torch.randn_like(Eexyzt)], dim=-1).contiguous()
        )
        exyzt = torch.view_as_complex(
            torch.stack([exyzt,
                         torch.randn_like(exyzt)], dim=-1).contiguous()
        )
    else:
        Eexyzt = torch.randn(E, e, x, y, z, t, dtype=dtype, device=device)
        exyzt = torch.randn(e, x, y, z, t, dtype=dtype, device=device)
    return Eexyzt, exyzt


def cpu_reference(Eexyzt: torch.Tensor, exyzt: torch.Tensor) -> torch.Tensor:
    """
    Ground-truth implementation: always runs on CPU via torch.einsum.
    Inputs are cast to float32/complex64 before contraction to maximise
    numerical precision of the reference, then cast back to the original dtype.
    """
    orig_dtype = Eexyzt.dtype
    is_complex = orig_dtype in torch_complex2real_dtype

    if is_complex:
        # Upcast to complex128 for a high-precision reference
        ref_dtype = torch.complex128
    else:
        ref_dtype = torch.float64

    result = torch.einsum(
        'Eexyzt,exyzt->Exyzt',
        Eexyzt.cpu().to(ref_dtype),
        exyzt.cpu().to(ref_dtype),
    )
    return result.to(orig_dtype)


def assert_close(cuda_out: torch.Tensor,
                 cpu_ref:  torch.Tensor,
                 dtype:    torch.dtype,
                 label:    str = ""):
    """
    Compare CUDA kernel output against the CPU reference.

    Tolerances are relaxed for float16 / complex32 because those types
    have limited precision (≈3 decimal digits).

    Parameters
    ----------
    cuda_out : tensor on any device — result from the CUDA kernel
    cpu_ref  : tensor on CPU        — result from cpu_reference()
    dtype    : original input dtype — used to select tolerances
    label    : human-readable test label for error messages
    """
    # Tolerances indexed by the real component of the dtype
    _atol = {
        torch.float16: 1e-2,
        torch.float32: 1e-4,
        torch.float64: 1e-9,
    }
    _rtol = {
        torch.float16: 1e-2,
        torch.float32: 1e-4,
        torch.float64: 1e-9,
    }

    real_dtype = torch_complex2real_dtype.get(dtype, dtype)
    atol = _atol[real_dtype]
    rtol = _rtol[real_dtype]

    # Move both tensors to CPU and upcast to float64/complex128 for comparison
    a = cuda_out.cpu()
    b = cpu_ref.cpu()

    if not torch.allclose(a.to(torch.complex128 if a.is_complex() else torch.float64),
                          b.to(torch.complex128 if b.is_complex()
                               else torch.float64),
                          atol=atol, rtol=rtol):
        diff = (a.to(torch.float64) - b.to(torch.float64)).abs()
        raise AssertionError(
            f"[{label}] CUDA vs CPU mismatch  "
            f"max_abs_err={diff.max().item():.3e}  "
            f"mean_abs_err={diff.mean().item():.3e}  "
            f"atol={atol}  rtol={rtol}"
        )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# (E, e, x, y, z, t) shape fixtures
SHAPES = [
    (2,  2,  2,  2,  2,  2),   # minimal
    (4,  4,  4,  4,  2,  2),   # small square spatial
    (8,  4,  3,  5,  2,  4),   # non-power-of-two spatial
    (16, 8,  4,  4,  4,  4),   # medium
    (1,  1,  1,  1,  1,  1),   # degenerate singletons
    (4,  4,  8,  8,  1,  1),   # flat spatial (z=t=1)
    (8,  1,  4,  4,  2,  2),   # single e element
]

REAL_DTYPES = [torch.float16, torch.float32, torch.float64]
COMPLEX_DTYPES = [torch.complex64, torch.complex128]
ALL_DTYPES = REAL_DTYPES + COMPLEX_DTYPES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAvsCPU:
    """
    Compare the CUDA kernel path against the CPU einsum reference for every
    combination of shape and dtype.
    """

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", REAL_DTYPES)
    def test_real_dtypes(self, shape, dtype):
        """
        Real-valued inputs: CUDA kernel result must match the CPU reference
        within dtype-appropriate tolerances.
        """
        E, e, x, y, z, t = shape
        Eexyzt_cpu, exyzt_cpu = make_inputs(E, e, x, y, z, t, dtype, 'cpu')
        Eexyzt_gpu = Eexyzt_cpu.cuda()
        exyzt_gpu = exyzt_cpu.cuda()

        cuda_out = Eexyzt_exyzt2Exyzt(Eexyzt_gpu, exyzt_gpu)
        cpu_ref = cpu_reference(Eexyzt_cpu, exyzt_cpu)

        label = f"real {dtype} shape={shape}"
        assert_close(cuda_out, cpu_ref, dtype, label)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
    def test_complex_dtypes(self, shape, dtype):
        """
        Complex-valued inputs: CUDA kernel result (four real calls + recombine)
        must match the CPU einsum reference.
        """
        E, e, x, y, z, t = shape
        Eexyzt_cpu, exyzt_cpu = make_inputs(E, e, x, y, z, t, dtype, 'cpu')
        Eexyzt_gpu = Eexyzt_cpu.cuda()
        exyzt_gpu = exyzt_cpu.cuda()

        cuda_out = Eexyzt_exyzt2Exyzt(Eexyzt_gpu, exyzt_gpu)
        cpu_ref = cpu_reference(Eexyzt_cpu, exyzt_cpu)

        label = f"complex {dtype} shape={shape}"
        assert_close(cuda_out, cpu_ref, dtype, label)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_output_shape_and_dtype(self, dtype):
        """
        Regardless of dtype, the output shape must be (E, x, y, z, t) and
        the output dtype must match the input dtype.
        """
        E, e, x, y, z, t = 4, 4, 4, 4, 2, 2
        Eexyzt, exyzt = make_inputs(E, e, x, y, z, t, dtype, 'cuda')

        out = Eexyzt_exyzt2Exyzt(Eexyzt, exyzt)

        assert out.shape == (E, x, y, z, t), (
            f"dtype={dtype}: expected shape {(E,x,y,z,t)}, got {out.shape}"
        )
        assert out.dtype == dtype, (
            f"dtype={dtype}: expected output dtype {dtype}, got {out.dtype}"
        )

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_cpu_and_cuda_agree_on_same_data(self, dtype):
        """
        Run both the CPU path and the CUDA path on identical data and confirm
        they agree.  This is the primary end-to-end correctness check.
        """
        E, e, x, y, z, t = 8, 4, 4, 4, 2, 2
        Eexyzt_cpu, exyzt_cpu = make_inputs(E, e, x, y, z, t, dtype, 'cpu')

        # CPU path
        cpu_out = Eexyzt_exyzt2Exyzt(Eexyzt_cpu, exyzt_cpu)

        # CUDA path
        cuda_out = Eexyzt_exyzt2Exyzt(
            Eexyzt_cpu.cuda(), exyzt_cpu.cuda()
        )

        label = f"end-to-end {dtype}"
        assert_close(cuda_out, cpu_out, dtype, label)

    def test_zero_inputs(self):
        """
        Contracting any tensor with an all-zero tensor must yield all zeros.
        """
        E, e, x, y, z, t = 4, 4, 4, 4, 2, 2
        for dtype in ALL_DTYPES:
            Eexyzt = torch.zeros(E, e, x, y, z, t, dtype=dtype, device='cuda')
            exyzt = torch.zeros(e, x, y, z, t, dtype=dtype, device='cuda')
            out = Eexyzt_exyzt2Exyzt(Eexyzt, exyzt)
            assert torch.all(
                out == 0), f"dtype={dtype}: expected all-zero output"

    def test_dtype_mismatch_raises(self):
        """
        Passing operands with different dtypes must raise TypeError.
        """
        Eexyzt = torch.randn(4, 4, 4, 4, 2, 2,
                             dtype=torch.float32, device='cuda')
        exyzt = torch.randn(4, 4, 4, 4, 2, 2,
                            dtype=torch.float16, device='cuda')
        with pytest.raises(TypeError):
            Eexyzt_exyzt2Exyzt(Eexyzt, exyzt)
