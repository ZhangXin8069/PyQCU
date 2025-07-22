import torch
from typing import Callable, Union
import numpy as np


def give_null_vecs(
    null_vecs: torch.Tensor,
    matvec: Callable[[torch.Tensor], torch.Tensor],
    tol: float = 1e-7,
    verbose: bool = True
) -> torch.Tensor:
    """
    Generates orthonormal near-null space vectors for a linear operator.
    This function refines initial random vectors to become approximate null vectors
    (eigenvectors corresponding to near-zero eigenvalues) through iterative refinement
    and orthogonalization.
    
    Supports both real and complex matrices through unified complex arithmetic.
    Real matrices are automatically handled as complex matrices with zero imaginary parts.
    
    Args:
        null_vecs: Initial random vectors [dof, *dims]
        matvec: Matrix-vector multiplication operator A(x)
        tol: Tolerance for convergence
        verbose: Print progress information
        
    Returns:
        Orthonormal near-null space vectors in complex format
    """
    # Convert to complex type for unified processing
    device = null_vecs.device
    original_dtype = null_vecs.dtype
    
    # Always work in complex domain for unified arithmetic
    if not null_vecs.dtype.is_complex:
        if null_vecs.dtype == torch.float64:
            null_vecs = null_vecs.to(torch.complex128)
        else:
            null_vecs = null_vecs.to(torch.complex64)
    
    # Get shape information
    dof = null_vecs.shape[0]  # Number of null space vectors
    original_shape = null_vecs.shape[1:]  # Original vector shape
    
    # Flatten vectors for easier processing
    flat_vecs = null_vecs.view(dof, -1)  # [dof, total_dim]
    total_dim = flat_vecs.shape[1]
    
    # Initial orthogonalization using complex Gram-Schmidt
    flat_vecs = gram_schmidt(flat_vecs)
    
    # Iterative refinement parameters
    max_iter = 1000
    alpha = 0.1  # Step size parameter
    alpha_decay = 0.95  # Decay factor for adaptive step size
    alpha_min = 1e-4  # Minimum step size
    
    prev_max_residual = float('inf')
    stagnation_count = 0
    
    for iteration in range(max_iter):
        # Compute residuals A·v for each vector
        residuals = []
        for i in range(dof):
            vec_reshaped = flat_vecs[i].view(original_shape)
            
            # Convert to original dtype for matvec if input was real
            if not original_dtype.is_complex:
                vec_for_matvec = vec_reshaped.real.to(original_dtype)
            else:
                vec_for_matvec = vec_reshaped
            
            # Apply matvec and ensure result is complex
            residual = matvec(vec_for_matvec)
            if not residual.dtype.is_complex:
                residual = residual.to(torch.complex64)
            
            residuals.append(residual.view(-1))
        
        residuals = torch.stack(residuals)  # [dof, total_dim]
        
        # Compute residual norms (magnitude for complex vectors)
        res_norms = torch.norm(residuals, dim=1)  # L2 norm handles complex automatically
        max_residual = torch.max(res_norms).item()
        avg_residual = torch.mean(res_norms).item()
        
        if verbose and (iteration % 50 == 0 or max_residual < tol):
            print(f"Iteration {iteration:4d}: Max residual = {max_residual:.6e}, "
                  f"Avg residual = {avg_residual:.6e}, Step size = {alpha:.6f}")
        
        # Check convergence
        if max_residual < tol:
            if verbose:
                print(f"Converged after {iteration} iterations")
            break
        
        # Adaptive step size control
        if max_residual >= prev_max_residual * 0.99:
            stagnation_count += 1
            if stagnation_count > 10:
                alpha = max(alpha * alpha_decay, alpha_min)
                stagnation_count = 0
        else:
            stagnation_count = 0
            alpha = min(alpha * 1.02, 0.5)  # Increase step size but cap it
        
        # Iterative improvement using inverse power method
        # For finding null vectors: v_new = v - α * A^H(A(v))
        for i in range(dof):
            residual_reshaped = residuals[i].view(original_shape)
            
            # Convert residual to appropriate dtype for matvec
            if not original_dtype.is_complex:
                residual_for_matvec = residual_reshaped.real.to(original_dtype)
            else:
                residual_for_matvec = torch.conj(residual_reshaped)
            
            # Compute A^H(residual) - adjoint operation
            # For symmetric matrices: A^H = A^T (conjugate transpose)
            # Here we approximate using the matvec itself as a simplification
            gradient = matvec(residual_for_matvec)
            if not gradient.dtype.is_complex:
                gradient = gradient.to(torch.complex64)
            
            # Update vector: v = v - α * gradient
            flat_vecs[i] = flat_vecs[i] - alpha * gradient.view(-1)
        
        # Re-orthogonalize after each iteration
        flat_vecs = gram_schmidt(flat_vecs)
        
        prev_max_residual = max_residual
    
    else:
        if verbose:
            print(f"Max iterations ({max_iter}) reached. Final residual: {max_residual:.6e}")
    
    # Final orthogonalization check
    flat_vecs = gram_schmidt(flat_vecs)
    
    # Restore original shape
    result_vecs = flat_vecs.view(dof, *original_shape)
    
    if verbose:
        print(f"\nFinal orthogonality check:")
        check_orthogonality(result_vecs)
        print(f"\nFinal residual check:")
        for i in range(dof):
            vec = result_vecs[i]
            
            # Convert vec to appropriate dtype for matvec
            if not original_dtype.is_complex:
                vec_for_matvec = vec.real.to(original_dtype) if vec.dtype.is_complex else vec
            else:
                vec_for_matvec = vec
                
            residual = matvec(vec_for_matvec)
            if not residual.dtype.is_complex:
                residual = residual.to(torch.complex64)
            residual_norm = torch.norm(residual).item()
            print(f"  Vector {i}: ||A*v|| = {residual_norm:.6e}")
    
    # Convert back to original dtype if input was real
    if not original_dtype.is_complex:
        result_vecs = result_vecs.real.to(original_dtype)
    
    return result_vecs


def gram_schmidt(vectors: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Complex Gram-Schmidt orthogonalization with numerical stability.
    Works for both real and complex vectors through unified complex arithmetic.
    
    Args:
        vectors: [num_vecs, dim] complex tensor
        eps: Small value to prevent division by zero
    
    Returns:
        Orthonormal complex vectors
    """
    num_vecs, dim = vectors.shape
    orthogonal_vecs = torch.zeros_like(vectors)
    
    for i in range(num_vecs):
        vec = vectors[i].clone()
        
        # Remove projections onto previous orthogonal vectors
        for j in range(i):
            # Complex inner product: <u,v> = u^H * v = sum(conj(u) * v)
            proj_coeff = torch.sum(torch.conj(orthogonal_vecs[j]) * vec)
            vec = vec - proj_coeff * orthogonal_vecs[j]
        
        # Normalize using complex norm
        norm = torch.sqrt(torch.sum(torch.conj(vec) * vec).real)
        
        if norm > eps:
            orthogonal_vecs[i] = vec / norm
        else:
            # Generate new random vector if current one is nearly zero
            if vectors.dtype == torch.complex64:
                random_vec = torch.randn(dim, dtype=torch.complex64, device=vectors.device)
            elif vectors.dtype == torch.complex128:
                random_vec = torch.randn(dim, dtype=torch.complex128, device=vectors.device)
            else:
                # Fallback for other complex types
                random_vec = torch.randn(dim, dtype=vectors.dtype, device=vectors.device)
            
            # Remove projections from existing vectors
            for j in range(i):
                proj_coeff = torch.sum(torch.conj(orthogonal_vecs[j]) * random_vec)
                random_vec = random_vec - proj_coeff * orthogonal_vecs[j]
            
            norm_new = torch.sqrt(torch.sum(torch.conj(random_vec) * random_vec).real)
            orthogonal_vecs[i] = random_vec / norm_new if norm_new > eps else random_vec
    
    return orthogonal_vecs


def check_orthogonality(vectors: torch.Tensor, threshold: float = 1e-6):
    """
    Check orthogonality of vector set using complex inner products.
    
    Args:
        vectors: Input vectors [num_vecs, *dims]
        threshold: Threshold for considering vectors as orthogonal
    """
    flat_vecs = vectors.view(vectors.shape[0], -1)
    num_vecs = flat_vecs.shape[0]
    
    print(f"Orthogonality matrix (should be identity):")
    max_off_diagonal = 0.0
    
    for i in range(num_vecs):
        row_str = ""
        for j in range(num_vecs):
            # Complex inner product
            inner_product = torch.sum(torch.conj(flat_vecs[i]) * flat_vecs[j])
            
            if vectors.dtype.is_complex:
                magnitude = abs(inner_product.item())
                row_str += f"{magnitude:8.4f} "
            else:
                value = inner_product.real.item()
                row_str += f"{value:8.4f} "
            
            # Track maximum off-diagonal element
            if i != j:
                max_off_diagonal = max(max_off_diagonal, abs(inner_product.item()))
        
        print(f"  {row_str}")
    
    if max_off_diagonal < threshold:
        print(f"✓ Vectors are orthogonal (max off-diagonal: {max_off_diagonal:.6e})")
    else:
        print(f"✗ Vectors may not be orthogonal (max off-diagonal: {max_off_diagonal:.6e})")


# ============================================================================
# COMPREHENSIVE EXAMPLES
# ============================================================================

def create_test_matrices():
    """Create various test matrices for demonstration."""
    
    # 1. Real symmetric positive definite matrix (smallest eigenvalue near zero)
    real_spd = torch.tensor([
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 0.5],
        [1.0, 0.5, 2.1]
    ], dtype=torch.float32)
    
    # 2. Real nearly singular matrix
    real_singular = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0001],  # Nearly linearly dependent
        [0.5, 1.0, 1.5]
    ], dtype=torch.float32)
    
    # 3. Complex Hermitian matrix
    complex_hermitian = torch.tensor([
        [3.0+0.0j, 1.0-1.0j, 0.5+0.5j],
        [1.0+1.0j, 2.0+0.0j, -0.3-0.7j],
        [0.5-0.5j, -0.3+0.7j, 1.5+0.0j]
    ], dtype=torch.complex64)
    
    # 4. Complex non-Hermitian matrix
    complex_general = torch.tensor([
        [2.0+1.0j, 1.0-0.5j, 0.3+0.2j],
        [0.5+1.5j, 1.8-0.3j, -0.1+0.8j],
        [-0.2+0.1j, 0.7-0.4j, 1.2+0.6j]
    ], dtype=torch.complex64)
    
    # 5. Large sparse-like matrix (represented as dense for simplicity)
    n = 50
    large_matrix = torch.eye(n, dtype=torch.float32) * 0.01  # Very small eigenvalues
    for i in range(n-1):
        large_matrix[i, i+1] = 0.1
        large_matrix[i+1, i] = 0.1
    
    return {
        'real_spd': real_spd,
        'real_singular': real_singular, 
        'complex_hermitian': complex_hermitian,
        'complex_general': complex_general,
        'large_matrix': large_matrix
    }


def run_comprehensive_examples():
    """Run comprehensive examples demonstrating the null space solver."""
    
    test_matrices = create_test_matrices()
    
    print("="*80)
    print("COMPREHENSIVE NULL SPACE SOLVER EXAMPLES")
    print("="*80)
    
    # Example 1: Real symmetric positive definite matrix
    print("\n1. REAL SYMMETRIC POSITIVE DEFINITE MATRIX")
    print("-" * 50)
    
    A_real = test_matrices['real_spd']
    print(f"Matrix shape: {A_real.shape}")
    print(f"Matrix:\n{A_real}")
    
    def matvec_real_spd(v):
        return A_real @ v.flatten()
    
    # Initialize 2 random vectors
    initial_vecs = torch.randn(2, 3, dtype=torch.float32)
    result_vecs = give_null_vecs(initial_vecs, matvec_real_spd, tol=1e-8, verbose=True)
    
    print(f"Found {result_vecs.shape[0]} null space vectors:")
    for i, vec in enumerate(result_vecs):
        print(f"  Vector {i}: {vec.numpy()}")
    
    # Example 2: Nearly singular real matrix
    print(f"\n{'='*80}")
    print("2. NEARLY SINGULAR REAL MATRIX") 
    print("-" * 50)
    
    A_singular = test_matrices['real_singular']
    print(f"Matrix:\n{A_singular}")
    
    def matvec_singular(v):
        return A_singular @ v.flatten()
    
    initial_vecs = torch.randn(1, 3, dtype=torch.float32)
    result_vecs = give_null_vecs(initial_vecs, matvec_singular, tol=1e-6, verbose=True)
    
    # Example 3: Complex Hermitian matrix
    print(f"\n{'='*80}")
    print("3. COMPLEX HERMITIAN MATRIX")
    print("-" * 50)
    
    A_hermitian = test_matrices['complex_hermitian']
    print(f"Matrix:\n{A_hermitian}")
    
    def matvec_hermitian(v):
        return A_hermitian @ v.flatten()
    
    initial_vecs = torch.randn(2, 3, dtype=torch.complex64)  
    result_vecs = give_null_vecs(initial_vecs, matvec_hermitian, tol=1e-7, verbose=True)
    
    # Example 4: Complex general matrix
    print(f"\n{'='*80}")
    print("4. COMPLEX GENERAL MATRIX")
    print("-" * 50)
    
    A_complex = test_matrices['complex_general']
    print(f"Matrix:\n{A_complex}")
    
    def matvec_complex(v):
        return A_complex @ v.flatten()
    
    initial_vecs = torch.randn(1, 3, dtype=torch.complex64)
    result_vecs = give_null_vecs(initial_vecs, matvec_complex, tol=1e-6, verbose=True)
    
    # Example 5: Large matrix demonstration
    print(f"\n{'='*80}")
    print("5. LARGE MATRIX (50x50) - PERFORMANCE TEST")
    print("-" * 50)
    
    A_large = test_matrices['large_matrix']
    print(f"Matrix shape: {A_large.shape}")
    print(f"Matrix condition number (approx): {torch.linalg.cond(A_large):.2e}")
    
    def matvec_large(v):
        return A_large @ v.flatten()
    
    # Find 3 null vectors for large matrix
    initial_vecs = torch.randn(3, 50, dtype=torch.float32)
    
    import time
    start_time = time.time()
    result_vecs = give_null_vecs(initial_vecs, matvec_large, tol=1e-6, verbose=True)
    end_time = time.time()
    
    print(f"Computation time: {end_time - start_time:.2f} seconds")
    print(f"Found {result_vecs.shape[0]} vectors of dimension {result_vecs.shape[1]}")
    
    # Example 6: Mixed precision demonstration
    print(f"\n{'='*80}")
    print("6. MIXED PRECISION (FLOAT64) DEMONSTRATION")
    print("-" * 50)
    
    # High precision computation
    A_hp = test_matrices['real_spd'].to(torch.float64)
    
    def matvec_hp(v):
        return A_hp @ v.flatten()
    
    initial_vecs_hp = torch.randn(1, 3, dtype=torch.float64)
    result_vecs_hp = give_null_vecs(initial_vecs_hp, matvec_hp, tol=1e-12, verbose=True)
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*80)


def benchmark_performance():
    """Benchmark performance for different matrix sizes."""
    
    print("\nPERFORMANCE BENCHMARK")
    print("-" * 50)
    
    import time
    sizes = [10, 25, 50, 100]
    
    for n in sizes:
        # Create test matrix
        A = torch.eye(n, dtype=torch.float32) * 0.1
        A += torch.randn(n, n, dtype=torch.float32) * 0.01  # Add small random perturbation
        
        def matvec_bench(v):
            return A @ v.flatten()
        
        initial_vecs = torch.randn(2, n, dtype=torch.float32)
        
        start_time = time.time()
        result_vecs = give_null_vecs(initial_vecs, matvec_bench, tol=1e-6, verbose=False)
        end_time = time.time()
        
        # Verify results
        max_residual = 0.0
        for vec in result_vecs:
            # Convert vec to appropriate dtype for matvec
            if not vec.dtype.is_complex and A.dtype == torch.float32:
                vec_for_matvec = vec
            else:
                vec_for_matvec = vec.real if vec.dtype.is_complex else vec
                
            residual_norm = torch.norm(matvec_bench(vec_for_matvec)).item()
            max_residual = max(max_residual, residual_norm)
        
        print(f"Size {n:3d}x{n:3d}: {end_time-start_time:6.2f}s, "
              f"Max residual: {max_residual:.2e}")


if __name__ == "__main__":
    # Run comprehensive examples
    run_comprehensive_examples()
    
    # Run performance benchmark
    benchmark_performance()