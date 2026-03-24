import torch
import numpy as np
from typing import Optional, Tuple

class StoutSmearing:
    """
    Stout smearing implementation for lattice QCD using PyTorch.
    
    Features:
    1. Optimized data layout: (color1, color2, direction, x, y, z, t)
    2. Support for both Cayley-Hamilton and direct eigendecomposition methods
    3. Batch processing for all lattice sites
    4. GPU acceleration support
    5. Mathematical clarity with explicit formulas
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Stout smearing class.
        
        Args:
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.Nc = 3  # SU(3) group
        self.Nd = 4  # 4 dimensions (x, y, z, t)
        
    def _reorder_u(self, U: torch.Tensor) -> torch.Tensor:
        """
        Reorder U tensor to optimized layout: (color1, color2, direction, x, y, z, t)
        
        Args:
            U: Input gauge field with shape (Nd, Lx, Ly, Lz, Lt, Nc, Nc)
            
        Returns:
            Reordered tensor with shape (Nc, Nc, Nd, Lx, Ly, Lz, Lt)
        """
        # Original: (direction, x, y, z, t, color1, color2)
        # New: (color1, color2, direction, x, y, z, t)
        return U.permute(5, 6, 0, 1, 2, 3, 4).contiguous()
    
    def _restore_u(self, U: torch.Tensor) -> torch.Tensor:
        """
        Restore U tensor to original layout.
        
        Args:
            U: Tensor with shape (Nc, Nc, Nd, Lx, Ly, Lz, Lt)
            
        Returns:
            Tensor with shape (Nd, Lx, Ly, Lz, Lt, Nc, Nc)
        """
        return U.permute(2, 3, 4, 5, 6, 0, 1).contiguous()
    
    def _compute_staples(self, U: torch.Tensor) -> torch.Tensor:
        """
        Compute staples C_mu(x) for all directions and lattice sites.
        
        Args:
            U: Gauge field with shape (Nc, Nc, Nd, Lx, Ly, Lz, Lt)
            
        Returns:
            Staples tensor with shape (Nc, Nc, Nd, Lx, Ly, Lz, Lt)
        """
        Nc, Nd = self.Nc, self.Nd
        Lx, Ly, Lz, Lt = U.shape[3:]
        
        # Get U and U_dagger
        U_dag = U.transpose(0, 1).conj()  # shape: (Nc, Nc, Nd, ...)
        
        # Initialize staples
        staples = torch.zeros_like(U)
        
        # For each direction mu (0, 1, 2, 3 corresponding to x, y, z, t)
        for mu in range(Nd):
            # Sum over nu != mu
            for nu in range(Nd):
                if mu == nu:
                    continue
                
                # Positive staple: U_nu(x) * U_mu(x+nu) * U_nu^dagger(x+mu)
                # Roll indices for spatial dimensions: dimensions 3, 4, 5, 6 correspond to x, y, z, t
                
                # Extract U_nu and U_mu for all lattice sites
                # U_nu has shape (Nc, Nc, 1, Lx, Ly, Lz, Lt)
                U_nu = U[..., nu:nu+1, :, :, :, :]
                
                # U_mu(x+nu): roll in the nu-th spatial dimension
                # Determine which dimension to roll based on nu
                U_mu = U[..., mu:mu+1, :, :, :, :]
                
                # Roll U_mu forward in nu direction
                if nu == 0:  # x direction
                    U_mu_forward = torch.roll(U_mu, shifts=-1, dims=3)
                elif nu == 1:  # y direction
                    U_mu_forward = torch.roll(U_mu, shifts=-1, dims=4)
                elif nu == 2:  # z direction
                    U_mu_forward = torch.roll(U_mu, shifts=-1, dims=5)
                else:  # t direction (nu == 3)
                    U_mu_forward = torch.roll(U_mu, shifts=-1, dims=6)
                
                # U_nu^dagger(x+mu): roll in the mu-th spatial dimension
                U_nu_dag = U_dag[..., nu:nu+1, :, :, :, :]
                if mu == 0:  # x direction
                    U_nu_dag_forward = torch.roll(U_nu_dag, shifts=-1, dims=3)
                elif mu == 1:  # y direction
                    U_nu_dag_forward = torch.roll(U_nu_dag, shifts=-1, dims=4)
                elif mu == 2:  # z direction
                    U_nu_dag_forward = torch.roll(U_nu_dag, shifts=-1, dims=5)
                else:  # t direction (mu == 3)
                    U_nu_dag_forward = torch.roll(U_nu_dag, shifts=-1, dims=6)
                
                # Compute positive staple
                pos_staple = torch.einsum('ab...,bc...,cd...->ad...', 
                                         U_nu.squeeze(2), 
                                         U_mu_forward.squeeze(2), 
                                         U_nu_dag_forward.squeeze(2))
                
                # Negative staple: U_nu^dagger(x-nu) * U_mu(x-nu) * U_nu(x-nu+mu)
                # U_nu^dagger(x-nu): roll backward in nu direction
                if nu == 0:
                    U_nu_dag_backward = torch.roll(U_nu_dag, shifts=1, dims=3)
                elif nu == 1:
                    U_nu_dag_backward = torch.roll(U_nu_dag, shifts=1, dims=4)
                elif nu == 2:
                    U_nu_dag_backward = torch.roll(U_nu_dag, shifts=1, dims=5)
                else:  # nu == 3
                    U_nu_dag_backward = torch.roll(U_nu_dag, shifts=1, dims=6)
                
                # U_mu(x-nu): roll backward in nu direction
                if nu == 0:
                    U_mu_backward = torch.roll(U_mu, shifts=1, dims=3)
                elif nu == 1:
                    U_mu_backward = torch.roll(U_mu, shifts=1, dims=4)
                elif nu == 2:
                    U_mu_backward = torch.roll(U_mu, shifts=1, dims=5)
                else:  # nu == 3
                    U_mu_backward = torch.roll(U_mu, shifts=1, dims=6)
                
                # U_nu(x-nu+mu): roll backward in nu, then forward in mu
                U_nu_original = U[..., nu:nu+1, :, :, :, :]
                if nu == 0:
                    U_nu_backward_temp = torch.roll(U_nu_original, shifts=1, dims=3)
                elif nu == 1:
                    U_nu_backward_temp = torch.roll(U_nu_original, shifts=1, dims=4)
                elif nu == 2:
                    U_nu_backward_temp = torch.roll(U_nu_original, shifts=1, dims=5)
                else:  # nu == 3
                    U_nu_backward_temp = torch.roll(U_nu_original, shifts=1, dims=6)
                
                if mu == 0:
                    U_nu_backward_forward = torch.roll(U_nu_backward_temp, shifts=-1, dims=3)
                elif mu == 1:
                    U_nu_backward_forward = torch.roll(U_nu_backward_temp, shifts=-1, dims=4)
                elif mu == 2:
                    U_nu_backward_forward = torch.roll(U_nu_backward_temp, shifts=-1, dims=5)
                else:  # mu == 3
                    U_nu_backward_forward = torch.roll(U_nu_backward_temp, shifts=-1, dims=6)
                
                # Compute negative staple
                neg_staple = torch.einsum('ab...,bc...,cd...->ad...',
                                         U_nu_dag_backward.squeeze(2),
                                         U_mu_backward.squeeze(2),
                                         U_nu_backward_forward.squeeze(2))
                
                # Add to total staples for direction mu
                staples[..., mu, :, :, :, :] += pos_staple + neg_staple
        
        return staples
    
    def _compute_q_matrix(self, U: torch.Tensor, staples: torch.Tensor, rho: float) -> torch.Tensor:
        """
        Compute Q_mu(x) matrix from staples.
        
        Q_mu(x) = (i/2) * (Omega_mu^dagger - Omega_mu) - (i/(2N)) * Tr(Omega_mu^dagger - Omega_mu) * I
        
        where Omega_mu = rho * C_mu(x) * U_mu^dagger(x)
        
        Args:
            U: Gauge field
            staples: Computed staples C_mu(x)
            rho: Smearing parameter
            
        Returns:
            Q matrix with shape (Nc, Nc, Nd, ...)
        """
        Nc = self.Nc
        U_dag = U.transpose(0, 1).conj()
        
        # Omega_mu = rho * C_mu * U_mu^dagger
        Omega = rho * torch.einsum('ab...,bc...->ac...', staples, U_dag)
        
        # Q_mu = (i/2) * (Omega_mu^dagger - Omega_mu)
        Omega_dag = Omega.transpose(0, 1).conj()
        Q = 0.5j * (Omega_dag - Omega)
        
        # Make Q traceless: subtract (i/(2N)) * Tr(Omega_dag - Omega) * I
        # Trace over color indices
        trace = torch.einsum('aa...->...', Omega_dag - Omega)
        trace_correction = (1j / (2 * Nc)) * trace
        
        # Expand dimensions for broadcasting
        trace_correction = trace_correction.unsqueeze(0).unsqueeze(0)
        
        # Create identity matrix for each lattice site
        identity = torch.eye(Nc, device=self.device, dtype=torch.complex64)
        identity = identity.view(Nc, Nc, 1, 1, 1, 1, 1)
        identity = identity.expand(-1, -1, self.Nd, *U.shape[3:])
        
        # Apply correction
        Q = Q - trace_correction * identity
        
        return Q
    
    def _exp_iq_cayley_hamilton(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Compute exp(iQ) using Cayley-Hamilton theorem for 3x3 matrices.
        
        For a 3x3 traceless Hermitian matrix Q, we have:
        exp(iQ) = f0 * I + f1 * Q + f2 * Q^2
        
        where f0, f1, f2 are functions of c0 = det(Q) and c1 = (1/2)Tr(Q^2).
        
        Args:
            Q: Traceless Hermitian matrix with shape (Nc, Nc, Nd, ...)
            
        Returns:
            exp(iQ) with same shape as Q
        """
        Nc = self.Nc
        
        # Compute Q^2
        Q_sq = torch.einsum('ab...,bc...->ac...', Q, Q)
        
        # Compute invariants
        # c0 = det(Q) = (1/3)Tr(Q^3)
        Q_cubed = torch.einsum('ab...,bc...,cd...->ad...', Q, Q, Q)
        c0 = torch.einsum('aa...->...', Q_cubed).real / 3.0
        
        # c1 = (1/2)Tr(Q^2)
        c1 = torch.einsum('aa...->...', Q_sq).real / 2.0
        
        # Handle small c1 case to avoid division by zero
        mask_small_c1 = c1 < 1e-12
        c1_safe = torch.where(mask_small_c1, torch.ones_like(c1) * 1e-12, c1)
        
        # Maximum possible |c0|
        c0_max = 2.0 * (c1_safe / 3.0) ** 1.5
        
        # Compute theta
        ratio = c0 / c0_max
        ratio = torch.clamp(ratio, -1.0, 1.0)
        theta = torch.acos(ratio)
        
        # For very small c1, set u and w to 0
        u = torch.zeros_like(c1)
        w = torch.zeros_like(c1)
        
        # Only compute for non-small c1
        mask_non_small = ~mask_small_c1
        if mask_non_small.any():
            sqrt_c1_over_3 = torch.sqrt(c1_safe[mask_non_small] / 3.0)
            u[mask_non_small] = sqrt_c1_over_3 * torch.cos(theta[mask_non_small] / 3.0)
            w[mask_non_small] = torch.sqrt(c1_safe[mask_non_small]) * torch.sin(theta[mask_non_small] / 3.0)
        
        u_sq = u ** 2
        w_sq = w ** 2
        
        # Precompute trigonometric functions
        cos_u = torch.cos(u)
        sin_u = torch.sin(u)
        cos_2u = torch.cos(2.0 * u)
        sin_2u = torch.sin(2.0 * u)
        cos_w = torch.cos(w)
        
        # Compute sinc(w) = sin(w)/w with Taylor expansion for small w
        sinc_w = torch.ones_like(w)
        mask_large = torch.abs(w) > 0.05
        if mask_large.any():
            sinc_w[mask_large] = torch.sin(w[mask_large]) / w[mask_large]
        
        # For small w, use Taylor expansion up to w^8
        mask_small_w = ~mask_large
        if mask_small_w.any():
            w_small = w[mask_small_w]
            w_sq_small = w_sq[mask_small_w]
            sinc_w[mask_small_w] = 1.0 - w_sq_small/6.0 * (
                1.0 - w_sq_small/20.0 * (
                    1.0 - w_sq_small/42.0 * (
                        1.0 - w_sq_small/72.0
                    )
                )
            )
        
        # Common denominator
        denom = 1.0 / (9.0 * u_sq - w_sq)
        # Handle case where denominator is 0 (when u=w=0)
        denom = torch.where(torch.abs(9.0 * u_sq - w_sq) < 1e-12, torch.ones_like(denom), denom)
        
        # Compute f coefficients
        f0_real = ((u_sq - w_sq) * cos_2u + 
                  8.0 * u_sq * cos_u * cos_w +
                  2.0 * u * (3.0 * u_sq + w_sq) * sin_u * sinc_w) * denom
        
        f0_imag = ((u_sq - w_sq) * sin_2u -
                  8.0 * u_sq * sin_u * cos_w +
                  2.0 * u * (3.0 * u_sq + w_sq) * cos_u * sinc_w) * denom
        
        f1_real = (2.0 * u * cos_2u -
                  2.0 * u * cos_u * cos_w +
                  (3.0 * u_sq - w_sq) * sin_u * sinc_w) * denom
        
        f1_imag = (2.0 * u * sin_2u +
                  2.0 * u * sin_u * cos_w +
                  (3.0 * u_sq - w_sq) * cos_u * sinc_w) * denom
        
        f2_real = (cos_2u - cos_u * cos_w - 3.0 * u * sin_u * sinc_w) * denom
        f2_imag = (sin_2u + sin_u * cos_w - 3.0 * u * cos_u * sinc_w) * denom
        
        # Handle parity (sign of c0)
        parity = c0 < 0
        if parity.any():
            f0_imag[parity] *= -1
            f1_real[parity] *= -1
            f2_imag[parity] *= -1
        
        # For very small c1, exp(iQ) ≈ I + iQ - Q^2/2
        if mask_small_c1.any():
            f0_real[mask_small_c1] = 1.0
            f0_imag[mask_small_c1] = 0.0
            f1_real[mask_small_c1] = 0.0
            f1_imag[mask_small_c1] = 1.0
            f2_real[mask_small_c1] = -0.5
            f2_imag[mask_small_c1] = 0.0
        
        # Combine real and imaginary parts
        f0 = f0_real + 1j * f0_imag
        f1 = f1_real + 1j * f1_imag
        f2 = f2_real + 1j * f2_imag
        
        # Expand dimensions for broadcasting
        # Add dimensions for color and direction
        f0 = f0.view(1, 1, 1, *f0.shape)
        f1 = f1.view(1, 1, 1, *f1.shape)
        f2 = f2.view(1, 1, 1, *f2.shape)
        
        # Create identity matrix
        identity = torch.eye(Nc, dtype=torch.complex64, device=self.device)
        identity = identity.view(Nc, Nc, 1, 1, 1, 1, 1)
        identity = identity.expand(-1, -1, self.Nd, *Q.shape[3:])
        
        # Compute exp(iQ) = f0*I + f1*Q + f2*Q^2
        exp_iQ = f0 * identity + f1 * Q + f2 * Q_sq
        
        return exp_iQ
    
    def _exp_iq_eigendecomposition(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Compute exp(iQ) using eigendecomposition.
        
        For Hermitian matrix Q, we have:
        exp(iQ) = V * diag(exp(iλ)) * V^†
        
        where λ are eigenvalues and V are eigenvectors.
        
        Args:
            Q: Traceless Hermitian matrix with shape (Nc, Nc, Nd, ...)
            
        Returns:
            exp(iQ) with same shape as Q
        """
        Nc, Nd = self.Nc, self.Nd
        shape = Q.shape
        spatial_shape = shape[3:]
        
        # Reshape for batch processing: (Nc, Nc, total_sites)
        total_sites = Nd * np.prod(spatial_shape)
        Q_flat = Q.reshape(Nc, Nc, total_sites)
        
        # Transpose to (total_sites, Nc, Nc) for batch eigendecomposition
        Q_flat = Q_flat.permute(2, 0, 1)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(Q_flat)
        
        # exp(iQ) = V * diag(exp(iλ)) * V^†
        exp_i_eigenvalues = torch.exp(1j * eigenvalues)
        
        # Create diagonal matrices
        diag_exp = torch.diag_embed(exp_i_eigenvalues)
        
        # Compute exp(iQ)
        V = eigenvectors
        V_dag = eigenvectors.transpose(1, 2).conj()
        exp_iQ_flat = torch.matmul(V, torch.matmul(diag_exp, V_dag))
        
        # Reshape back to original shape
        exp_iQ_flat = exp_iQ_flat.permute(1, 2, 0)
        exp_iQ = exp_iQ_flat.reshape(Nc, Nc, Nd, *spatial_shape)
        
        return exp_iQ
    
    def stout_smear(self, 
                   U: torch.Tensor, 
                   nstep: int = 20, 
                   rho: float = 0.12,
                   method: str = 'cayley_hamilton') -> torch.Tensor:
        """
        Perform Stout smearing on gauge field U.
        
        Args:
            U: Input gauge field with shape (Nd, Lx, Ly, Lz, Lt, Nc, Nc)
            nstep: Number of smearing iterations
            rho: Smearing parameter
            method: Method for computing exp(iQ), either 'cayley_hamilton' or 'eigendecomposition'
            
        Returns:
            Smeared gauge field with same shape as input
        """
        # Validate input
        assert U.dim() == 7, f"U must have 7 dimensions, got {U.dim()}"
        assert U.shape[0] == self.Nd, f"First dimension must be {self.Nd} (directions)"
        assert U.shape[-2:] == (self.Nc, self.Nc), f"Last two dimensions must be ({self.Nc}, {self.Nc})"
        assert method in ['cayley_hamilton', 'eigendecomposition'], \
            f"method must be 'cayley_hamilton' or 'eigendecomposition', got {method}"
        
        # Move to device and reorder layout
        U = U.to(self.device)
        U_reordered = self._reorder_u(U)
        
        # Perform smearing iterations
        for step in range(nstep):
            # Compute staples
            staples = self._compute_staples(U_reordered)
            
            # Compute Q matrix
            Q = self._compute_q_matrix(U_reordered, staples, rho)
            
            # Compute exp(iQ)
            if method == 'cayley_hamilton':
                exp_iQ = self._exp_iq_cayley_hamilton(Q)
            else:  # eigendecomposition
                exp_iQ = self._exp_iq_eigendecomposition(Q)
            
            # Update gauge field: U_new = exp(iQ) * U_old
            U_reordered = torch.einsum('ab...,bc...->ac...', exp_iQ, U_reordered)
            
            # Optional: print progress
            if (step + 1) % 5 == 0:
                print(f"Stout smearing step {step + 1}/{nstep} completed")
        
        # Restore original layout
        U_smeared = self._restore_u(U_reordered)
        
        return U_smeared
    
    def test_consistency(self, 
                        lattice_size: Tuple[int, int, int, int] = (4, 4, 4, 4),
                        nstep: int = 3,
                        rho: float = 0.1,
                        rtol: float = 1e-6,
                        atol: float = 1e-8) -> bool:
        """
        Test consistency between Cayley-Hamilton and eigendecomposition methods.
        
        Args:
            lattice_size: Size of test lattice (Lx, Ly, Lz, Lt)
            nstep: Number of smearing steps for test
            rho: Smearing parameter for test
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            True if tests pass, False otherwise
        """
        print("Running consistency tests...")
        
        # Generate random SU(3) matrices
        Lx, Ly, Lz, Lt = lattice_size
        Nd, Nc = self.Nd, self.Nc
        
        # Create random complex matrices
        torch.manual_seed(42)
        random_real = torch.randn(Nd, Lx, Ly, Lz, Lt, Nc, Nc)
        random_imag = torch.randn(Nd, Lx, Ly, Lz, Lt, Nc, Nc)
        U_random = random_real + 1j * random_imag
        
        # Project to SU(3) using QR decomposition
        U_su3 = []
        for d in range(Nd):
            for x in range(Lx):
                for y in range(Ly):
                    for z in range(Lz):
                        for t in range(Lt):
                            mat = U_random[d, x, y, z, t].to(self.device)
                            # QR decomposition
                            Q, R = torch.linalg.qr(mat)
                            # Make determinant = 1
                            det = torch.det(Q)
                            phase = torch.exp(-1j * torch.angle(det) / Nc)
                            Q = Q * phase
                            U_su3.append(Q)
        
        U_su3 = torch.stack(U_su3).reshape(Nd, Lx, Ly, Lz, Lt, Nc, Nc)
        U_su3 = U_su3.to(self.device)
        
        print(f"Test lattice size: {lattice_size}")
        print(f"Test parameters: nstep={nstep}, rho={rho}")
        
        # Test 1: Cayley-Hamilton method
        print("\n1. Testing Cayley-Hamilton method...")
        U_ch = self.stout_smear(U_su3.clone(), nstep=nstep, rho=rho, method='cayley_hamilton')
        
        # Test 2: Eigendecomposition method
        print("\n2. Testing eigendecomposition method...")
        U_eig = self.stout_smear(U_su3.clone(), nstep=nstep, rho=rho, method='eigendecomposition')
        
        # Compare results
        diff = torch.abs(U_ch - U_eig)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nComparison results:")
        print(f"  Maximum difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")
        
        # Check if matrices are still SU(3)
        def check_su3(U_tensor):
            """Check if matrices are approximately SU(3)."""
            errors = []
            U_tensor = U_tensor.cpu()  # Move to CPU for checking
            for d in range(Nd):
                for x in range(Lx):
                    for y in range(Ly):
                        for z in range(Lz):
                            for t in range(Lt):
                                mat = U_tensor[d, x, y, z, t]
                                # Check unitarity: U^† U = I
                                identity = torch.eye(Nc, dtype=torch.complex64)
                                unitary_check = torch.norm(mat.conj().T @ mat - identity)
                                errors.append(unitary_check.item())
            
            max_error = max(errors)
            mean_error = sum(errors) / len(errors)
            return max_error, mean_error
        
        print("\n3. Checking SU(3) properties after smearing:")
        max_err_ch, mean_err_ch = check_su3(U_ch)
        max_err_eig, mean_err_eig = check_su3(U_eig)
        
        print(f"  Cayley-Hamilton - Max unitarity error: {max_err_ch:.6e}, Mean: {mean_err_ch:.6e}")
        print(f"  Eigendecomposition - Max unitarity error: {max_err_eig:.6e}, Mean: {mean_err_eig:.6e}")
        
        # Determine if tests pass
        ch_eig_close = torch.allclose(U_ch.cpu(), U_eig.cpu(), rtol=rtol, atol=atol)
        su3_valid = max_err_ch < 1e-5 and max_err_eig < 1e-5
        
        if ch_eig_close and su3_valid:
            print("\n✅ All tests passed!")
            return True
        else:
            print("\n❌ Tests failed!")
            if not ch_eig_close:
                print("  - Cayley-Hamilton and eigendecomposition results differ")
            if not su3_valid:
                print("  - SU(3) property not preserved")
            return False


# Example usage and test
if __name__ == "__main__":
    # Initialize smearing class
    smearing = StoutSmearing(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a small test lattice
    Lx, Ly, Lz, Lt = 4, 4, 4, 4
    Nd, Nc = 4, 3
    
    # Generate random SU(3) matrices
    torch.manual_seed(42)
    U_test = torch.randn(Nd, Lx, Ly, Lz, Lt, Nc, Nc, dtype=torch.complex64)
    
    # Project to SU(3) (simplified)
    for d in range(Nd):
        for x in range(Lx):
            for y in range(Ly):
                for z in range(Lz):
                    for t in range(Lt):
                        mat = U_test[d, x, y, z, t]
                        # Make unitary
                        U, S, V = torch.svd(mat)
                        mat_unitary = U @ V.conj().T
                        # Make determinant = 1
                        det = torch.det(mat_unitary)
                        phase = torch.exp(-1j * torch.angle(det) / Nc)
                        U_test[d, x, y, z, t] = mat_unitary * phase
    
    print("Testing Stout smearing implementation...")
    print(f"Device: {smearing.device}")
    print(f"Lattice size: ({Lx}, {Ly}, {Lz}, {Lt})")
    
    # Run consistency tests
    success = smearing.test_consistency(
        lattice_size=(Lx, Ly, Lz, Lt),
        nstep=3,  # Use fewer steps for quick test
        rho=0.1
    )
    
    if success:
        # Example of actual smearing
        print("\nRunning full smearing example...")
        U_smeared = smearing.stout_smear(
            U_test.to(smearing.device),
            nstep=5,  # Reduced for quick test
            rho=0.12,
            method='cayley_hamilton'  # or 'eigendecomposition'
        )
        
        print(f"Input shape: {U_test.shape}")
        print(f"Output shape: {U_smeared.shape}")
        print("Smearing completed successfully!")
