import time
import matplotlib.pyplot as plt
import numpy as np

# Ensure old Numpy compatibility
np.Inf = np.inf
# if not hasattr(np, 'inf'):
#     np.inf = float('inf')

# ==============================================================================
# 1. Core Operator
# ==============================================================================


class MatrixVectorOperator:
    """
    Encapsulates 2D matrix-vector multiplication, avoiding explicit matrix storage.
    """

    def __init__(self, nx, ny, alpha=1.0, beta=1.0j, dtype=np.complex128):
        self.nx = nx
        self.ny = ny
        self.alpha = alpha
        self.beta = beta
        np.complex128 = dtype
        self.n = nx * ny
        self.hx = 1.0 / (nx + 1)
        self.hy = 1.0 / (ny + 1)

        # Precompute diagonal elements
        self.main_diag = (2 * alpha * (1/self.hx**2 + 1/self.hy**2) +
                          beta) * np.ones(self.n, dtype=np.complex128)

    def matvec(self, v):
        """
        Performs matrix-vector multiplication A*v.
        This implementation is optimized to vectorize operations.
        """
        v_2d = v.reshape((self.ny, self.nx))
        result_2d = np.zeros_like(v_2d)

        # Handle main diagonal
        result_2d = self.main_diag.reshape((self.ny, self.nx)) * v_2d

        # x-direction differences (vectorized)
        result_2d[:, 1:] -= (self.alpha / self.hx**2) * v_2d[:, :-1]
        result_2d[:, :-1] -= (self.alpha / self.hx**2) * v_2d[:, 1:]

        # y-direction differences (vectorized)
        result_2d[1:, :] -= (self.alpha / self.hy**2) * v_2d[:-1, :]
        result_2d[:-1, :] -= (self.alpha / self.hy**2) * v_2d[1:, :]

        return result_2d.flatten()

# ==============================================================================
# 2. N-dimensional Operator Wrapper
# ==============================================================================


class NDimOperatorWrapper:
    """
    A wrapper that decomposes an N-dimensional problem into independent operations
    on multiple 2D slices.
    """

    def __init__(self, grid_shape, mg_axes, alpha, beta):
        self.grid_shape = grid_shape
        self.mg_axes = sorted(mg_axes)

        # Get the shape of the 2D slice (ny, nx)
        self.ny = grid_shape[self.mg_axes[0]]
        self.nx = grid_shape[self.mg_axes[1]]

        # Create a 2D operator instance
        self.op_2d = MatrixVectorOperator(self.nx, self.ny, alpha, beta)

        # Calculate batch/viewer dimension information
        batch_dims = [dim for i, dim in enumerate(
            grid_shape) if i not in self.mg_axes]
        self.num_slices = int(np.prod(batch_dims)) if batch_dims else 1

        # Define dimension permutation order, moving mg_axes to the end
        self.permute_to_2d = [i for i in range(
            len(grid_shape)) if i not in self.mg_axes] + self.mg_axes
        self.inverse_permute = np.argsort(self.permute_to_2d)

    def matvec(self, v):
        """Performs matrix-vector multiplication for N-dimensional vector v"""
        if len(self.grid_shape) == 2:
            return self.op_2d.matvec(v)

        v_nd = v.reshape(self.grid_shape)
        v_permuted = np.transpose(v_nd, self.permute_to_2d)
        v_slices = v_permuted.reshape(self.num_slices, self.ny * self.nx)

        result_slices = np.zeros_like(v_slices)
        for i in range(self.num_slices):
            result_slices[i, :] = self.op_2d.matvec(v_slices[i, :])

        result_permuted = result_slices.reshape(v_permuted.shape)
        result_nd = np.transpose(result_permuted, self.inverse_permute)

        return result_nd.flatten()

# ==============================================================================
# 3. Restored GMRES Smoother
# ==============================================================================


class GMRESSmoother:
    """GMRES Smoother Implementation (restored to manual version)"""

    def __init__(self, max_krylov=5, max_restarts=1, tol=0.1):
        self.max_krylov = max_krylov
        self.max_restarts = max_restarts
        self.tol = tol

    def smooth(self, op, b, x0):
        x = x0.copy()
        # Initial residual
        r = b - op.matvec(x0)
        r_norm_initial = np.linalg.norm(r)

        if r_norm_initial < 1e-12:  # If already converged
            return x0

        for _ in range(self.max_restarts):
            # Arnoldi process to build Krylov subspace
            Q, H, j = self._arnoldi(op, r, self.max_krylov)

            # Set up and solve least squares problem
            e1 = np.zeros(j + 1, dtype=np.complex128)
            e1[0] = np.linalg.norm(r)

            y, _, _, _ = np.linalg.lstsq(H, e1, rcond=None)

            # Update solution
            update = Q[:, :j] @ y
            x = x + update

            # Compute new residual and check for convergence
            r = b - op.matvec(x)
            r_norm_new = np.linalg.norm(r)
            if r_norm_new < self.tol * r_norm_initial:
                break

        return x

    def _arnoldi(self, op, r0, m):
        n = len(r0)
        Q = np.zeros((n, m + 1), dtype=np.complex128)
        H = np.zeros((m + 1, m), dtype=np.complex128)

        r0_norm = np.linalg.norm(r0)
        if r0_norm == 0:
            return Q, H, 0

        Q[:, 0] = r0 / r0_norm

        for j in range(m):
            w = op.matvec(Q[:, j])

            for i in range(j + 1):
                H[i, j] = np.vdot(w, Q[:, i])
                w = w - H[i, j] * Q[:, i]

            H[j + 1, j] = np.linalg.norm(w)

            if H[j + 1, j] < 1e-12:  # Early termination (Lucky Breakdown)
                return Q[:, :j+1], H[:j+2, :j+1], j+1

            Q[:, j + 1] = w / H[j + 1, j]

        return Q[:, :m], H[:m+1, :m], m


# ==============================================================================
# 4. Updated Adaptive Multigrid Solver
# ==============================================================================

class AdaptiveMultigridComplex:
    """
    Adaptive Multigrid method for solving complex systems (updated to support N-dimensions)
    """

    def __init__(self, grid_shape, mg_axes, max_levels=5, tolerance=1e-8, max_iterations=100):
        self.grid_shape = grid_shape
        self.mg_axes = sorted(mg_axes)
        self.max_levels = max_levels
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.convergence_history = []

        self.ny_mg = self.grid_shape[self.mg_axes[0]]
        self.nx_mg = self.grid_shape[self.mg_axes[1]]

        self.gmres_smoother = GMRESSmoother(
            max_krylov=10, max_restarts=1, tol=0.5)

    def create_operator(self, shape, alpha, beta):
        return NDimOperatorWrapper(shape, self.mg_axes, alpha, beta)

    def create_rhs(self, shape, func_type='sine'):
        ny = shape[self.mg_axes[0]]
        nx = shape[self.mg_axes[1]]

        hx = 1.0 / (nx + 1)
        hy = 1.0 / (ny + 1)
        x = np.linspace(hx, 1 - hx, nx)
        y = np.linspace(hy, 1 - hy, ny)
        X, Y = np.meshgrid(x, y)

        if func_type == 'sine':
            f_2d = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) * (1 + 1j)
        else:
            f_2d = np.exp(X + 1j * Y)

        if len(shape) > 2:
            target_shape_for_tile = [
                dim if i not in self.mg_axes else 1 for i, dim in enumerate(shape)]
            f_nd = np.tile(f_2d, target_shape_for_tile)
            return f_nd.flatten()
        else:
            return f_2d.flatten()

    def restrict(self, u_fine, shape_fine):
        shape_coarse_list = list(shape_fine)
        ax1, ax2 = self.mg_axes
        
        # Calculate target coarse shape
        target_ny_coarse = max(1, shape_fine[ax1] // 2)
        target_nx_coarse = max(1, shape_fine[ax2] // 2)

        # Handle cases where dimensions become too small or already 1
        if shape_fine[ax1] < 2 or shape_fine[ax2] < 2: 
            return u_fine.copy()

        shape_coarse_list[ax1] = target_ny_coarse
        shape_coarse_list[ax2] = target_nx_coarse
        shape_coarse = tuple(shape_coarse_list)
        
        # If dimensions are already effectively 1, return copy
        if shape_fine == shape_coarse: 
            return u_fine.copy()

        u_fine_nd = u_fine.reshape(shape_fine)
        
        # Permute dimensions to bring MG axes to the end for easier 2D slice processing
        permute_order = [i for i in range(len(shape_fine)) if i not in self.mg_axes] + self.mg_axes
        u_fine_permuted = np.transpose(u_fine_nd, axes=permute_order)
        
        # Determine the shape of the coarse grid after permutation
        coarse_permuted_shape = u_fine_permuted.shape[:-2] + (target_ny_coarse, target_nx_coarse)
        u_coarse_permuted = np.zeros(coarse_permuted_shape, dtype=np.complex128)

        # Iterate over batch dimensions (non-MG dimensions)
        for index in np.ndindex(u_fine_permuted.shape[:-2]):
            fine_slice = u_fine_permuted[index]
            coarse_slice = u_coarse_permuted[index]
            # ny_fine_slice, nx_fine_slice = fine_slice.shape # Not needed explicitly here, padded_fine_slice handles bounds

            # Pad the fine slice with zeros for ghost points to handle boundaries
            # This is equivalent to applying Dirichlet BCs to the residual
            padded_fine_slice = np.pad(fine_slice, 1, mode='constant', constant_values=0)

            ny_coarse_slice, nx_coarse_slice = coarse_slice.shape

            for i in range(ny_coarse_slice):
                for j in range(nx_coarse_slice):
                    # Map coarse grid (i,j) to fine grid (ii,jj) (cell-centered mapping)
                    # The indices are now relative to the *padded* array (offset by 1 due to padding)
                    ii_padded, jj_padded = (2 * i + 1) + 1, (2 * j + 1) + 1 

                    # Apply 9-point restriction stencil
                    # All accesses are now guaranteed to be within padded_fine_slice
                    val = (
                        4 * padded_fine_slice[ii_padded, jj_padded] +
                        2 * (padded_fine_slice[ii_padded - 1, jj_padded] +
                             padded_fine_slice[ii_padded + 1, jj_padded] +
                             padded_fine_slice[ii_padded, jj_padded - 1] +
                             padded_fine_slice[ii_padded, jj_padded + 1]) +
                        1 * (padded_fine_slice[ii_padded - 1, jj_padded - 1] +
                             padded_fine_slice[ii_padded + 1, jj_padded - 1] +
                             padded_fine_slice[ii_padded - 1, jj_padded + 1] +
                             padded_fine_slice[ii_padded + 1, jj_padded + 1])
                    )
                    coarse_slice[i, j] = val / 16.0

        # Permute back to original dimensions and flatten
        inverse_permute_order = np.argsort(permute_order)
        return np.transpose(u_coarse_permuted, axes=inverse_permute_order).flatten()

    def prolongate(self, u_coarse, shape_fine):
        shape_coarse_list = list(shape_fine)
        ax1, ax2 = self.mg_axes
        
        # Calculate target coarse shape
        target_ny_coarse = max(1, shape_fine[ax1] // 2)
        target_nx_coarse = max(1, shape_fine[ax2] // 2)

        # Handle cases where dimensions become too small or already 1
        if shape_fine[ax1] < 2 or shape_fine[ax2] < 2:
            return u_coarse.copy()

        shape_coarse_list[ax1] = target_ny_coarse
        shape_coarse_list[ax2] = target_nx_coarse
        shape_coarse = tuple(shape_coarse_list)
        
        # If dimensions are already effectively 1, return copy
        if shape_fine == shape_coarse:
            return u_coarse.copy()

        u_coarse_nd = u_coarse.reshape(shape_coarse)
        
        # Permute dimensions to bring MG axes to the end for easier 2D slice processing
        permute_order = [i for i in range(len(shape_fine)) if i not in self.mg_axes] + self.mg_axes
        u_coarse_permuted = np.transpose(u_coarse_nd, axes=permute_order)
        
        # Determine the shape of the fine grid after permutation
        fine_permuted_shape = u_coarse_permuted.shape[:-2] + (shape_fine[ax1], shape_fine[ax2])
        u_fine_permuted = np.zeros(fine_permuted_shape, dtype=np.complex128)

        # Iterate over batch dimensions
        for index in np.ndindex(u_coarse_permuted.shape[:-2]):
            coarse_slice = u_coarse_permuted[index]
            fine_slice = u_fine_permuted[index]
            ny_coarse_slice, nx_coarse_slice = coarse_slice.shape
            ny_fine_slice, nx_fine_slice = fine_slice.shape

            # 1. Direct injection for coarse grid points (even rows, even columns)
            # fine_slice[2*i, 2*j] = coarse_slice[i, j]
            fine_slice[::2, ::2] = coarse_slice

            # 2. Linear interpolation for midpoints in x-direction (even rows, odd columns)
            # fine_slice[2*i, 2*j+1] = 0.5 * (fine_slice[2*i, 2*j] + fine_slice[2*i, 2*j+2])
            for j_fine in range(1, nx_fine_slice, 2): # Iterate through all odd columns on fine grid
                # Ensure the points used for interpolation are within bounds
                left_neighbor_col = j_fine - 1
                right_neighbor_col = j_fine + 1

                if left_neighbor_col >= 0 and right_neighbor_col < nx_fine_slice:
                    fine_slice[::2, j_fine] = 0.5 * (fine_slice[::2, left_neighbor_col] + fine_slice[::2, right_neighbor_col])
                elif left_neighbor_col >= 0: # Right boundary (use only left neighbor)
                    fine_slice[::2, j_fine] = fine_slice[::2, left_neighbor_col]
                elif right_neighbor_col < nx_fine_slice: # Left boundary (use only right neighbor, less common for standard grids)
                    fine_slice[::2, j_fine] = fine_slice[::2, right_neighbor_col]
                # If neither, point remains zero (default for boundary) or could handle differently

            # 3. Linear interpolation for midpoints in y-direction (odd rows, even columns)
            # fine_slice[2*i+1, 2*j] = 0.5 * (fine_slice[2*i, 2*j] + fine_slice[2*i+2, 2*j])
            for i_fine in range(1, ny_fine_slice, 2): # Iterate through all odd rows on fine grid
                # Ensure the points used for interpolation are within bounds
                top_neighbor_row = i_fine - 1
                bottom_neighbor_row = i_fine + 1

                if top_neighbor_row >= 0 and bottom_neighbor_row < ny_fine_slice:
                    fine_slice[i_fine, ::2] = 0.5 * (fine_slice[top_neighbor_row, ::2] + fine_slice[bottom_neighbor_row, ::2])
                elif top_neighbor_row >= 0: # Bottom boundary (use only top neighbor)
                    fine_slice[i_fine, ::2] = fine_slice[top_neighbor_row, ::2]
                elif bottom_neighbor_row < ny_fine_slice: # Top boundary (use only bottom neighbor)
                    fine_slice[i_fine, ::2] = fine_slice[bottom_neighbor_row, ::2]


            # 4. Bilinear interpolation for center points (odd rows, odd columns)
            # fine_slice[2*i+1, 2*j+1] = 0.25 * (fine_slice[2*i, 2*j] + fine_slice[2*i+2, 2*j] + fine_slice[2*i, 2*j+2] + fine_slice[2*i+2, 2*j+2])
            for i_fine in range(1, ny_fine_slice, 2):
                for j_fine in range(1, nx_fine_slice, 2):
                    val = 0.0
                    count = 0
                    
                    # Top-left neighbor
                    if i_fine - 1 >= 0 and j_fine - 1 >= 0:
                        val += fine_slice[i_fine - 1, j_fine - 1]
                        count += 1
                    # Top-right neighbor
                    if i_fine - 1 >= 0 and j_fine + 1 < nx_fine_slice:
                        val += fine_slice[i_fine - 1, j_fine + 1]
                        count += 1
                    # Bottom-left neighbor
                    if i_fine + 1 < ny_fine_slice and j_fine - 1 >= 0:
                        val += fine_slice[i_fine + 1, j_fine - 1]
                        count += 1
                    # Bottom-right neighbor
                    if i_fine + 1 < ny_fine_slice and j_fine + 1 < nx_fine_slice:
                        val += fine_slice[i_fine + 1, j_fine + 1]
                        count += 1
                    
                    if count > 0:
                        fine_slice[i_fine, j_fine] = val / count
                    else:
                        fine_slice[i_fine, j_fine] = 0.0 # Default if no neighbors found (should not happen with proper grids)


        # Permute back to original dimensions and flatten
        inverse_permute_order = np.argsort(permute_order)
        return np.transpose(u_fine_permuted, axes=inverse_permute_order).flatten()

    def smooth(self, op, b, u, num_iterations=2):
        for _ in range(num_iterations):
            u = self.gmres_smoother.smooth(op, b, u)
        return u

    def compute_residual(self, op, b, u):
        return b - op.matvec(u)

    def bistabcg_solver(self, op, b, x0=None, tol=1e-10, maxiter=1000):
        """Restored manual BiCGSTAB implementation"""
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()

        r = b - op.matvec(x)
        r0_hat = r.copy()

        rho_prev = 1.0
        alpha = 1.0
        omega = 1.0
        v = np.zeros_like(b)
        p = np.zeros_like(b)

        b_norm = np.linalg.norm(b)
        if b_norm == 0.0:
            b_norm = 1.0

        for i in range(maxiter):
            rho_curr = np.vdot(r0_hat, r)
            if abs(rho_curr) < 1e-50:
                break

            if i > 0:
                beta = (rho_curr / rho_prev) * (alpha / omega)
                p = r + beta * (p - omega * v)
            else:
                p = r

            v = op.matvec(p)
            alpha = rho_curr / np.vdot(r0_hat, v)
            s = r - alpha * v

            if np.linalg.norm(s) < tol * b_norm:
                x += alpha * p
                break

            t = op.matvec(s)
            omega = np.vdot(t, s) / np.vdot(t, t)
            x += alpha * p + omega * s
            r = s - omega * t

            rho_prev = rho_curr

            if np.linalg.norm(r) < tol * b_norm:
                break
        return x, 0  # info=0 for success

    def v_cycle(self, op_hierarchy, b_hierarchy, u_hierarchy, grid_shapes, level=0):
        current_level = len(op_hierarchy) - 1 - level
        current_shape = grid_shapes[current_level]

        op = op_hierarchy[current_level]
        b = b_hierarchy[current_level]
        u = u_hierarchy[current_level]

        if current_level == 0 or level >= self.max_levels - 1:
            u_coarse, info = self.bistabcg_solver(
                op, b, tol=1e-12, maxiter=2000)
            if info != 0:
                print(f"   Warning: Coarsest grid solve did not converge!")
            u_hierarchy[current_level] = u_coarse
            return u_hierarchy[current_level]

        u = self.smooth(op, b, u)
        residual = self.compute_residual(op, b, u)

        r_coarse = self.restrict(residual, current_shape)
        b_hierarchy[current_level - 1] = r_coarse
        u_hierarchy[current_level - 1] = np.zeros_like(r_coarse)

        e_coarse = self.v_cycle(op_hierarchy, b_hierarchy,
                                 u_hierarchy, grid_shapes, level + 1)

        e_fine = self.prolongate(e_coarse, current_shape)
        u = u + e_fine

        u = self.smooth(op, b, u)
        u_hierarchy[current_level] = u
        return u

    def solve(self, alpha=1.0, beta=1.0j, func_type='sine'):
        print("="*60 + "\nStarting Adaptive Multigrid Complex Solver (N-D Version)\n" + "="*60)
        start_time = time.time()
        grid_shapes = []
        current_shape_list = list(self.grid_shape)
        ax1, ax2 = self.mg_axes
        print(f"Building grid hierarchy (coarsening on dimensions {self.mg_axes}):")
        while current_shape_list[ax1] >= 4 and current_shape_list[ax2] >= 4 and len(grid_shapes) < self.max_levels:
            current_shape = tuple(current_shape_list)
            grid_shapes.append(current_shape)
            print(f"   Level {len(grid_shapes)-1}: {current_shape}")
            current_shape_list[ax1] //= 2
            current_shape_list[ax2] //= 2
        num_levels = len(grid_shapes)
        grid_shapes.reverse() # Coarsest to finest
        op_hierarchy = [self.create_operator(
            shape, alpha, beta) for shape in grid_shapes]
        b_hierarchy = [self.create_rhs(shape, func_type) if i == num_levels - 1 else np.zeros(
            int(np.prod(shape)), dtype=np.complex128) for i, shape in enumerate(grid_shapes)]
        u_hierarchy = [np.zeros(int(np.prod(shape)), dtype=np.complex128)
                       for shape in grid_shapes]

        print(f"\nStarting Multigrid Iterations (total {num_levels} levels):")
        for iteration in range(self.max_iterations):
            self.v_cycle(op_hierarchy, b_hierarchy, u_hierarchy, grid_shapes)
            residual_norm = np.linalg.norm(self.compute_residual(
                op_hierarchy[-1], b_hierarchy[-1], u_hierarchy[-1]))
            self.convergence_history.append(residual_norm)
            print(f"Iteration {iteration + 1} complete, Residual Norm: {residual_norm:.4e}")
            if residual_norm < self.tolerance:
                print(f"   ✓ Converged to tolerance {self.tolerance}")
                break
        solve_time = time.time() - start_time
        print("\n" + "="*60 + "\nSolution Complete!")
        print(f"Total Iterations: {len(self.convergence_history)}")
        if self.convergence_history:
            print(f"Final Residual: {self.convergence_history[-1]:.2e}")
        print(f"Solve Time: {solve_time:.4f} seconds\n" + "="*60)
        return u_hierarchy[-1].reshape(self.grid_shape)

    def verify_solution(self, solution_2d, alpha=1.0, beta=1.0j, func_type='sine'):
        print("\nVerifying solution correctness (for a 2D slice):")
        ny, nx = solution_2d.shape
        op_2d = MatrixVectorOperator(nx, ny, alpha, beta)
        hx = 1.0/(nx+1)
        hy = 1.0/(ny+1)
        x = np.linspace(hx, 1-hx, nx)
        y = np.linspace(hy, 1-hy, ny)
        X, Y = np.meshgrid(x, y)
        if func_type == 'sine':
            b_2d = (np.sin(2*np.pi*X)*np.sin(2*np.pi*Y)*(1+1j)).flatten()
        else:
            b_2d = (np.exp(X+1j*Y)).flatten()
        residual = op_2d.matvec(solution_2d.flatten())-b_2d
        residual_norm = np.linalg.norm(residual)
        relative_error = residual_norm / np.linalg.norm(b_2d)
        print(f"Verification Residual Norm: {residual_norm:.4e}, Relative Error: {relative_error:.2e}")
        if relative_error < 1e-5:
            print("✓ Solution verification passed!")
        else:
            print("⚠ Solution might have precision issues")

    def plot_results(self, solution_2d):
        print("\nGenerating visualization plots...")
        ny, nx = solution_2d.shape
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            'Adaptive Multigrid Complex Solution (2D Slice)', fontsize=16)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        im1 = axes[0, 0].contourf(X, Y, np.real(
            solution_2d), levels=20, cmap='RdBu_r')
        axes[0, 0].set_title('Solution - Real Part')
        plt.colorbar(im1, ax=axes[0, 0])
        im2 = axes[0, 1].contourf(X, Y, np.imag(
            solution_2d), levels=20, cmap='RdBu_r')
        axes[0, 1].set_title('Solution - Imaginary Part')
        plt.colorbar(im2, ax=axes[0, 1])
        im3 = axes[1, 0].contourf(X, Y, np.abs(
            solution_2d), levels=20, cmap='viridis')
        axes[1, 0].set_title('Solution - Magnitude')
        plt.colorbar(im3, ax=axes[1, 0])
        axes[1, 1].semilogy(
            range(1, len(self.convergence_history)+1), self.convergence_history, 'b-o')
        axes[1, 1].set_title('Convergence History')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Residual Norm')
        axes[1, 1].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


# ==============================================================================
# 5. Main Program
# ==============================================================================
if __name__ == "__main__":
    print("Adaptive Multigrid Complex Solver Demo (N-D General Version)")
    print("=" * 60)
    full_grid_shape = (2, 32, 64)
    multigrid_axes = (1, 2)
    solver = AdaptiveMultigridComplex(
        grid_shape=full_grid_shape,
        mg_axes=multigrid_axes,
        max_levels=4,
        tolerance=1e-7,
        max_iterations=20
    )
    case = {"alpha": 1.0, "beta": 1.0j, "func_type": "sine", "name": "Complex Elliptic Problem"}
    print(f"\n{'='*80}\nTest Case: {case['name']}")
    print(f"Grid Shape: {full_grid_shape}, Operating Dimensions: {multigrid_axes}\n{'='*80}")
    solution_nd = solver.solve(
        alpha=case["alpha"], beta=case["beta"], func_type=case["func_type"])
    print(f"\nN-D Solution Complete, Solution Shape: {solution_nd.shape}")
    solution_2d_slice = solution_nd[0, :, :]
    print(f"Selected first 2D slice (shape: {solution_2d_slice.shape}) for verification and plotting.")
    solver.verify_solution(
        solution_2d_slice, alpha=case["alpha"], beta=case["beta"], func_type=case["func_type"])
    solver.plot_results(solution_2d_slice)
    print(f"\n{'='*80}\nAll Tests Complete!\n{'='*80}")
