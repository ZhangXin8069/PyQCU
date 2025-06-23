# %%
from pyqcu import io, gauge, cg, bistabcg, define, qcu, linalg,eigen
from pyqcu.set import *
import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import time
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import spsolve, LinearOperator, bicgstab, aslinearoperator
from scipy.linalg import qr
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

np.Inf = np.inf

# %%
params[define._LAT_X_] = 32
params[define._LAT_Y_] = 16
params[define._LAT_Z_] = 4
params[define._LAT_T_] = 4
params[define._LAT_XYZT_] = params[define._LAT_X_] * \
    params[define._LAT_Y_] * params[define._LAT_Z_] * params[define._LAT_T_]
params[define._LAT_E_] = 8
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
params[define._DATA_TYPE_] = define._LAT_C128_
argv = argv.astype(define.dtype_half(params[define._DATA_TYPE_]))
argv[define._MASS_] = 0.0
argv[define._TOL_] = 1e-8
kappa = 1 / (2 * argv[define._MASS_] + 8)
print('My rank is ', define.rank)
print("Parameters:", params)
print("Args:", argv)
wilson_cg_params = params.copy()
wilson_cg_params[define._SET_INDEX_] = 0
wilson_cg_params[define._SET_PLAN_] = define._SET_PLAN1_
qcu.applyInitQcu(set_ptrs, wilson_cg_params, argv)
wilson_dslash_eo_params = params.copy()
wilson_dslash_eo_params[define._SET_INDEX_] = 1
wilson_dslash_eo_params[define._SET_PLAN_] = define._SET_PLAN0_
wilson_dslash_eo_params[define._PARITY_] = define._EVEN_
wilson_dslash_eo_params[define._DAGGER_] = define._NO_USE_
qcu.applyInitQcu(set_ptrs, wilson_dslash_eo_params, argv)
wilson_dslash_eo_dag_params = params.copy()
wilson_dslash_eo_dag_params[define._SET_INDEX_] = 2
wilson_dslash_eo_dag_params[define._SET_PLAN_] = define._SET_PLAN0_
wilson_dslash_eo_dag_params[define._PARITY_] = define._EVEN_
wilson_dslash_eo_dag_params[define._DAGGER_] = define._USE_
qcu.applyInitQcu(set_ptrs, wilson_dslash_eo_dag_params, argv)
wilson_dslash_oe_params = params.copy()
wilson_dslash_oe_params[define._SET_INDEX_] = 3
wilson_dslash_oe_params[define._SET_PLAN_] = define._SET_PLAN0_
wilson_dslash_oe_params[define._PARITY_] = define._ODD_
wilson_dslash_oe_params[define._DAGGER_] = define._NO_USE_
qcu.applyInitQcu(set_ptrs, wilson_dslash_oe_params, argv)
wilson_dslash_oe_dag_params = params.copy()
wilson_dslash_oe_dag_params[define._SET_INDEX_] = 4
wilson_dslash_oe_dag_params[define._SET_PLAN_] = define._SET_PLAN0_
wilson_dslash_oe_dag_params[define._PARITY_] = define._ODD_
wilson_dslash_oe_dag_params[define._DAGGER_] = define._USE_
qcu.applyInitQcu(set_ptrs, wilson_dslash_oe_dag_params, argv)
print("Set pointers:", set_ptrs)
print("Set pointers data:", set_ptrs.data)

# %%
U = gauge.give_gauss_SU3(sigma=0.1, seed=12138,
                         dtype=define.dtype(params[define._DATA_TYPE_]), size=params[define._LAT_XYZT_]*define._LAT_S_)
U = io.dptzyxcc2ccdptzyx(
    io.gauge2dptzyxcc(gauge=U, params=params))

# %%
def pdslash_no_dag(src):
    tmp0 = cp.zeros_like(src)
    tmp1 = cp.zeros_like(src)
    qcu.applyWilsonDslashQcu(
        tmp0, src, U, set_ptrs, wilson_dslash_eo_params)
    qcu.applyWilsonDslashQcu(
        tmp1, tmp0, U, set_ptrs, wilson_dslash_oe_params)
    return src-kappa**2*tmp1


def pdslash_dag(src):
    tmp0 = cp.zeros_like(src)
    tmp1 = cp.zeros_like(src)
    qcu.applyWilsonDslashQcu(
        tmp0, src, U, set_ptrs, wilson_dslash_eo_dag_params)
    qcu.applyWilsonDslashQcu(
        tmp1, tmp0, U, set_ptrs, wilson_dslash_oe_dag_params)
    return src-kappa**2*tmp1


def dslash_no_dag(src):
    dest = cp.zeros_like(src)
    qcu.applyWilsonDslashQcu(
        dest, src, U, set_ptrs, wilson_dslash_eo_params)
    return dest


def dslash_dag(src):
    dest = cp.zeros_like(src)
    qcu.applyWilsonDslashQcu(
        dest, src, U, set_ptrs, wilson_dslash_eo_dag_params)
    return dest


def bistabcg_dslash(src):
    return pdslash_no_dag(src)


def cg_dslash(src):
    return pdslash_dag(pdslash_no_dag(src))

# %%
src = cp.zeros(
    params[define._LAT_XYZT_]*define._LAT_SC_, dtype=define.dtype(params[define._DATA_TYPE_]))
src = linalg.initialize_random_vector(src)
src=io.fermion2ptzyxsc(src, params=params)
dest = cp.zeros_like(src)
# qcu.applyWilsonBistabCgQcu(dest, src,
#                            U, set_ptrs, wilson_cg_params)
qcu.applyWilsonCgQcu(dest, src,
                     U, set_ptrs, wilson_cg_params)

# %%
class LatticeLinearOperator:
    def __init__(self):
        self.shape = (params[define._LAT_XYZT_]*define._LAT_SC_,params[define._LAT_XYZT_]*define._LAT_SC_)
        self.dtype = define.dtype(params[define._DATA_TYPE_])
    def matvec(self, src):
        return cg_dslash(cp.array(src)).get()
    

# %%
class AMGEigenvectorCoarsening:
    """
    Algebraic multigrid coarsening strategy based on near-kernel eigenvectors.
    """

    def __init__(self, num_eigenvectors=3, max_coarse_size=100,
                 coarsening_ratio=0.3, smoothness_threshold=0.5,
                 power_iterations=15, chebyshev_degree=5):
        self.num_eigenvectors = num_eigenvectors
        self.max_coarse_size = max_coarse_size
        self.coarsening_ratio = coarsening_ratio
        self.smoothness_threshold = smoothness_threshold
        self.power_iterations = power_iterations
        self.chebyshev_degree = chebyshev_degree

    def chebyshev_filter(self, A, v, lambda_min, lambda_max):
        """
        Applies a Chebyshev filter to accelerate power iteration.
        This is the corrected implementation.
        """
        # Calculate Chebyshev polynomial parameters
        delta = (lambda_max - lambda_min) / 2.0
        sigma = (lambda_max + lambda_min) / 2.0

        # Avoid division by zero if the spectral estimate is poor
        if abs(delta) < 1e-12:
            return v

        # Initialize vectors for the three-term recurrence T_{k+1}(x) = 2xT_k(x) - T_{k-1}(x)
        # for the shifted/scaled matrix A' = (A - sigma*I) / delta.
        # Let v_k = T_k(A')v.

        # v_0 = v
        v_prev = v

        # v_1 = A' * v_0
        v_curr = (A.matvec(v_prev) - sigma * v_prev) / delta

        # Higher order terms (k > 1)
        for _ in range(1, self.chebyshev_degree):
            # v_{k+1} = 2 * A' * v_k - v_{k-1}
            v_next = 2 * (A.matvec(v_curr) - sigma * v_curr) / delta - v_prev
            v_prev = v_curr
            v_curr = v_next

        return v_curr

    def power_iteration_with_chebyshev(self, A, num_vectors, max_iter=20, tol=1e-6):
        """Computes eigenvectors using Chebyshev-accelerated power iteration."""
        n = A.shape[0]  # A is a LinearOperator with a shape attribute
        eigenvectors = np.zeros((n, num_vectors), dtype=np.complex128)

        # Estimate spectral radius (simplified estimation)
        lambda_max = 0.0
        for _ in range(10):
            v_rand = np.random.randn(n) + 1j * np.random.randn(n)
            v_rand /= np.linalg.norm(v_rand)
            Av = A.matvec(v_rand)
            lambda_est = np.abs(np.vdot(v_rand, Av))
            if lambda_est > lambda_max:
                lambda_max = lambda_est

        lambda_min = 0.1 * lambda_max  # Simplified minimum eigenvalue estimate

        print(
            f"  Estimated eigenvalue range for filter: [{lambda_min:.4e}, {lambda_max:.4e}]")

        # Perform power iteration for each eigenvector
        for k in range(num_vectors):
            # Random initialization
            v = np.random.randn(n) + 1j * np.random.randn(n)
            v /= np.linalg.norm(v)

            prev_norm = 0.0
            for i in range(max_iter):
                # Apply Chebyshev filter
                v = self.chebyshev_filter(A, v, lambda_min, lambda_max)

                # Orthogonalize against previously found eigenvectors
                for j in range(k):
                    proj = np.vdot(eigenvectors[:, j], v)
                    v -= proj * eigenvectors[:, j]

                # Normalize
                norm_v = np.linalg.norm(v)
                if norm_v < 1e-12:
                    # Re-initialize if vector becomes too small
                    v = np.random.randn(n) + 1j * np.random.randn(n)
                    v /= np.linalg.norm(v)
                    continue

                v /= norm_v

                # Check for convergence
                if i > 0 and abs(norm_v - prev_norm) < tol:
                    break

                prev_norm = norm_v

            eigenvectors[:, k] = v

        return eigenvectors

    def compute_near_kernel_eigenvectors(self, A):
        """Computes near-kernel eigenvectors using power iteration."""
        print(
            f"Computing {self.num_eigenvectors} near-kernel eigenvectors using power iteration...")

        if isinstance(A, LinearOperator):
            n = A.shape[0]
        else:
            n = A.shape[0]

        if n <= self.max_coarse_size:
            return None

        # Ensure we are working with a linear operator
        if not isinstance(A, LinearOperator):
            A_op = aslinearoperator(A)
        else:
            A_op = A

        # Use Chebyshev-accelerated power iteration
        num_vec = min(self.num_eigenvectors, n-2)
        if num_vec <= 0:
            return None
        """"
        # too slow...
        eigenvectors = self.power_iteration_with_chebyshev(
            A_op, 
            num_vec
        eigenvalues = np.array([np.vdot(v, A_op.matvec(v)) for v in eigenvectors.T])
        )
        """
        def _matvec(src):
            return cp.array(A.matvec(src.get()))
        eigenvalues, eigenvectors = eigen.solver(
            n=n, k=num_vec, matvec=_matvec, dtype=cp.complex128)
        eigenvalues, eigenvectors = eigenvalues.get(), eigenvectors.get().T
        # # Estimate eigenvalues
        sorted_indices = np.argsort(np.abs(eigenvalues))
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        print(
            f"  Estimated eigenvalues: {[f'{abs(ev):.4e}' for ev in eigenvalues]}")

        return eigenvectors

    def analyze_smoothness(self, A, eigenvectors):
        """Analyzes the smoothness of eigenvectors using matrix-vector multiplication."""
        print("Analyzing eigenvector smoothness...")

        if eigenvectors is None or eigenvectors.size == 0:
            return np.zeros(A.shape[0])

        n = eigenvectors.shape[0]
        smoothness_indicators = np.zeros(n)

        # Note: This is computationally expensive as it is matrix-free.
        def diff_operator(x):
            return self._diff_operator_matvec(A, x)

        for i, vec in enumerate(eigenvectors.T):
            # Compute the "smoothness" of the vector
            diff_vec = diff_operator(vec)
            local_smoothness = np.abs(diff_vec) / (np.abs(vec) + 1e-12)

            # Accumulate smoothness indicators
            smoothness_indicators += local_smoothness / (i + 1)

        # Normalize smoothness indicators
        if np.max(smoothness_indicators) > 0:
            smoothness_indicators = smoothness_indicators / \
                np.max(smoothness_indicators)

        print(
            f"  Smoothness indicator range: [{np.min(smoothness_indicators):.4f}, {np.max(smoothness_indicators):.4f}]")

        return smoothness_indicators

    def _diff_operator_matvec(self, A, x):
        """Matrix-vector multiplication implementation for the difference operator."""
        if isinstance(A, LinearOperator):
            A_op = A
        else:
            A_op = aslinearoperator(A)

        n = len(x)
        row_sums = np.zeros(n, dtype=np.complex128)

        # This is a very slow way to get row sums for a LinearOperator,
        # but necessary without direct matrix access.
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            A_row = A_op.matvec(e_i)
            row_sums[i] = np.sum(np.abs(A_row)) - np.abs(A_row[i])

        diff_result = np.zeros_like(x)

        for i in range(n):
            if row_sums[i] > 1e-12:
                # This is also slow, performing a matvec for each row.
                weighted_sum = 0.0
                e_i = np.zeros(n)
                e_i[i] = 1.0
                A_row = A_op.matvec(e_i)

                for j in range(n):
                    if j != i:
                        weight = np.abs(A_row[j]) / row_sums[i]
                        weighted_sum += weight * x[j]

                diff_result[i] = weighted_sum - x[i]
            else:
                diff_result[i] = 0.0

        return diff_result

    def eigenvector_based_coarsening(self, A):
        """Coarsening algorithm based on eigenvectors (using matvecs)."""
        print("Performing eigenvector-based coarsening...")

        n = A.shape[0]
        if n <= self.max_coarse_size:
            return np.ones(n, dtype=bool)

        # Compute near-kernel eigenvectors
        eigenvectors = self.compute_near_kernel_eigenvectors(A)
        if eigenvectors is None:
            return np.ones(n, dtype=bool)

        # Analyze smoothness
        smoothness = self.analyze_smoothness(A, eigenvectors)

        # Use QR decomposition to select representative points
        c_points = self._qr_based_selection(eigenvectors, smoothness)

        # Ensure a reasonable coarsening ratio
        target_coarse_size = max(
            self.max_coarse_size,
            int(n * self.coarsening_ratio)
        )

        if np.sum(c_points) > target_coarse_size:
            c_points = self._reduce_coarse_points(
                c_points, smoothness, target_coarse_size)
        elif np.sum(c_points) < target_coarse_size * 0.5:
            c_points = self._increase_coarse_points(
                c_points, smoothness, target_coarse_size)

        print(f"  Coarse grid points: {np.sum(c_points)}")
        print(f"  Fine grid points: {np.sum(~c_points)}")
        print(f"  Coarsening ratio: {np.sum(c_points) / n:.3f}")

        return c_points

    def _qr_based_selection(self, eigenvectors, smoothness):
        """Point selection strategy based on QR decomposition."""
        if eigenvectors is None or eigenvectors.size == 0:
            return np.zeros(smoothness.shape, dtype=bool)

        n, k = eigenvectors.shape

        # QR decomposition with pivoting on eigenvectors
        Q, R, pivots = qr(eigenvectors, mode='economic', pivoting=True)

        # Select points based on column pivots from QR
        important_indices = set(pivots[:min(k, n//4)])  # Limit selection count

        # Add additional coarse points based on smoothness
        smooth_points = np.where(smoothness < self.smoothness_threshold)[0]

        # Select smooth points with even distribution
        if len(smooth_points) > 0:
            step = max(1, len(smooth_points) // (n // 20))  # Control density
            selected_smooth = smooth_points[::step]
            important_indices.update(selected_smooth)

        # Create coarse grid marker
        c_points = np.zeros(n, dtype=bool)
        for idx in important_indices:
            if idx < n:
                c_points[idx] = True

        return c_points

    def _reduce_coarse_points(self, c_points, smoothness, target_size):
        """Reduces the number of coarse grid points."""
        current_c_indices = np.where(c_points)[0]
        current_size = len(current_c_indices)

        if current_size <= target_size:
            return c_points

        # Sort by smoothness and keep the most important points
        smoothness_c = smoothness[current_c_indices]
        sorted_indices = np.argsort(smoothness_c)

        # Keep the top 'target_size' most important points
        keep_indices = current_c_indices[sorted_indices[:target_size]]

        new_c_points = np.zeros_like(c_points)
        new_c_points[keep_indices] = True

        return new_c_points

    def _increase_coarse_points(self, c_points, smoothness, target_size):
        """Increases the number of coarse grid points."""
        current_size = np.sum(c_points)
        needed = target_size - current_size

        if needed <= 0:
            return c_points

        # Select the smoothest points from the fine grid points
        f_indices = np.where(~c_points)[0]
        if len(f_indices) == 0:
            return c_points

        smoothness_f = smoothness[f_indices]
        sorted_indices = np.argsort(smoothness_f)

        # Add the smoothest points to the coarse grid
        add_count = min(needed, len(f_indices))
        add_indices = f_indices[sorted_indices[:add_count]]

        new_c_points = c_points.copy()
        new_c_points[add_indices] = True

        return new_c_points


class AMGInterpolation:
    """
    Builds the AMG interpolation operator.
    """

    def __init__(self, truncation_factor=0.2):
        self.truncation_factor = truncation_factor

    def build_interpolation(self, A, c_points):
        """Builds the interpolation operator (using matvecs)."""
        print("Building interpolation operator...")

        n = A.shape[0]
        c_indices = np.where(c_points)[0]
        f_indices = np.where(~c_points)[0]
        nc = len(c_indices)
        nf = len(f_indices)

        print(f"  Fine points: {nf}, Coarse points: {nc}")

        # Create a map from C-point index to coarse grid index
        c_to_coarse = {c_indices[i]: i for i in range(nc)}

        # Build interpolation matrix P: R^nc -> R^n
        P_rows = []
        P_cols = []
        P_data = []

        # Interpolation for C-points is direct injection
        for i, c_idx in enumerate(c_indices):
            P_rows.append(c_idx)
            P_cols.append(i)
            P_data.append(1.0)

        # Interpolation for F-points is based on weighted average of strong C-neighbors
        if isinstance(A, LinearOperator):
            A_op = A
        else:
            A_op = aslinearoperator(A)

        theta = 0.25  # Strength threshold

        for f_idx in f_indices:
            # Get the strong C-neighbors for the F-point
            e_f = np.zeros(n)
            e_f[f_idx] = 1.0
            A_row = A_op.matvec(e_f)

            # Determine strength threshold for this row
            max_val = 0.0
            for j in range(n):
                if j != f_idx and np.abs(A_row[j]) > max_val:
                    max_val = np.abs(A_row[j])

            threshold = theta * max_val

            strong_c_neighbors = []
            for j in range(n):
                if c_points[j] and np.abs(A_row[j]) >= threshold:
                    strong_c_neighbors.append(j)

            # Fallback: if no strong C-neighbors, find the strongest connection
            if len(strong_c_neighbors) == 0:
                max_val = 0.0
                max_j = -1
                for j in range(n):
                    if c_points[j] and np.abs(A_row[j]) > max_val:
                        max_val = np.abs(A_row[j])
                        max_j = j
                if max_j != -1:
                    strong_c_neighbors.append(max_j)

            if len(strong_c_neighbors) > 0:
                # Calculate interpolation weights
                a_ff = A_row[f_idx]
                sum_a_fc = 0.0
                for j in strong_c_neighbors:
                    sum_a_fc += A_row[j]

                if abs(sum_a_fc) > 1e-12:
                    for c_neighbor in strong_c_neighbors:
                        weight = -A_row[c_neighbor] / sum_a_fc
                        coarse_idx = c_to_coarse[c_neighbor]

                        P_rows.append(f_idx)
                        P_cols.append(coarse_idx)
                        P_data.append(weight)
                else:
                    # Degenerate case: equal weights
                    weight = 1.0 / len(strong_c_neighbors)
                    for c_neighbor in strong_c_neighbors:
                        coarse_idx = c_to_coarse[c_neighbor]
                        P_rows.append(f_idx)
                        P_cols.append(coarse_idx)
                        P_data.append(weight)

        # Build the sparse interpolation matrix
        P = csr_matrix((P_data, (P_rows, P_cols)),
                       shape=(n, nc), dtype=np.complex128)

        print(f"  Interpolation matrix shape: {P.shape}")
        print(f"  Interpolation matrix non-zeros: {P.nnz}")

        return P


class FGMRESSmoother:
    """Flexible GMRES (FGMRES) smoother."""

    def __init__(self, max_krylov=5, max_restarts=1, tol=0.1):
        self.max_krylov = max_krylov
        self.max_restarts = max_restarts
        self.tol = tol

    def smooth(self, A, b, x0):
        """Performs FGMRES smoothing."""
        n = len(b)
        x = x0.copy()
        r = b - A(x0)  # Uses matrix-vector multiplication
        r_norm = np.linalg.norm(r)

        if r_norm < 1e-12:
            return x0

        # Store search directions
        V = np.zeros((n, self.max_krylov + 1), dtype=np.complex128)
        # Preconditioned directions
        Z = np.zeros((n, self.max_krylov), dtype=np.complex128)
        H = np.zeros((self.max_krylov + 1, self.max_krylov),
                     dtype=np.complex128)

        # Initial residual vector
        V[:, 0] = r / r_norm

        # Givens rotation storage
        cs = np.zeros(self.max_krylov, dtype=np.complex128)
        sn = np.zeros(self.max_krylov, dtype=np.complex128)
        s = np.zeros(self.max_krylov + 1, dtype=np.complex128)
        s[0] = r_norm

        iters = 0
        for j in range(self.max_krylov):
            iters = j + 1
            # Apply preconditioning (simple diagonal preconditioning here)
            # Preconditioning is disabled for simplicity in this example
            Z[:, j] = V[:, j]  # No preconditioning

            # Matrix-vector multiplication
            w = A(Z[:, j])

            # Arnoldi process
            for i in range(j + 1):
                H[i, j] = np.vdot(V[:, i], w)
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = np.linalg.norm(w)
            if abs(H[j + 1, j]) < 1e-12:
                break

            V[:, j + 1] = w / H[j + 1, j]

            # Apply Givens rotations
            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i].conj() * H[i, j] + \
                    cs[i].conj() * H[i + 1, j]
                H[i, j] = temp

            # Compute new Givens rotation
            h1 = H[j, j]
            h2 = H[j + 1, j]
            if abs(h2) < 1e-12:
                cs[j] = 1.0
                sn[j] = 0.0
            elif abs(h2) > abs(h1):
                t = h1 / h2
                sn[j] = 1.0 / np.sqrt(1.0 + abs(t)**2)
                cs[j] = t * sn[j]
            else:
                t = h2 / h1
                cs[j] = 1.0 / np.sqrt(1.0 + abs(t)**2)
                sn[j] = t * cs[j]

            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0

            # Update s vector
            s[j + 1] = -sn[j].conj() * s[j]
            s[j] = cs[j] * s[j]

            # Check for convergence
            if abs(s[j + 1]) < self.tol * r_norm:
                break

        # Solve the least-squares problem
        if iters > 0:
            y = np.linalg.lstsq(H[:iters, :iters], s[:iters], rcond=None)[0]
            # Update solution
            dx = Z[:, :iters] @ y
            x = x + dx

        return x


class AlgebraicMultigridComplex:
    """
    Algebraic multigrid solver for complex matrices using eigenvector coarsening.
    """

    def __init__(self, max_levels=10, tolerance=1e-8, max_iterations=100,
                 num_eigenvectors=3, max_coarse_size=50,
                 power_iterations=15, chebyshev_degree=5):
        self.max_levels = max_levels
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.num_eigenvectors = num_eigenvectors
        self.max_coarse_size = max_coarse_size
        self.power_iterations = power_iterations
        self.chebyshev_degree = chebyshev_degree
        self.convergence_history = []
        self.bicgstab_history = []

        # Initialize components
        self.coarsening = AMGEigenvectorCoarsening(
            num_eigenvectors=num_eigenvectors,
            max_coarse_size=max_coarse_size,
            power_iterations=power_iterations,
            chebyshev_degree=chebyshev_degree
        )
        self.interpolation = AMGInterpolation()
        self.smoother = FGMRESSmoother()

        # Store multigrid hierarchy
        self.matrices = []
        self.matvec_ops = []
        self.interpolation_ops = []
        self.restriction_ops = []

    def _create_coarse_matvec(self, P, R, fine_matrix):
        """Factory function to create a coarse-grid matvec."""
        def A_coarse_matvec(x):
            if len(x) != P.shape[1]:
                # This indicates a logic error in how the V-cycle is called
                # or how the hierarchy was built.
                raise ValueError(
                    f"Shape mismatch in coarse matvec: input {x.shape} vs P {P.shape}")

            Px = P.dot(x)
            A_Px = fine_matrix.matvec(Px)
            return R.dot(A_Px)
        return A_coarse_matvec

    def setup_hierarchy(self, A):
        """Sets up the AMG hierarchy using eigenvector-based coarsening."""
        print("Setting up AMG hierarchy (Eigenvector Coarsening)...")
        print("=" * 50)

        # Wrap A in a LinearOperator
        if not isinstance(A, LinearOperator):
            A_op = aslinearoperator(A)
        else:
            A_op = A

        self.matrices = [A_op]
        self.matvec_ops = [A_op.matvec]
        self.interpolation_ops = []
        self.restriction_ops = []

        current_matrix = A_op
        level = 0

        while (current_matrix.shape[0] > self.max_coarse_size and
               level < self.max_levels - 1):

            print(f"Level {level}: Matrix size {current_matrix.shape[0]}")

            # Eigenvector-based coarsening
            c_points = self.coarsening.eigenvector_based_coarsening(
                current_matrix)

            if np.sum(c_points) <= 1 or np.sum(c_points) == current_matrix.shape[0]:
                print("  Cannot coarsen further, stopping hierarchy setup.")
                break

            # Build interpolation operator
            P = self.interpolation.build_interpolation(
                current_matrix, c_points)
            self.interpolation_ops.append(P)

            # Build restriction operator (conjugate transpose of interpolation)
            R = P.conj().T
            self.restriction_ops.append(R)

            # Build coarse grid operator A_coarse = R * A * P
            print("  Building coarse grid operator...")

            # Use the factory to correctly bind the operators for this level
            A_coarse_matvec = self._create_coarse_matvec(P, R, current_matrix)

            # Create the coarse grid LinearOperator
            A_coarse = LinearOperator(
                (P.shape[1], P.shape[1]),
                matvec=A_coarse_matvec,
                dtype=np.complex128
            )

            self.matrices.append(A_coarse)
            # Use the matvec from the new operator
            self.matvec_ops.append(A_coarse.matvec)

            current_matrix = A_coarse
            level += 1

            print(f"  Coarse grid size: {A_coarse.shape[0]}")
            if level > 0 and len(self.matrices) > 1:
                prev_size = self.matrices[-2].shape[0]
                curr_size = A_coarse.shape[0]
                print(f"  Coarsening factor: {curr_size / prev_size:.3f}")

        print(f"\nTotal levels: {len(self.matrices)}")
        print("Hierarchy setup complete!")
        print("=" * 50)

    def v_cycle(self, b, x, level=0):
        """A single V-cycle iteration."""
        if level >= len(self.matvec_ops):
            return x

        A_matvec = self.matvec_ops[level]

        if level == len(self.matrices) - 1:
            # Coarsest level: solve directly or with more smoothing
            n_coarse = self.matrices[level].shape[0]

            # Ensure x has the correct length
            if len(x) != n_coarse:
                x_padded = np.zeros(n_coarse, dtype=b.dtype)
                x_padded[:len(x)] = x
                x = x_padded

            # For the coarsest level, it's often better to use a direct solver if possible,
            # or a robust iterative solver like FGMRES for more iterations.
            max_krylov_coarse = min(20, n_coarse - 1)
            if max_krylov_coarse > 0:
                coarse_smoother = FGMRESSmoother(
                    max_krylov=max_krylov_coarse, tol=1e-2)
                return coarse_smoother.smooth(A_matvec, b, x)
            return x  # Cannot solve if space is too small

        # Ensure interpolation/restriction operators exist
        if level >= len(self.interpolation_ops):
            return x

        P = self.interpolation_ops[level]
        R = self.restriction_ops[level]

        # Pre-smoothing
        x = self.smoother.smooth(A_matvec, b, x)

        # Compute residual
        residual = b - A_matvec(x)

        # Restrict residual to coarse grid
        coarse_residual = R.dot(residual)
        n_coarse = len(coarse_residual)
        coarse_error = np.zeros(n_coarse, dtype=coarse_residual.dtype)

        # Recursively solve the coarse grid correction equation
        coarse_error = self.v_cycle(coarse_residual, coarse_error, level + 1)

        # Ensure coarse_error has the correct length after recursion
        if len(coarse_error) != n_coarse:
            error_padded = np.zeros(n_coarse, dtype=coarse_residual.dtype)
            error_padded[:len(coarse_error)] = coarse_error
            coarse_error = error_padded

        # Interpolate correction to fine grid and update solution
        fine_correction = P.dot(coarse_error)
        x = x + fine_correction

        # Post-smoothing
        x = self.smoother.smooth(A_matvec, b, x)

        return x

    def solve(self, A, b, x0=None):
        """Main solver function."""
        print("Starting AMG solve (Eigenvector Coarsening)...")
        print("=" * 60)

        if x0 is None:
            x = np.zeros_like(b, dtype=np.complex128)
        else:
            x = x0.copy()

        # Set up the hierarchy
        self.setup_hierarchy(A)

        # Main iteration loop
        print(f"\nStarting AMG iterations:")
        print("-" * 40)

        start_time = time.time()

        b_norm = np.linalg.norm(b)
        if b_norm == 0:
            b_norm = 1.0  # Avoid division by zero

        for iteration in range(self.max_iterations):
            # Perform a V-cycle
            x = self.v_cycle(b, x, level=0)

            # Compute residual
            residual = b - self.matvec_ops[0](x)
            residual_norm = np.linalg.norm(residual)
            relative_residual = residual_norm / b_norm
            self.convergence_history.append(residual_norm)

            print(
                f"Iteration {iteration + 1:3d}: Residual Norm = {residual_norm:.4e} (Rel: {relative_residual:.4e})")

            # Check for convergence
            if residual_norm < self.tolerance:
                print(f"✓ Convergence reached with tolerance {self.tolerance}")
                break
        else:
            print("⚠ Reached maximum iterations.")

        solve_time = time.time() - start_time

        print(f"\nSolve complete!")
        print(f"Total iterations: {len(self.convergence_history)}")
        if self.convergence_history:
            print(f"Final residual: {self.convergence_history[-1]:.2e}")
        print(f"Solve time: {solve_time:.4f} seconds")
        print("=" * 60)

        return x, solve_time

    def solve_bicgstab(self, A, b, x0=None):
        """Solves the system using BiCGSTAB for comparison."""
        print("\nStarting BiCGSTAB solve...")
        print("=" * 40)

        if x0 is None:
            x0 = np.zeros_like(b)

        # Callback to record residual history
        residuals = []

        def callback(xk):
            r = b - A.dot(xk)
            residuals.append(np.linalg.norm(r))

        start_time = time.time()
        x, info = bicgstab(A, b, x0=x0, callback=callback,
                           atol=self.tolerance, maxiter=self.max_iterations*5)  # Give BiCGSTAB more iters
        solve_time = time.time() - start_time

        if info == 0:
            print(f"✓ BiCGSTAB converged to tolerance {self.tolerance}")
        else:
            print(f"⚠ BiCGSTAB did not converge (status code: {info})")

        final_residual = residuals[-1] if residuals else np.nan
        print(f"Iterations: {len(residuals)}")
        print(f"Final residual: {final_residual:.2e}")
        print(f"Solve time: {solve_time:.4f} seconds")
        print("=" * 40)

        self.bicgstab_history = residuals
        return x, solve_time, residuals

    def create_rhs(self, nx, ny, nz, func_type='sine'):
        """Creates a right-hand side vector."""
        hx = 1.0 / (nx + 1)
        hy = 1.0 / (ny + 1)
        hz = 1.0 / (nz + 1)

        x_coords = np.linspace(hx, 1-hx, nx)
        y_coords = np.linspace(hy, 1-hy, ny)
        z_coords = np.linspace(hz, 1-hz, nz)

        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

        if func_type == 'sine':
            f = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y) * \
                np.sin(2*np.pi*Z) * (1 + 1j)
        elif func_type == 'exponential':
            f = np.exp(X + 1j*Y + 1j*Z)
        else:
            f = np.ones((nx, ny, nz), dtype=np.complex128)

        return f.flatten()

    def verify_solution(self, A, b, x):
        """Verifies the correctness of the solution."""
        print("\nVerifying solution correctness:")
        print("-" * 30)

        # Define matrix-vector multiplication
        if isinstance(A, LinearOperator):
            Ax = A.matvec(x)
        else:
            Ax = A.dot(x)

        residual = Ax - b
        residual_norm = np.linalg.norm(residual)
        b_norm = np.linalg.norm(b)
        relative_error = residual_norm / b_norm if b_norm > 0 else residual_norm

        print(
            f"Solution real part range: [{np.real(x).min():.4f}, {np.real(x).max():.4f}]")
        print(
            f"Solution imag part range: [{np.imag(x).min():.4f}, {np.imag(x).max():.4f}]")
        print(
            f"Solution magnitude range: [{np.abs(x).min():.4f}, {np.abs(x).max():.4f}]")
        print(f"Verification residual norm: {residual_norm:.4e}")
        print(f"Relative error: {relative_error:.2e}")

        if relative_error < 1e-6:
            print("✓ Solution verification PASSED!")
        else:
            print("⚠ Solution may have precision issues.")

        return residual_norm, relative_error

    def plot_convergence(self, amg_time, bicg_time):
        """Plots a comparison of the convergence histories."""
        plt.figure(figsize=(10, 6))

        # AMG convergence history
        if self.convergence_history:
            amg_iter = range(1, len(self.convergence_history) + 1)
            plt.semilogy(amg_iter, self.convergence_history, 'b-o',
                         label=f'AMG (Time: {amg_time:.2f}s)')

        # BiCGSTAB convergence history
        if self.bicgstab_history:
            bicg_iter = range(1, len(self.bicgstab_history) + 1)
            plt.semilogy(bicg_iter, self.bicgstab_history, 'r--s',
                         label=f'BiCGSTAB (Time: {bicg_time:.2f}s)')

        plt.title('Convergence History Comparison')
        plt.xlabel('Iteration')
        plt.ylabel('Residual Norm')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()

        # Save the figure
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        filename = f"Convergence_Comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Convergence plot saved to: {filename}")

# %%
# Main program
if __name__ == "__main__":
    print("Algebraic Multigrid Solver for Complex Matrices Demo")
    print("=" * 50)

    # Problem parameters - using a small grid for testing
    # nx, ny, nz = 16, 16, 12
    nx, ny, nz = params[define._LAT_X_]//define._LAT_P_, params[define._LAT_Y_], params[define._LAT_Z_] * \
        params[define._LAT_T_]*define._LAT_SC_

    # # Build the coefficient matrix
    # matrix_builder = ComplexMatrixBuilder(nx, ny, nz)
    # A = matrix_builder.build_matrix()
    A = LatticeLinearOperator()

    # Create the AMG solver
    solver = AlgebraicMultigridComplex(
        max_levels=10,
        tolerance=1e-8,
        max_iterations=100,
        power_iterations=10,
        chebyshev_degree=4,
        num_eigenvectors=params[define._LAT_E_],
        max_coarse_size=4
    )

    b = solver.create_rhs(nx=nx, ny=ny, nz=nz)

    # Solve using AMG
    x_amg, amg_time = solver.solve(A=A, b=b)

    # Solve using BiCGSTAB
    x_bicg, bicg_time, _ = solver.solve_bicgstab(A=A, b=b)

    # Calculate speedup
    speedup = bicg_time / amg_time if amg_time > 0 else float('inf')
    print(f"\nPerformance Comparison:")
    print(f"AMG solve time: {amg_time:.4f}s")
    print(f"BiCGSTAB solve time: {bicg_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    print("=" * 50)

    # Verify the solution
    solver.verify_solution(A=A, x=x_amg, b=b)

    # Plot convergence history
    solver.plot_convergence(amg_time, bicg_time)

    # Performance statistics
    print(f"\nPerformance Summary:")
    print(f"Grid size: {nx}x{ny}x{nz}")
    print(f"Number of unknowns: {nx*ny*nz}")
    print(f"AMG convergence iterations: {len(solver.convergence_history)}")
    print(f"BiCGSTAB convergence iterations: {len(solver.bicgstab_history)}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"\n{'='*80}")
    print("All tests complete!")
    print(f"{'='*80}")


