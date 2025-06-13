import time
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
np.Inf = np.inf


class MatrixVectorOperator:
    def __init__(self, nx, ny, nz, alpha=1.0, beta=1.0j):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.alpha = alpha
        self.beta = beta
        self.n = nx * ny * nz
        self.hx = 1.0 / (nx + 1)
        self.hy = 1.0 / (ny + 1)
        self.hz = 1.0 / (nz + 1)
        self.main_diag = (2 * alpha * (1/self.hx**2 + 1/self.hy**2 +
                          1/self.hz**2) + beta) * np.ones(self.n, dtype=np.complex128)

    def matvec(self, v):
        result = np.zeros_like(v)
        nx, ny, nz = self.nx, self.ny, self.nz
        result[:] = self.main_diag * v
        v_3d = v.reshape((nz, ny, nx))
        result_3d = result.reshape((nz, ny, nx))
        for k in range(nz):
            for i in range(ny):
                for j in range(nx):
                    idx = k * ny * nx + i * nx + j
                    if j > 0:
                        result_3d[k, i, j] -= (self.alpha /
                                               self.hx**2) * v_3d[k, i, j-1]
                    if j < nx - 1:
                        result_3d[k, i, j] -= (self.alpha /
                                               self.hx**2) * v_3d[k, i, j+1]
                    if i > 0:
                        result_3d[k, i, j] -= (self.alpha /
                                               self.hy**2) * v_3d[k, i-1, j]
                    if i < ny - 1:
                        result_3d[k, i, j] -= (self.alpha /
                                               self.hy**2) * v_3d[k, i+1, j]
                    if k > 0:
                        result_3d[k, i, j] -= (self.alpha /
                                               self.hz**2) * v_3d[k-1, i, j]
                    if k < nz - 1:
                        result_3d[k, i, j] -= (self.alpha /
                                               self.hz**2) * v_3d[k+1, i, j]
        return result_3d.flatten()

    def diagonal(self):
        return self.main_diag.copy()


class GMRESSmoother:
    def __init__(self, max_krylov=5, max_restarts=1, tol=0.1):
        self.max_krylov = max_krylov
        self.max_restarts = max_restarts
        self.tol = tol

    def smooth(self, op, b, x0):
        x = x0.copy()
        r = b - op.matvec(x0)
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-12:
            return x0
        for _ in range(self.max_restarts):
            Q, H = self._arnoldi(op, r, r_norm)
            e1 = np.zeros(H.shape[1] + 1, dtype=b.dtype)
            e1[0] = r_norm
            y = self._solve_least_squares(H, e1)
            dx = Q[:, :-1] @ y
            x = x + dx
            new_r = b - op.matvec(x)
            new_r_norm = np.linalg.norm(new_r)
            if new_r_norm < self.tol * r_norm:
                break
            r = new_r
            r_norm = new_r_norm
        return x

    def _arnoldi(self, op, r0, r_norm):
        m = self.max_krylov
        n = len(r0)
        Q = np.zeros((n, m+1), dtype=r0.dtype)
        H = np.zeros((m+1, m), dtype=r0.dtype)
        Q[:, 0] = r0 / r_norm
        for j in range(m):
            w = op.matvec(Q[:, j])
            for i in range(j+1):
                H[i, j] = np.vdot(Q[:, i], w)
                w = w - H[i, j] * Q[:, i]
            h_norm = np.linalg.norm(w)
            H[j+1, j] = h_norm
            if h_norm < 1e-12:
                return Q[:, :j+1], H[:j+1, :j]
            if j < m:
                Q[:, j+1] = w / h_norm
        return Q, H

    def _solve_least_squares(self, H, e1):
        m = H.shape[1]
        h_height = H.shape[0]
        R = np.zeros((h_height, m+1), dtype=H.dtype)
        R[:, :m] = H
        R[:, m] = e1[:h_height]
        Q, R_qr = np.linalg.qr(R, mode='complete')
        y = np.linalg.solve(R_qr[:m, :m], Q[:m, :m].conj().T @ e1[:m])
        return y


class AdaptiveMultigridComplex:
    def __init__(self, nx, ny, nz=1, max_levels=5, tolerance=1e-8, max_iterations=100, dtype=np.complex128):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.max_levels = max_levels
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.dtype = dtype
        self.convergence_history = []
        self.level_info = []
        self.gmres_smoother = GMRESSmoother(
            max_krylov=5, max_restarts=1, tol=0.1)

    def create_operator(self, nx, ny, nz, alpha=1.0, beta=1.0j):
        print(f"创建 {nx}x{ny}x{nz} 复数算子...")
        print(f"  系数: α = {alpha}, β = {beta}")
        return MatrixVectorOperator(nx, ny, nz, alpha, beta)

    def create_rhs(self, nx, ny, nz, func_type='sine'):
        hx = 1.0 / (nx + 1)
        hy = 1.0 / (ny + 1)
        hz = 1.0 / (nz + 1)
        x = np.linspace(hx, 1-hx, nx)
        y = np.linspace(hy, 1-hy, ny)
        z = np.linspace(hz, 1-hz, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        if func_type == 'sine':
            f = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y) * \
                np.sin(2*np.pi*Z) * (1 + 1j)
        elif func_type == 'exponential':
            f = np.exp(X + 1j*Y + 1j*Z)
        else:
            f = np.ones((nx, ny, nz), dtype=self.dtype)
        return f.flatten()

    def restrict(self, u_fine, nx_fine, ny_fine, nz_fine):
        if nx_fine < 2 or ny_fine < 2:
            return u_fine
        nx_coarse = max(2, nx_fine // 2)
        ny_coarse = max(2, ny_fine // 2)
        nz_coarse = nz_fine
        u_fine_3d = u_fine.reshape((nz_fine, ny_fine, nx_fine))
        u_coarse_3d = np.zeros(
            (nz_coarse, ny_coarse, nx_coarse), dtype=self.dtype)
        for k in range(nz_fine):
            u_fine_slice = u_fine_3d[k, :, :]
            u_coarse_slice = np.zeros((ny_coarse, nx_coarse), dtype=self.dtype)
            for i in range(ny_coarse):
                for j in range(nx_coarse):
                    ii, jj = 2*i, 2*j
                    weight_sum = 0
                    value_sum = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = ii + di, jj + dj
                            if 0 <= ni < ny_fine and 0 <= nj < nx_fine:
                                if di == 0 and dj == 0:
                                    weight = 1/4
                                elif di == 0 or dj == 0:
                                    weight = 1/8
                                else:
                                    weight = 1/16
                                weight_sum += weight
                                value_sum += weight * u_fine_slice[ni, nj]
                    u_coarse_slice[i, j] = value_sum / \
                        weight_sum if weight_sum > 0 else 0
            u_coarse_3d[k, :, :] = u_coarse_slice
        return u_coarse_3d.flatten()

    def prolongate(self, u_coarse, nx_fine, ny_fine, nz_fine):
        nx_coarse = nx_fine // 2
        ny_coarse = ny_fine // 2
        nz_coarse = nz_fine
        u_coarse_3d = u_coarse.reshape((nz_coarse, ny_coarse, nx_coarse))
        u_fine_3d = np.zeros((nz_fine, ny_fine, nx_fine), dtype=self.dtype)
        for k in range(nz_fine):
            u_coarse_slice = u_coarse_3d[k, :, :]
            u_fine_slice = np.zeros((ny_fine, nx_fine), dtype=self.dtype)
            for i in range(ny_fine):
                for j in range(nx_fine):
                    i_c = i / 2.0
                    j_c = j / 2.0
                    i0, j0 = int(i_c), int(j_c)
                    i1 = min(i0 + 1, ny_coarse - 1)
                    j1 = min(j0 + 1, nx_coarse - 1)
                    wx = i_c - i0
                    wy = j_c - j0
                    u_fine_slice[i, j] = (1 - wx) * (1 - wy) * u_coarse_slice[i0, j0] + \
                                         (1 - wx) * wy * u_coarse_slice[i0, j1] + \
                        wx * (1 - wy) * u_coarse_slice[i1, j0] + \
                        wx * wy * u_coarse_slice[i1, j1]
            u_fine_3d[k, :, :] = u_fine_slice
        return u_fine_3d.flatten()

    def smooth(self, op, b, u, num_iterations=1, method='gmres'):
        return self.gmres_smoother.smooth(op, b, u)

    def compute_residual(self, op, b, u):
        return b - op.matvec(u)

    def bistabcg_solver(self, op, b, x0=None, tol=1e-10, maxiter=1000):
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        r = b - op.matvec(x)
        r0 = r.copy()
        rho = 1.0
        alpha = 1.0
        omega = 1.0
        v = np.zeros_like(b)
        p = np.zeros_like(b)
        for i in range(maxiter):
            rho1 = np.vdot(r0, r)
            beta = (rho1 / rho) * (alpha / omega)
            p = r + beta * (p - omega * v)
            v = op.matvec(p)
            alpha = rho1 / np.vdot(r0, v)
            s = r - alpha * v
            t = op.matvec(s)
            omega = np.vdot(t, s) / np.vdot(t, t)
            x = x + alpha * p + omega * s
            r = s - omega * t
            residual_norm = np.linalg.norm(r)
            if residual_norm < tol:
                return x, 0
            rho = rho1
        return x, 1

    def v_cycle(self, op_hierarchy, b_hierarchy, u_hierarchy, grid_params, level=0):
        current_level_idx = len(op_hierarchy) - 1 - level
        nx, ny, nz = grid_params[current_level_idx]
        print(
            f"V-循环 level {level}, 当前层索引: {current_level_idx}, 网格大小: {nx}x{ny}x{nz}")
        op = op_hierarchy[current_level_idx]
        b = b_hierarchy[current_level_idx]
        u = u_hierarchy[current_level_idx]
        if current_level_idx == 0 or level >= self.max_levels - 1:
            residual = self.compute_residual(op, b, u)
            residual_norm = np.linalg.norm(residual)
            print(f"    前残差范数: {residual_norm:.4e}")
            print(f"    最粗网格直接求解...")
            u_coarse, info = self.bistabcg_solver(
                op, b, u, tol=self.tolerance, maxiter=1000)
            if info != 0:
                print(f"    警告: 最粗网格求解未收敛! Info: {info}")
            u_hierarchy[current_level_idx] = u_coarse
            residual = self.compute_residual(
                op, b, u_hierarchy[current_level_idx])
            residual_norm = np.linalg.norm(residual)
            print(f"    残差范数: {residual_norm:.4e}")
            return u_hierarchy[current_level_idx]
        residual_before_smooth = self.compute_residual(op, b, u)
        residual_norm_before_smooth = np.linalg.norm(residual_before_smooth)
        print(f"    前光滑前残差范数: {residual_norm_before_smooth:.4e}")
        print(f"    前光滑...")
        u = self.smooth(op, b, u)
        u_hierarchy[current_level_idx] = u
        residual = self.compute_residual(op, b, u_hierarchy[current_level_idx])
        residual_norm = np.linalg.norm(residual)
        print(f"    前光滑后残差范数: {residual_norm:.4e}")
        if current_level_idx > 0:
            r_coarse = self.restrict(residual, nx, ny, nz)
            b_hierarchy[current_level_idx - 1] = r_coarse
            u_hierarchy[current_level_idx -
                        1] = np.zeros_like(r_coarse, dtype=self.dtype)
            e_coarse = self.v_cycle(
                op_hierarchy, b_hierarchy, u_hierarchy, grid_params, level + 1)
            nx_fine, ny_fine, nz_fine = grid_params[current_level_idx]
            e_fine = self.prolongate(e_coarse, nx_fine, ny_fine, nz_fine)
            u = u + e_fine
            u_hierarchy[current_level_idx] = u
        residual_before_post_smooth = self.compute_residual(op, b, u)
        residual_norm_before_post_smooth = np.linalg.norm(
            residual_before_post_smooth)
        print(f"    后光滑前残差范数: {residual_norm_before_post_smooth:.4e}")
        print(f"    后光滑...")
        u = self.smooth(op, b, u)
        u_hierarchy[current_level_idx] = u
        residual = self.compute_residual(op, b, u_hierarchy[current_level_idx])
        residual_norm = np.linalg.norm(residual)
        print(f"    后光滑后残差范数: {residual_norm:.4e}")
        return u

    def adaptive_criterion(self, residual_norms):
        if len(residual_norms) < 3:
            return False
        conv_rate = residual_norms[-1] / \
            residual_norms[-2] if residual_norms[-2] != 0 else 1
        return conv_rate > 0.8

    def solve(self, alpha=1.0, beta=1.0j, func_type='sine'):
        print("="*60)
        print("开始自适应多重网格复数求解")
        print("="*60)
        start_time = time.time()
        grid_params = []
        current_nx, current_ny = self.nx, self.ny
        current_nz = self.nz
        print(f"构建网格层次结构:")
        while min(current_nx, current_ny) >= 4 and len(grid_params) < self.max_levels:
            grid_params.append((current_nx, current_ny, current_nz))
            print(
                f"  Level {len(grid_params)-1}: {current_nx}x{current_ny}x{current_nz}")
            current_nx = max(2, current_nx // 2)
            current_ny = max(2, current_ny // 2)
        num_levels = len(grid_params)
        print(f"总共 {num_levels} 层网格")
        print(f"\n构建各层系统算子:")
        op_hierarchy = []
        b_hierarchy = []
        u_hierarchy = []
        for i, (nx, ny, nz) in enumerate(grid_params):
            print(f"Level {i} ({nx}x{ny}x{nz}):")
            op = self.create_operator(nx, ny, nz, alpha, beta)
            b = self.create_rhs(nx, ny, nz, func_type)
            u = np.zeros(nx * ny * nz, dtype=self.dtype)
            op_hierarchy.append(op)
            b_hierarchy.append(b)
            u_hierarchy.append(u)
        op_hierarchy.reverse()
        b_hierarchy.reverse()
        u_hierarchy.reverse()
        grid_params.reverse()
        print(f"\n开始多重网格迭代:")
        print("-" * 40)
        for iteration in range(self.max_iterations):
            print(f"\n迭代 {iteration + 1}:")
            u_hierarchy[-1] = self.v_cycle(op_hierarchy,
                                           b_hierarchy, u_hierarchy, grid_params)
            op_finest = op_hierarchy[-1]
            b_finest = b_hierarchy[-1]
            u_finest = u_hierarchy[-1]
            finest_residual = self.compute_residual(
                op_finest, b_finest, u_finest)
            residual_norm = np.linalg.norm(finest_residual)
            self.convergence_history.append(residual_norm)
            print(f"  迭代 {iteration + 1} 完成，残差范数: {residual_norm:.4e}")
            if residual_norm < self.tolerance:
                print(f"  ✓ 收敛达到容差 {self.tolerance}")
                break
            if self.adaptive_criterion(self.convergence_history):
                print(f"  注意: 收敛较慢，可能需要更多网格层")
        else:
            print("  警告: 达到最大迭代次数，可能未收敛")
        solve_time = time.time() - start_time
        print("\n" + "="*60)
        print("求解完成!")
        print(f"总迭代次数: {len(self.convergence_history)}")
        print(f"最终残差: {self.convergence_history[-1]:.2e}")
        print(f"求解时间: {solve_time:.4f} 秒")
        print("="*60)
        return u_hierarchy[-1].reshape((self.nz, self.ny, self.nx))

    def verify_solution(self, solution, alpha=1.0, beta=1.0j, func_type='sine'):
        print("\n验证解的正确性:")
        print("-" * 30)
        op = self.create_operator(self.nx, self.ny, self.nz, alpha, beta)
        b = self.create_rhs(self.nx, self.ny, self.nz, func_type)
        u_flat = solution.flatten()
        residual = op.matvec(u_flat) - b
        residual_norm = np.linalg.norm(residual)
        relative_error = residual_norm / np.linalg.norm(b)
        print(
            f"解的实部范围: [{np.real(solution).min():.4f}, {np.real(solution).max():.4f}]")
        print(
            f"解的虚部范围: [{np.imag(solution).min():.4f}, {np.imag(solution).max():.4f}]")
        print(
            f"解的模长范围: [{np.abs(solution).min():.4f}, {np.abs(solution).max():.4f}]")
        print(f"验证残差范数: {residual_norm:.4e}")
        print(f"相对误差: {relative_error:.2e}")
        if relative_error < 1e-6:
            print("✓ 解验证通过!")
        else:
            print("⚠ 解可能存在精度问题")
        return residual_norm, relative_error


if __name__ == "__main__":
    print("自适应多重网格复数求解器演示")
    print("=" * 50)
    nx = 32
    ny = 64
    nz = 8
    solver = AdaptiveMultigridComplex(
        nx=nx, ny=ny, nz=nz, max_levels=10, tolerance=1e-8, max_iterations=1000, dtype=np.complex128)
    test_cases = [
        {"alpha": 1.0, "beta": 1.0j, "func_type": "sine",
            "name": "复数椭圆问题 (正弦右端项)"},
        {"alpha": 2.0, "beta": 0.5j, "func_type": "exponential",
            "name": "修正复数问题 (指数右端项)"}
    ]
    for i, case in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"测试案例 {i+1}: {case['name']}")
        print(f"{'='*80}")
        solution = solver.solve(
            alpha=case["alpha"], beta=case["beta"], func_type=case["func_type"])
        residual_norm, relative_error = solver.verify_solution(
            solution, alpha=case["alpha"], beta=case["beta"], func_type=case["func_type"]
        )
        print(f"\n性能统计:")
        print(f"网格大小: {nx}x{ny}x{nz}")
        print(f"未知数个数: {nx*ny*nz}")
        print(f"收敛迭代次数: {len(solver.convergence_history)}")
        print(f"最终残差: {solver.convergence_history[-1]:.2e}")
        plt.title(
            f'Adaptive Multigrid Complex Solution Results', fontsize=16)
        plt.semilogy(range(1, len(solver.convergence_history) + 1),
                     solver.convergence_history, 'b-o', markersize=4)
        plt.tight_layout()
        solve_time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        plt.savefig(
            f"Adaptive_Multigrid_Complex_Solution_Results_{solve_time_str}.png", dpi=300)
        solver.convergence_history = []
    print(f"\n{'='*80}")
    print("所有测试完成!")
    print(f"{'='*80}")
