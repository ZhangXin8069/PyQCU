import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

np.Inf = np.inf


class ComplexMatrixBuilder:
    """
    构建复数系数矩阵的工具类
    """

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

    def build_matrix(self):
        """构建稀疏矩阵表示"""
        print(f"构建 {self.nx}x{self.ny}x{self.nz} 复数系数矩阵...")

        # 计算系数
        cx = self.alpha / (self.hx**2)
        cy = self.alpha / (self.hy**2)
        cz = self.alpha / (self.hz**2)

        # 主对角线系数
        main_diag = 2 * (cx + cy + cz) + self.beta

        # 构建稀疏矩阵
        diagonals = []
        offsets = []

        # 主对角线
        diagonals.append(np.full(self.n, main_diag, dtype=np.complex128))
        offsets.append(0)

        # x方向相邻点 (offset = ±1)
        x_diag = np.full(self.n - 1, -cx, dtype=np.complex128)
        # 排除跨越x边界的连接
        for i in range(1, self.n):
            if i % self.nx == 0:
                x_diag[i-1] = 0
        diagonals.extend([x_diag, x_diag])
        offsets.extend([1, -1])

        # y方向相邻点 (offset = ±nx)
        if self.nx < self.n:
            y_diag = np.full(self.n - self.nx, -cy, dtype=np.complex128)
            diagonals.extend([y_diag, y_diag])
            offsets.extend([self.nx, -self.nx])

        # z方向相邻点 (offset = ±nx*ny)
        if self.nx * self.ny < self.n:
            z_diag = np.full(self.n - self.nx * self.ny, -
                             cz, dtype=np.complex128)
            diagonals.extend([z_diag, z_diag])
            offsets.extend([self.nx * self.ny, -self.nx * self.ny])

        # 构建稀疏矩阵
        A = diags(diagonals, offsets, shape=(self.n, self.n),
                  format='csr', dtype=np.complex128)

        print(f"  矩阵维度: {A.shape}")
        print(f"  非零元素: {A.nnz}")
        print(f"  稀疏度: {A.nnz / (self.n**2) * 100:.2f}%")

        return A


class AMGCoarsening:
    """
    代数多重网格粗化策略
    """

    def __init__(self, strength_threshold=0.25, max_coarse_size=100):
        self.strength_threshold = strength_threshold
        self.max_coarse_size = max_coarse_size

    def strength_of_connection(self, A):
        """计算强连接图"""
        n = A.shape[0]
        A_abs = A.copy()
        A_abs.data = np.abs(A_abs.data)

        # 计算每行的最大非对角元素
        S = A_abs.copy()
        S.setdiag(0)  # 移除对角元素

        # 强连接条件: |a_ij| >= theta * max_k(|a_ik|) for k != i
        row_max = np.array([S.getrow(i).max() for i in range(n)])

        # 构建强连接矩阵
        strong_connections = csr_matrix((n, n), dtype=bool)

        for i in range(n):
            row = S.getrow(i)
            threshold = self.strength_threshold * row_max[i]
            if threshold > 0:
                strong_mask = row.data >= threshold
                if np.any(strong_mask):
                    row_indices = row.indices[strong_mask]
                    for j in row_indices:
                        strong_connections[i, j] = True

        return strong_connections

    def classical_coarsening(self, A):
        """经典RS粗化算法"""
        print("执行经典RS粗化...")

        n = A.shape[0]
        if n <= self.max_coarse_size:
            # 如果矩阵已经足够小，所有点都是粗网格点
            return np.ones(n, dtype=bool)

        # 计算强连接
        S = self.strength_of_connection(A)

        # 初始化点的状态: 0=未定义, 1=C点(粗网格), -1=F点(细网格)
        point_type = np.zeros(n, dtype=int)

        # 计算每个点的λ值(强依赖它的点的数量)
        lambda_values = np.array([S.getcol(i).nnz for i in range(n)])

        # 按λ值降序排列点
        sorted_indices = np.argsort(-lambda_values)

        for idx in sorted_indices:
            if point_type[idx] != 0:  # 已经确定类型
                continue

            # 获取强依赖idx的点
            strong_neighbors = S.getcol(idx).indices

            # 检查是否所有强邻居都有C点邻居
            can_be_f_point = True
            for neighbor in strong_neighbors:
                if point_type[neighbor] == 0:  # 邻居未确定
                    # 检查邻居是否有C点邻居
                    neighbor_strong = S.getrow(neighbor).indices
                    has_c_neighbor = np.any(point_type[neighbor_strong] == 1)
                    if not has_c_neighbor:
                        can_be_f_point = False
                        break

            if can_be_f_point:
                point_type[idx] = -1  # F点
                # 将未确定的强邻居设为C点
                for neighbor in strong_neighbors:
                    if point_type[neighbor] == 0:
                        point_type[neighbor] = 1
            else:
                point_type[idx] = 1  # C点
                # 将未确定的强邻居倾向于设为F点
                for neighbor in strong_neighbors:
                    if point_type[neighbor] == 0:
                        # 检查是否可以安全设为F点
                        neighbor_strong = S.getrow(neighbor).indices
                        has_other_c = np.any(point_type[neighbor_strong] == 1)
                        if has_other_c:
                            point_type[neighbor] = -1

        # 确保所有未定义的点都被分类
        undefined_mask = (point_type == 0)
        if np.any(undefined_mask):
            # 简单策略：将剩余未定义点设为C点
            point_type[undefined_mask] = 1

        c_points = (point_type == 1)
        f_points = (point_type == -1)

        print(f"  粗网格点数: {np.sum(c_points)}")
        print(f"  细网格点数: {np.sum(f_points)}")
        print(f"  粗化比: {np.sum(c_points) / n:.3f}")

        return c_points


class AMGInterpolation:
    """
    代数多重网格插值算子构建
    """

    def __init__(self, truncation_factor=0.2):
        self.truncation_factor = truncation_factor

    def build_interpolation(self, A, c_points):
        """构建插值算子"""
        print("构建插值算子...")

        n = A.shape[0]
        c_indices = np.where(c_points)[0]
        f_indices = np.where(~c_points)[0]
        nc = len(c_indices)
        nf = len(f_indices)

        print(f"  细网格点数: {nf}, 粗网格点数: {nc}")

        # 创建C点到粗网格索引的映射
        c_to_coarse = {c_indices[i]: i for i in range(nc)}

        # 构建插值矩阵 P: R^nc -> R^n
        P_rows = []
        P_cols = []
        P_data = []

        # C点的插值：直接注入
        for i, c_idx in enumerate(c_indices):
            P_rows.append(c_idx)
            P_cols.append(i)
            P_data.append(1.0)

        # F点的插值：基于强连接的加权平均
        S = self._strength_of_connection(A, 0.25)

        for f_idx in f_indices:
            # 获取F点的强C邻居
            row = A.getrow(f_idx)
            strong_c_neighbors = []

            for j in row.indices:
                if c_points[j] and S[f_idx, j]:
                    strong_c_neighbors.append(j)

            if len(strong_c_neighbors) == 0:
                # 如果没有强C邻居，寻找最近的C点
                for j in row.indices:
                    if c_points[j]:
                        strong_c_neighbors.append(j)
                        break

            if len(strong_c_neighbors) > 0:
                # 计算插值权重
                a_ff = A[f_idx, f_idx]
                sum_a_fc = sum(A[f_idx, j] for j in strong_c_neighbors)

                if abs(sum_a_fc) > 1e-12:
                    for c_neighbor in strong_c_neighbors:
                        weight = -A[f_idx, c_neighbor] / sum_a_fc
                        coarse_idx = c_to_coarse[c_neighbor]

                        P_rows.append(f_idx)
                        P_cols.append(coarse_idx)
                        P_data.append(weight)
                else:
                    # 退化情况：等权重分配
                    weight = 1.0 / len(strong_c_neighbors)
                    for c_neighbor in strong_c_neighbors:
                        coarse_idx = c_to_coarse[c_neighbor]
                        P_rows.append(f_idx)
                        P_cols.append(coarse_idx)
                        P_data.append(weight)

        # 构建稀疏插值矩阵
        P = csr_matrix((P_data, (P_rows, P_cols)),
                       shape=(n, nc), dtype=np.complex128)

        print(f"  插值矩阵形状: {P.shape}")
        print(f"  插值矩阵非零元素: {P.nnz}")

        return P

    def _strength_of_connection(self, A, theta=0.25):
        """辅助函数：计算强连接"""
        n = A.shape[0]
        A_abs = A.copy()
        A_abs.data = np.abs(A_abs.data)

        S = A_abs.copy()
        S.setdiag(0)

        strong_connections = csr_matrix((n, n), dtype=bool)

        for i in range(n):
            row = S.getrow(i)
            if row.nnz > 0:
                threshold = theta * row.max()
                strong_mask = row.data >= threshold
                if np.any(strong_mask):
                    row_indices = row.indices[strong_mask]
                    for j in row_indices:
                        strong_connections[i, j] = True

        return strong_connections


class GMRESSmoother:
    """GMRES光滑器"""

    def __init__(self, max_krylov=5, max_restarts=1, tol=0.1):
        self.max_krylov = max_krylov
        self.max_restarts = max_restarts
        self.tol = tol

    def smooth(self, A, b, x0):
        """执行GMRES光滑"""
        x = x0.copy()
        r = b - A @ x0
        r_norm = np.linalg.norm(r)

        if r_norm < 1e-12:
            return x0

        for _ in range(self.max_restarts):
            Q, H = self._arnoldi(A, r, r_norm)

            e1 = np.zeros(H.shape[1] + 1, dtype=b.dtype)
            e1[0] = r_norm
            y = self._solve_least_squares(H, e1)

            dx = Q[:, :-1] @ y
            x = x + dx

            new_r = b - A @ x
            new_r_norm = np.linalg.norm(new_r)
            if new_r_norm < self.tol * r_norm:
                break

            r = new_r
            r_norm = new_r_norm

        return x

    def _arnoldi(self, A, r0, r_norm):
        """Arnoldi过程"""
        m = self.max_krylov
        n = len(r0)
        Q = np.zeros((n, m+1), dtype=r0.dtype)
        H = np.zeros((m+1, m), dtype=r0.dtype)

        Q[:, 0] = r0 / r_norm

        for j in range(m):
            w = A @ Q[:, j]

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
        """求解最小二乘问题"""
        m = H.shape[1]
        h_height = H.shape[0]

        R = np.zeros((h_height, m+1), dtype=H.dtype)
        R[:, :m] = H
        R[:, m] = e1[:h_height]

        Q, R_qr = np.linalg.qr(R, mode='complete')
        y = np.linalg.solve(R_qr[:m, :m], Q[:m, :m].conj().T @ e1[:m])
        return y


class AlgebraicMultigridComplex:
    """
    代数多重网格复数求解器
    """

    def __init__(self, max_levels=10, tolerance=1e-8, max_iterations=100,
                 strength_threshold=0.25, max_coarse_size=50):
        self.max_levels = max_levels
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.strength_threshold = strength_threshold
        self.max_coarse_size = max_coarse_size
        self.convergence_history = []

        # 初始化组件
        self.coarsening = AMGCoarsening(strength_threshold, max_coarse_size)
        self.interpolation = AMGInterpolation()
        self.smoother = GMRESSmoother()

        # 存储多重网格层次结构
        self.matrices = []
        self.interpolation_ops = []
        self.restriction_ops = []

    def setup_hierarchy(self, A):
        """建立AMG层次结构"""
        print("建立AMG层次结构...")
        print("=" * 50)

        self.matrices = [A]
        self.interpolation_ops = []
        self.restriction_ops = []

        current_matrix = A
        level = 0

        while (current_matrix.shape[0] > self.max_coarse_size and
               level < self.max_levels - 1):

            print(f"Level {level}: 矩阵大小 {current_matrix.shape[0]}")

            # 粗化
            c_points = self.coarsening.classical_coarsening(current_matrix)

            if np.sum(c_points) == 0 or np.sum(c_points) == current_matrix.shape[0]:
                print("  无法进一步粗化，停止层次建立")
                break

            # 构建插值算子
            P = self.interpolation.build_interpolation(
                current_matrix, c_points)
            self.interpolation_ops.append(P)

            # 构建限制算子 (插值算子的共轭转置)
            R = P.conj().T
            self.restriction_ops.append(R)

            # 构建粗网格算子 A_coarse = R * A * P
            print("  构建粗网格算子...")
            A_coarse = R @ current_matrix @ P
            self.matrices.append(A_coarse)

            current_matrix = A_coarse
            level += 1

            print(f"  粗网格大小: {A_coarse.shape[0]}")
            print(
                f"  粗化比: {A_coarse.shape[0] / self.matrices[-2].shape[0]:.3f}")

        print(f"\n总层数: {len(self.matrices)}")
        print("层次结构建立完成!")
        print("=" * 50)

    def v_cycle(self, b, x, level=0):
        """V-循环迭代"""
        if level == len(self.matrices) - 1:
            # 最粗层：直接求解
            A_coarse = self.matrices[level]
            if A_coarse.shape[0] <= 200:  # 小矩阵直接求解
                try:
                    x[:] = spsolve(A_coarse, b)
                except:
                    # 如果直接求解失败，使用迭代方法
                    x[:] = self.smoother.smooth(A_coarse, b, x)
            else:
                # 大矩阵使用光滑器
                x[:] = self.smoother.smooth(A_coarse, b, x)
            return x

        A = self.matrices[level]
        P = self.interpolation_ops[level]
        R = self.restriction_ops[level]

        # 前光滑
        x = self.smoother.smooth(A, b, x)

        # 计算残差
        residual = b - A @ x

        # 限制残差到粗网格
        coarse_residual = R @ residual
        coarse_error = np.zeros(
            coarse_residual.shape[0], dtype=coarse_residual.dtype)

        # 递归求解粗网格修正方程
        coarse_error = self.v_cycle(coarse_residual, coarse_error, level + 1)

        # 插值修正到细网格
        fine_correction = P @ coarse_error
        x = x + fine_correction

        # 后光滑
        x = self.smoother.smooth(A, b, x)

        return x

    def solve(self, A, b, x0=None):
        """主求解函数"""
        print("开始AMG求解...")
        print("=" * 60)

        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()

        # 建立层次结构
        self.setup_hierarchy(A)

        # 主迭代循环
        print(f"\n开始AMG迭代:")
        print("-" * 40)

        start_time = time.time()

        for iteration in range(self.max_iterations):
            # 执行V-循环
            x = self.v_cycle(b, x)

            # 计算残差
            residual = b - A @ x
            residual_norm = np.linalg.norm(residual)
            self.convergence_history.append(residual_norm)

            print(f"迭代 {iteration + 1:3d}: 残差范数 = {residual_norm:.4e}")

            # 检查收敛
            if residual_norm < self.tolerance:
                print(f"✓ 收敛达到容差 {self.tolerance}")
                break
        else:
            print("⚠ 达到最大迭代次数")

        solve_time = time.time() - start_time

        print(f"\n求解完成!")
        print(f"总迭代次数: {len(self.convergence_history)}")
        print(f"最终残差: {self.convergence_history[-1]:.2e}")
        print(f"求解时间: {solve_time:.4f} 秒")
        print("=" * 60)

        return x

    def create_rhs(self, nx, ny, nz, func_type='sine'):
        """创建右端项"""
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
            f = np.ones((nx, ny, nz), dtype=np.complex128)

        return f.flatten()

    def verify_solution(self, A, b, x):
        """验证解的正确性"""
        print("\n验证解的正确性:")
        print("-" * 30)

        residual = A @ x - b
        residual_norm = np.linalg.norm(residual)
        relative_error = residual_norm / np.linalg.norm(b)

        print(f"解的实部范围: [{np.real(x).min():.4f}, {np.real(x).max():.4f}]")
        print(f"解的虚部范围: [{np.imag(x).min():.4f}, {np.imag(x).max():.4f}]")
        print(f"解的模长范围: [{np.abs(x).min():.4f}, {np.abs(x).max():.4f}]")
        print(f"验证残差范数: {residual_norm:.4e}")
        print(f"相对误差: {relative_error:.2e}")

        if relative_error < 1e-6:
            print("✓ 解验证通过!")
        else:
            print("⚠ 解可能存在精度问题")

        return residual_norm, relative_error

    def plot_results(self, solution, nx, ny, nz):
        """可视化结果"""
        print("\n生成可视化图像...")

        solution_3d = solution.reshape((nz, ny, nx))

        if nz > 1:
            # 3D情况：显示中间切片
            mid_z = nz // 2
            solution_2d = solution_3d[mid_z, :, :]
            title_suffix = f" (Z-slice at k={mid_z})"
        else:
            # 2D情况
            solution_2d = solution_3d[0, :, :]
            title_suffix = ""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f'AMG Complex Solution Results{title_suffix}', fontsize=16)

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # 实部
        im1 = axes[0, 0].contourf(X, Y, np.real(
            solution_2d), levels=20, cmap='RdBu_r')
        axes[0, 0].set_title('Solution - Real Part')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 0])

        # 虚部
        im2 = axes[0, 1].contourf(X, Y, np.imag(
            solution_2d), levels=20, cmap='RdBu_r')
        axes[0, 1].set_title('Solution - Imaginary Part')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0, 1])

        # 模长
        im3 = axes[0, 2].contourf(X, Y, np.abs(
            solution_2d), levels=20, cmap='viridis')
        axes[0, 2].set_title('Solution - Magnitude')
        axes[0, 2].set_xlabel('x')
        axes[0, 2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[0, 2])

        # 收敛历史
        axes[1, 0].semilogy(range(1, len(self.convergence_history) + 1),
                            self.convergence_history, 'b-o', markersize=4)
        axes[1, 0].set_title('Convergence History')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Residual Norm')
        axes[1, 0].grid(True)

        # 相位
        phase = np.angle(solution_2d)
        im4 = axes[1, 1].contourf(X, Y, phase, levels=20, cmap='hsv')
        axes[1, 1].set_title('Solution - Phase')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        plt.colorbar(im4, ax=axes[1, 1])

        # 3D表面图
        ax3d = fig.add_subplot(2, 3, 6, projection='3d')
        surf = ax3d.plot_surface(X, Y, np.abs(
            solution_2d), cmap='viridis', alpha=0.8)
        ax3d.set_title('Solution Magnitude - 3D View')
        ax3d.set_xlabel('x')
        ax3d.set_ylabel('y')
        ax3d.set_zlabel('|u|')

        plt.tight_layout()
        plt.show()

        # 保存图像
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        filename = f"AMG_Complex_Solution_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图像已保存: {filename}")


# 主程序
if __name__ == "__main__":
    print("代数多重网格复数求解器演示")
    print("=" * 50)

    # 问题参数
    nx, ny, nz = 16, 16, 12

    # 构建系数矩阵
    matrix_builder = ComplexMatrixBuilder(nx, ny, nz)
    A = matrix_builder.build_matrix()

    # 创建AMG求解器
    solver = AlgebraicMultigridComplex(
        max_levels=10,
        tolerance=1e-8)

    b = solver.create_rhs(nx=nx, ny=ny, nz=nz)
    x = solver.solve(A=A, b=b)
    # 验证
    solver.verify_solution(A=A, x=x, b=b)
    # 可视化
    solver.plot_results(solution=x, nx=nx, ny=ny, nz=nz)
    # 性能统计
    print(f"\n性能统计:")
    print(f"网格大小: {nx}x{ny}x{nz}")
    print(f"未知数个数: {nx*ny*nz}")
    print(f"收敛迭代次数: {len(solver.convergence_history)}")
    print(f"最终残差: {solver.convergence_history[-1]:.2e}")
    print(f"\n{'='*80}")
    print("所有测试完成!")
    print(f"{'='*80}")
