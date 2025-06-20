import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve, eigsh
from scipy.linalg import qr
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


class AMGEigenvectorCoarsening:
    """
    基于近核特征向量的代数多重网格粗化策略
    """

    def __init__(self, num_eigenvectors=3, max_coarse_size=100,
                 coarsening_ratio=0.3, smoothness_threshold=0.5):
        self.num_eigenvectors = num_eigenvectors
        self.max_coarse_size = max_coarse_size
        self.coarsening_ratio = coarsening_ratio
        self.smoothness_threshold = smoothness_threshold

    def compute_near_kernel_eigenvectors(self, A):
        """计算近核特征向量"""
        print(f"计算 {self.num_eigenvectors} 个最小特征值对应的特征向量...")

        n = A.shape[0]
        if n <= self.max_coarse_size:
            return None

        try:
            # 对于复数矩阵，寻找最小模长特征值
            sigma = 0.0  # 寻找接近0的特征值

            # 使用shift-invert模式寻找接近sigma的特征值
            eigenvalues, eigenvectors = eigsh(
                A, k=min(self.num_eigenvectors, n-2),
                sigma=sigma, which='LM',
                maxiter=1000, tol=1e-8
            )

            # 按特征值模长排序
            sorted_indices = np.argsort(np.abs(eigenvalues))
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            print(f"  特征值: {[f'{abs(ev):.4e}' for ev in eigenvalues]}")

            return eigenvectors

        except Exception as e:
            print(f"  特征值计算失败: {e}")
            print("  回退到随机向量方案")
            # 回退方案：使用随机向量
            return np.random.random((n, self.num_eigenvectors)) + \
                1j * np.random.random((n, self.num_eigenvectors))

    def analyze_smoothness(self, A, eigenvectors):
        """分析特征向量的平滑性"""
        print("分析特征向量平滑性...")

        n = A.shape[0]
        smoothness_indicators = np.zeros(n)

        # 构建差分算子来度量平滑性
        D = self._build_difference_operator(A)

        for i, vec in enumerate(eigenvectors.T):
            # 计算向量的"平滑度"：差分算子作用后的范数相对较小
            diff_vec = D @ vec
            local_smoothness = np.abs(diff_vec) / (np.abs(vec) + 1e-12)

            # 累积平滑性指标
            smoothness_indicators += local_smoothness / (i + 1)

        # 归一化平滑性指标
        smoothness_indicators = smoothness_indicators / \
            np.max(smoothness_indicators)

        print(
            f"  平滑性指标范围: [{np.min(smoothness_indicators):.4f}, {np.max(smoothness_indicators):.4f}]")

        return smoothness_indicators

    def _build_difference_operator(self, A):
        """构建差分算子用于平滑性分析"""
        n = A.shape[0]

        # 简化的差分算子：使用矩阵A的结构
        A_abs = A.copy()
        A_abs.data = np.abs(A_abs.data)

        # 构建加权拉普拉斯算子
        D = A_abs.copy()
        D.setdiag(0)  # 移除对角元素

        # 对每行进行归一化
        row_sums = np.array(D.sum(axis=1)).flatten()
        for i in range(n):
            if row_sums[i] > 1e-12:
                D.data[D.indptr[i]:D.indptr[i+1]] /= row_sums[i]

        # 添加对角元素使其成为差分算子
        D.setdiag(-1.0)

        return D

    def eigenvector_based_coarsening(self, A):
        """基于特征向量的粗化算法"""
        print("执行基于特征向量的粗化...")

        n = A.shape[0]
        if n <= self.max_coarse_size:
            return np.ones(n, dtype=bool)

        # 计算近核特征向量
        eigenvectors = self.compute_near_kernel_eigenvectors(A)
        if eigenvectors is None:
            return np.ones(n, dtype=bool)

        # 分析平滑性
        smoothness = self.analyze_smoothness(A, eigenvectors)

        # 使用QR分解选择代表点
        c_points = self._qr_based_selection(eigenvectors, smoothness)

        # 确保粗化比例合理
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

        print(f"  粗网格点数: {np.sum(c_points)}")
        print(f"  细网格点数: {np.sum(~c_points)}")
        print(f"  粗化比: {np.sum(c_points) / n:.3f}")

        return c_points

    def _qr_based_selection(self, eigenvectors, smoothness):
        """基于QR分解的点选择策略"""
        n, k = eigenvectors.shape

        # 对特征向量进行QR分解
        Q, R, _ = qr(eigenvectors, mode='economic', pivoting=True)

        # 基于QR分解的列主元选择策略
        # 选择在特征向量空间中线性无关程度最高的点
        important_indices = set()

        # 从QR分解的主元信息选择重要点
        for i in range(min(k, n//4)):  # 限制选择数量
            col_norms = np.abs(R[i, :])
            max_idx = np.argmax(col_norms)
            important_indices.add(max_idx)

        # 基于平滑性添加额外的粗网格点
        smooth_points = np.where(smoothness < self.smoothness_threshold)[0]

        # 均匀分布选择平滑点
        if len(smooth_points) > 0:
            step = max(1, len(smooth_points) // (n // 20))  # 控制密度
            selected_smooth = smooth_points[::step]
            important_indices.update(selected_smooth)

        # 创建粗网格标记
        c_points = np.zeros(n, dtype=bool)
        for idx in important_indices:
            if idx < n:
                c_points[idx] = True

        return c_points

    def _reduce_coarse_points(self, c_points, smoothness, target_size):
        """减少粗网格点数量"""
        current_c_indices = np.where(c_points)[0]
        current_size = len(current_c_indices)

        if current_size <= target_size:
            return c_points

        # 根据平滑性排序，保留最重要的点
        smoothness_c = smoothness[current_c_indices]
        sorted_indices = np.argsort(smoothness_c)

        # 保留前target_size个最重要的点
        keep_indices = current_c_indices[sorted_indices[:target_size]]

        new_c_points = np.zeros_like(c_points)
        new_c_points[keep_indices] = True

        return new_c_points

    def _increase_coarse_points(self, c_points, smoothness, target_size):
        """增加粗网格点数量"""
        current_size = np.sum(c_points)
        needed = target_size - current_size

        if needed <= 0:
            return c_points

        # 从非粗网格点中选择最平滑的点
        f_indices = np.where(~c_points)[0]
        if len(f_indices) == 0:
            return c_points

        smoothness_f = smoothness[f_indices]
        sorted_indices = np.argsort(smoothness_f)

        # 选择最平滑的点添加到粗网格
        add_count = min(needed, len(f_indices))
        add_indices = f_indices[sorted_indices[:add_count]]

        new_c_points = c_points.copy()
        new_c_points[add_indices] = True

        return new_c_points


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
    代数多重网格复数求解器 - 使用特征向量粗化
    """

    def __init__(self, max_levels=10, tolerance=1e-8, max_iterations=100,
                 num_eigenvectors=3, max_coarse_size=50):
        self.max_levels = max_levels
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.num_eigenvectors = num_eigenvectors
        self.max_coarse_size = max_coarse_size
        self.convergence_history = []

        # 初始化组件
        self.coarsening = AMGEigenvectorCoarsening(
            num_eigenvectors=num_eigenvectors,
            max_coarse_size=max_coarse_size
        )
        self.interpolation = AMGInterpolation()
        self.smoother = GMRESSmoother()

        # 存储多重网格层次结构
        self.matrices = []
        self.interpolation_ops = []
        self.restriction_ops = []

    def setup_hierarchy(self, A):
        """建立AMG层次结构"""
        print("建立AMG层次结构 (基于特征向量粗化)...")
        print("=" * 50)

        self.matrices = [A]
        self.interpolation_ops = []
        self.restriction_ops = []

        current_matrix = A
        level = 0

        while (current_matrix.shape[0] > self.max_coarse_size and
               level < self.max_levels - 1):

            print(f"Level {level}: 矩阵大小 {current_matrix.shape[0]}")

            # 特征向量粗化
            c_points = self.coarsening.eigenvector_based_coarsening(
                current_matrix)

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
        print("开始AMG求解 (特征向量粗化)...")
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
    nx, ny, nz = 32, 32, 12

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
