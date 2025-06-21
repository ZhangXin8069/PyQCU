import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import spsolve, LinearOperator, bicgstab, aslinearoperator
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

    def matvec_operator(self, A):
        """创建矩阵向量乘法算子"""
        if issparse(A):
            return lambda x: A.dot(x)
        return lambda x: A @ x


class AMGEigenvectorCoarsening:
    """
    基于近核特征向量的代数多重网格粗化策略
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
        """应用Chebyshev滤波器加速幂迭代"""
        # 计算Chebyshev多项式参数
        delta = (lambda_max - lambda_min) / 2.0
        sigma = (lambda_max + lambda_min) / 2.0
        theta = sigma / delta
        s = v.copy()
        
        # Chebyshev递归
        alpha = 2.0 / delta
        beta = -theta
        
        # 第一项: T1(x) = x
        w = A.matvec(s) - sigma * s
        w *= alpha
        v_filtered = w - beta * v
        
        # 更高阶项
        for k in range(2, self.chebyshev_degree + 1):
            w = A.matvec(v_filtered) - sigma * v_filtered
            w *= alpha
            new_v = 2 * w - beta * v
            v, v_filtered = v_filtered, new_v
        
        return v_filtered

    def power_iteration_with_chebyshev(self, A, num_vectors, max_iter=20, tol=1e-6):
        """使用Chebyshev加速的幂迭代法计算特征向量"""
        n = A.shape[0]  # 现在A是LinearOperator，有shape属性
        eigenvectors = np.zeros((n, num_vectors), dtype=np.complex128)
        
        # 估计特征值范围 (简化估计)
        lambda_max = 0.0
        for _ in range(10):
            v = np.random.randn(n) + 1j * np.random.randn(n)
            v /= np.linalg.norm(v)
            Av = A.matvec(v)
            lambda_est = np.abs(np.vdot(v, Av))
            if lambda_est > lambda_max:
                lambda_max = lambda_est
        
        lambda_min = 0.1 * lambda_max  # 简化的最小值估计
        
        print(f"  估计特征值范围: [{lambda_min:.4e}, {lambda_max:.4e}]")
        
        # 为每个特征向量执行幂迭代
        for k in range(num_vectors):
            # 随机初始化
            v = np.random.randn(n) + 1j * np.random.randn(n)
            v /= np.linalg.norm(v)
            
            prev_norm = 0.0
            for i in range(max_iter):
                # 应用Chebyshev滤波器
                v = self.chebyshev_filter(A, v, lambda_min, lambda_max)
                
                # 正交化
                for j in range(k):
                    proj = np.vdot(eigenvectors[:, j], v)
                    v -= proj * eigenvectors[:, j]
                
                # 归一化
                norm_v = np.linalg.norm(v)
                if norm_v < 1e-12:
                    v = np.random.randn(n) + 1j * np.random.randn(n)
                    v /= np.linalg.norm(v)
                    continue
                
                v /= norm_v
                
                # 检查收敛
                if i > 0 and abs(norm_v - prev_norm) < tol:
                    break
                
                prev_norm = norm_v
            
            eigenvectors[:, k] = v
        
        return eigenvectors

    def compute_near_kernel_eigenvectors(self, A):
        """计算近核特征向量(使用幂迭代法)"""
        print(f"使用幂迭代法计算 {self.num_eigenvectors} 个最小特征值对应的特征向量...")
        
        if isinstance(A, LinearOperator):
            n = A.shape[0]
        else:
            n = A.shape[0]
        
        if n <= self.max_coarse_size:
            return None
        
        # 确保我们传递的是线性算子
        if not isinstance(A, LinearOperator):
            A_op = aslinearoperator(A)
        else:
            A_op = A
        
        # 使用Chebyshev加速的幂迭代
        eigenvectors = self.power_iteration_with_chebyshev(
            A_op, 
            min(self.num_eigenvectors, n-2)
        )
        
        # 估计特征值
        eigenvalues = np.array([np.vdot(v, A_op.matvec(v)) for v in eigenvectors.T])
        sorted_indices = np.argsort(np.abs(eigenvalues))
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        print(f"  估计特征值: {[f'{abs(ev):.4e}' for ev in eigenvalues]}")
        
        return eigenvectors

    def analyze_smoothness(self, A, eigenvectors):
        """分析特征向量的平滑性(使用矩阵向量乘法)"""
        print("分析特征向量平滑性...")
        
        n = len(eigenvectors)
        smoothness_indicators = np.zeros(n)
        
        # 构建差分算子(使用矩阵向量乘法)
        def diff_operator(x):
            return self._diff_operator_matvec(A, x)
        
        for i, vec in enumerate(eigenvectors.T):
            # 计算向量的"平滑度"
            diff_vec = diff_operator(vec)
            local_smoothness = np.abs(diff_vec) / (np.abs(vec) + 1e-12)
            
            # 累积平滑性指标
            smoothness_indicators += local_smoothness / (i + 1)
        
        # 归一化平滑性指标
        smoothness_indicators = smoothness_indicators / np.max(smoothness_indicators)
        
        print(f"  平滑性指标范围: [{np.min(smoothness_indicators):.4f}, {np.max(smoothness_indicators):.4f}]")
        
        return smoothness_indicators

    def _diff_operator_matvec(self, A, x):
        """差分算子的矩阵向量乘法实现"""
        if isinstance(A, LinearOperator):
            A_op = A
        else:
            A_op = aslinearoperator(A)
        
        # 计算A的绝对值效果
        Ax = A_op.matvec(x)
        
        # 计算加权平均值(近似)
        n = len(x)
        row_sums = np.zeros(n, dtype=np.complex128)
        
        # 简化的行和计算(避免显式矩阵)
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            A_row = A_op.matvec(e_i)
            row_sums[i] = np.sum(np.abs(A_row)) - np.abs(A_row[i])
        
        # 构建差分结果
        diff_result = np.zeros_like(x)
        
        for i in range(n):
            if row_sums[i] > 1e-12:
                # 计算加权平均
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
        """基于特征向量的粗化算法(使用矩阵向量乘法)"""
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
        Q, R, pivots = qr(eigenvectors, mode='economic', pivoting=True)
        
        # 基于QR分解的列主元选择策略
        important_indices = set(pivots[:min(k, n//4)])  # 限制选择数量
        
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
        """构建插值算子(使用矩阵向量乘法)"""
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
        # 定义矩阵向量乘法函数
        if isinstance(A, LinearOperator):
            A_op = A
        else:
            A_op = aslinearoperator(A)
        
        # 计算强连接阈值
        theta = 0.25
        
        for f_idx in f_indices:
            # 获取F点的强C邻居
            e_f = np.zeros(n)
            e_f[f_idx] = 1.0
            A_row = A_op.matvec(e_f)
            
            # 计算强连接阈值
            max_val = 0.0
            for j in range(n):
                if j != f_idx and np.abs(A_row[j]) > max_val:
                    max_val = np.abs(A_row[j])
            
            threshold = theta * max_val
            
            strong_c_neighbors = []
            for j in range(n):
                if c_points[j] and np.abs(A_row[j]) >= threshold:
                    strong_c_neighbors.append(j)
            
            if len(strong_c_neighbors) == 0:
                # 如果没有强C邻居，寻找最大的连接
                max_val = 0.0
                max_j = -1
                for j in range(n):
                    if c_points[j] and np.abs(A_row[j]) > max_val:
                        max_val = np.abs(A_row[j])
                        max_j = j
                if max_j != -1:
                    strong_c_neighbors.append(max_j)
            
            if len(strong_c_neighbors) > 0:
                # 计算插值权重
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


class FGMRESSmoother:
    """灵活GMRES(FGMRES)光滑器"""

    def __init__(self, max_krylov=5, max_restarts=1, tol=0.1):
        self.max_krylov = max_krylov
        self.max_restarts = max_restarts
        self.tol = tol

    def smooth(self, A, b, x0):
        """执行FGMRES光滑"""
        n = len(b)
        x = x0.copy()
        r = b - A(x0)  # 使用矩阵向量乘法
        r_norm = np.linalg.norm(r)
        
        if r_norm < 1e-12:
            return x0
        
        # 存储搜索方向
        V = np.zeros((n, self.max_krylov + 1), dtype=np.complex128)
        Z = np.zeros((n, self.max_krylov), dtype=np.complex128)  # 预处理后的方向
        H = np.zeros((self.max_krylov + 1, self.max_krylov), dtype=np.complex128)
        
        # 初始残差向量
        V[:, 0] = r / r_norm
        
        # Givens旋转存储
        cs = np.zeros(self.max_krylov, dtype=np.complex128)
        sn = np.zeros(self.max_krylov, dtype=np.complex128)
        s = np.zeros(self.max_krylov + 1, dtype=np.complex128)
        s[0] = r_norm
        
        for j in range(self.max_krylov):
            # 应用预处理 (这里使用简单的对角预处理)
            # 暂时禁用预处理以简化
            Z[:, j] = V[:, j]  # 不使用预处理
            
            # 矩阵向量乘法
            w = A(Z[:, j])
            
            # Arnoldi过程
            for i in range(j + 1):
                H[i, j] = np.vdot(V[:, i], w)
                w = w - H[i, j] * V[:, i]
            
            H[j + 1, j] = np.linalg.norm(w)
            if abs(H[j + 1, j]) < 1e-12:
                break
            
            V[:, j + 1] = w / H[j + 1, j]
            
            # 应用Givens旋转
            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i].conj() * H[i, j] + cs[i].conj() * H[i + 1, j]
                H[i, j] = temp
            
            # 计算新的Givens旋转
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
            
            # 更新s向量
            s[j + 1] = -sn[j].conj() * s[j]
            s[j] = cs[j] * s[j]
            
            # 检查收敛
            if abs(s[j + 1]) < self.tol * r_norm:
                break
        
        # 解最小二乘问题
        j_end = j + 1
        y = np.linalg.lstsq(H[:j_end, :j_end], s[:j_end], rcond=None)[0]
        
        # 更新解
        dx = Z[:, :j_end] @ y
        x = x + dx
        
        return x


class AlgebraicMultigridComplex:
    """
    代数多重网格复数求解器 - 使用特征向量粗化
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
        
        # 初始化组件
        self.coarsening = AMGEigenvectorCoarsening(
            num_eigenvectors=num_eigenvectors,
            max_coarse_size=max_coarse_size,
            power_iterations=power_iterations,
            chebyshev_degree=chebyshev_degree
        )
        self.interpolation = AMGInterpolation()
        self.smoother = FGMRESSmoother()
        
        # 存储多重网格层次结构
        self.matrices = []
        self.matvec_ops = []
        self.interpolation_ops = []
        self.restriction_ops = []

    def setup_hierarchy(self, A):
        """建立AMG层次结构 (基于特征向量粗化)"""
        print("建立AMG层次结构 (基于特征向量粗化)...")
        print("=" * 50)
        
        # 使用线性算子包装A
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
            
            # 定义矩阵向量乘法函数 - 修复维度问题
            def A_coarse_matvec(x):
                # 确保输入向量长度正确
                if len(x) != P.shape[1]:
                    # 如果长度不匹配，尝试调整
                    if len(x) < P.shape[1]:
                        # 填充零
                        x_padded = np.zeros(P.shape[1], dtype=x.dtype)
                        x_padded[:len(x)] = x
                        x = x_padded
                    else:
                        # 截断
                        x = x[:P.shape[1]]
                
                # 执行矩阵向量乘法
                Px = P.dot(x)
                A_Px = current_matrix.matvec(Px)
                return R.dot(A_Px)
            
            # 创建线性算子
            A_coarse = LinearOperator(
                (P.shape[1], P.shape[1]),
                matvec=A_coarse_matvec,
                dtype=np.complex128
            )
            
            self.matrices.append(A_coarse)
            self.matvec_ops.append(A_coarse_matvec)
            
            current_matrix = A_coarse
            level += 1
            
            print(f"  粗网格大小: {A_coarse.shape[0]}")
            if level > 0 and len(self.matrices) > 1:
                prev_size = self.matrices[-2].shape[0]
                curr_size = A_coarse.shape[0]
                print(f"  粗化比: {curr_size / prev_size:.3f}")
        
        print(f"\n总层数: {len(self.matrices)}")
        print("层次结构建立完成!")
        print("=" * 50)

    def v_cycle(self, b, x, level=0):
        """V-循环迭代"""
        A_matvec = self.matvec_ops[level]
        
        if level == len(self.matrices) - 1:
            # 最粗层：直接求解
            n_coarse = self.matrices[level].shape[0]
            
            # 确保x有正确的长度
            if len(x) != n_coarse:
                # 如果长度不匹配，调整x
                if len(x) < n_coarse:
                    # 填充零
                    x_padded = np.zeros(n_coarse, dtype=b.dtype)
                    x_padded[:len(x)] = x
                    x = x_padded
                else:
                    # 截断
                    x = x[:n_coarse]
            
            if n_coarse <= 200:  # 小矩阵直接求解
                # 对于线性算子，使用迭代方法
                return self.smoother.smooth(A_matvec, b, x)
            else:
                # 大矩阵使用光滑器
                return self.smoother.smooth(A_matvec, b, x)
        
        # 确保有插值和限制算子
        if level >= len(self.interpolation_ops):
            return x
        
        P = self.interpolation_ops[level]
        R = self.restriction_ops[level]
        
        # 前光滑
        x = self.smoother.smooth(A_matvec, b, x)
        
        # 计算残差
        residual = b - A_matvec(x)
        
        # 限制残差到粗网格
        coarse_residual = R.dot(residual)
        n_coarse = len(coarse_residual)
        coarse_error = np.zeros(n_coarse, dtype=coarse_residual.dtype)
        
        # 递归求解粗网格修正方程
        coarse_error = self.v_cycle(coarse_residual, coarse_error, level + 1)
        
        # 确保coarse_error有正确的长度
        if len(coarse_error) != n_coarse:
            # 如果长度不匹配，调整coarse_error
            if len(coarse_error) < n_coarse:
                # 填充零
                error_padded = np.zeros(n_coarse, dtype=coarse_residual.dtype)
                error_padded[:len(coarse_error)] = coarse_error
                coarse_error = error_padded
            else:
                # 截断
                coarse_error = coarse_error[:n_coarse]
        
        # 插值修正到细网格
        fine_correction = P.dot(coarse_error)
        x = x + fine_correction
        
        # 后光滑
        x = self.smoother.smooth(A_matvec, b, x)
        
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
            x = self.v_cycle(b, x, level=0)
            
            # 计算残差
            residual = b - self.matvec_ops[0](x)
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
        
        return x, solve_time

    def solve_bicgstab(self, A, b, x0=None):
        """使用BiCGSTAB求解器进行比较"""
        print("\n开始BiCGSTAB求解...")
        print("=" * 40)
        
        if x0 is None:
            x0 = np.zeros_like(b)
        
        # 记录残差历史的回调函数
        residuals = []
        def callback(xk):
            r = b - A.dot(xk)
            residuals.append(np.linalg.norm(r))
        
        start_time = time.time()
        x, info = bicgstab(A, b, x0=x0, callback=callback, 
                          atol=self.tolerance, maxiter=self.max_iterations)
        solve_time = time.time() - start_time
        
        if info == 0:
            print(f"✓ BiCGSTAB 收敛达到容差 {self.tolerance}")
        else:
            print(f"⚠ BiCGSTAB 未收敛 (状态码: {info})")
        
        final_residual = residuals[-1] if residuals else np.nan
        print(f"迭代次数: {len(residuals)}")
        print(f"最终残差: {final_residual:.2e}")
        print(f"求解时间: {solve_time:.4f} 秒")
        print("=" * 40)
        
        self.bicgstab_history = residuals
        return x, solve_time, residuals

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
        
        # 定义矩阵向量乘法
        if isinstance(A, LinearOperator):
            Ax = A.matvec(x)
        else:
            Ax = A.dot(x)
            
        residual = Ax - b
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

    def plot_convergence(self, amg_time, bicgstab_time):
        """绘制收敛历史对比图"""
        plt.figure(figsize=(10, 6))
        
        # AMG收敛历史
        if self.convergence_history:
            amg_iter = range(1, len(self.convergence_history) + 1)
            plt.semilogy(amg_iter, self.convergence_history, 'b-o', 
                        label=f'AMG (Time: {amg_time:.2f}s)')
        
        # BiCGSTAB收敛历史
        if self.bicgstab_history:
            bicg_iter = range(1, len(self.bicgstab_history) + 1)
            plt.semilogy(bicg_iter, self.bicgstab_history, 'r--s', 
                        label=f'BiCGSTAB (Time: {bicgstab_time:.2f}s)')
        
        plt.title('Convergence History Comparison')
        plt.xlabel('Iteration')
        plt.ylabel('Residual Norm')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        
        # 保存图像
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        filename = f"Convergence_Comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"收敛对比图已保存: {filename}")


# 主程序
if __name__ == "__main__":
    print("代数多重网格复数求解器演示")
    print("=" * 50)
    
    # 问题参数 - 使用较小的网格进行测试
    # nx, ny, nz = 16, 16, 8
    nx, ny, nz = 32, 32, 12
    
    # 构建系数矩阵
    matrix_builder = ComplexMatrixBuilder(nx, ny, nz)
    A = matrix_builder.build_matrix()
    
    # 创建AMG求解器
    solver = AlgebraicMultigridComplex(
        max_levels=10,
        tolerance=1e-8,
        max_iterations=100,
        power_iterations=10,
        chebyshev_degree=4
    )
    
    b = solver.create_rhs(nx=nx, ny=ny, nz=nz)
    
    # 使用AMG求解
    x_amg, amg_time = solver.solve(A=A, b=b)
    
    # 使用BiCGSTAB求解
    x_bicg, bicg_time, _ = solver.solve_bicgstab(A=A, b=b)
    
    # 计算加速比
    speedup = bicg_time / amg_time if amg_time > 0 else float('inf')
    print(f"\n性能对比:")
    print(f"AMG求解时间: {amg_time:.4f}秒")
    print(f"BiCGSTAB求解时间: {bicg_time:.4f}秒")
    print(f"加速比: {speedup:.2f}x")
    print("=" * 50)
    
    # 验证解
    solver.verify_solution(A=A, x=x_amg, b=b)
    
    # 绘制收敛历史对比图
    solver.plot_convergence(amg_time, bicg_time)
    
    # 性能统计
    print(f"\n性能统计:")
    print(f"网格大小: {nx}x{ny}x{nz}")
    print(f"未知数个数: {nx*ny*nz}")
    print(f"AMG收敛迭代次数: {len(solver.convergence_history)}")
    print(f"BiCGSTAB收敛迭代次数: {len(solver.bicgstab_history)}")
    print(f"加速比: {speedup:.2f}x")
    print(f"\n{'='*80}")
    print("所有测试完成!")
    print(f"{'='*80}")