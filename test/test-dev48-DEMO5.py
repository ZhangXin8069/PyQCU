import time
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting

np.Inf = np.inf

class MatrixVectorOperator:
    """
    封装矩阵向量乘法操作，避免显式存储矩阵
    """
    def __init__(self, nx, ny, nz, alpha=1.0, beta=1.0j):
        self.nx = nx
        self.ny = ny
        self.nz = nz  # 新增维度
        self.alpha = alpha
        self.beta = beta
        self.n = nx * ny * nz # 更新总尺寸
        self.hx = 1.0 / (nx + 1)
        self.hy = 1.0 / (ny + 1)
        self.hz = 1.0 / (nz + 1) # 新增z方向步长
        
        # 预计算对角线元素
        # 考虑 z 方向的项：-alpha/hz^2 * (neighbor_z)
        self.main_diag = (2 * alpha * (1/self.hx**2 + 1/self.hy**2 + 1/self.hz**2) + beta) * np.ones(self.n, dtype=np.complex128)
        
    def matvec(self, v):
        """
        执行矩阵向量乘法 A*v
        """
        result = np.zeros_like(v)
        nx, ny, nz = self.nx, self.ny, self.nz
        
        # 处理主对角线
        result[:] = self.main_diag * v
        
        # 将一维向量v reshape为三维，方便索引
        v_3d = v.reshape((nz, ny, nx))
        result_3d = result.reshape((nz, ny, nx))

        # 遍历三维网格
        for k in range(nz): # Z 维度
            for i in range(ny): # Y 维度
                for j in range(nx): # X 维度
                    idx = k * ny * nx + i * nx + j
                    
                    # 处理x方向相邻元素 (-alpha/hx²)
                    if j > 0:
                        result_3d[k, i, j] -= (self.alpha / self.hx**2) * v_3d[k, i, j-1]
                    if j < nx - 1:
                        result_3d[k, i, j] -= (self.alpha / self.hx**2) * v_3d[k, i, j+1]
                    
                    # 处理y方向相邻元素 (-alpha/hy²)
                    if i > 0:
                        result_3d[k, i, j] -= (self.alpha / self.hy**2) * v_3d[k, i-1, j]
                    if i < ny - 1:
                        result_3d[k, i, j] -= (self.alpha / self.hy**2) * v_3d[k, i+1, j]
                        
                    # 处理z方向相邻元素 (-alpha/hz²)
                    if k > 0:
                        result_3d[k, i, j] -= (self.alpha / self.hz**2) * v_3d[k-1, i, j]
                    if k < nz - 1:
                        result_3d[k, i, j] -= (self.alpha / self.hz**2) * v_3d[k+1, i, j]
        
        return result_3d.flatten()
        
    def diagonal(self):
        """返回对角线元素，用于Jacobi迭代"""
        return self.main_diag.copy()

class GMRESSmoother:
    """GMRES 光滑器实现"""
    def __init__(self, max_krylov=5, max_restarts=1, tol=0.1):
        self.max_krylov = max_krylov  # Krylov子空间大小
        self.max_restarts = max_restarts  # 重启次数
        self.tol = tol  # 相对容差
        
    def smooth(self, op, b, x0):
        """
        执行GMRES光滑
        :param op: 矩阵向量算子
        :param b: 右端项
        :param x0: 初始解
        :return: 光滑后的解
        """
        x = x0.copy()
        r = b - op.matvec(x0)
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-12:
            return x0
            
        # 重启循环
        for _ in range(self.max_restarts):
            # Arnoldi过程构建Krylov子空间
            Q, H = self._arnoldi(op, r, r_norm)
            
            # 解最小二乘问题
            e1 = np.zeros(H.shape[1] + 1, dtype=b.dtype) # Use b's dtype
            e1[0] = r_norm
            y = self._solve_least_squares(H, e1)
            
            # 更新解
            dx = Q[:, :-1] @ y
            x = x + dx
            
            # 检查收敛
            new_r = b - op.matvec(x)
            new_r_norm = np.linalg.norm(new_r)
            if new_r_norm < self.tol * r_norm:
                break
                
            r = new_r
            r_norm = new_r_norm
            
        return x
        
    def _arnoldi(self, op, r0, r_norm):
        """Arnoldi过程构建Krylov子空间"""
        m = self.max_krylov
        n = len(r0)
        Q = np.zeros((n, m+1), dtype=r0.dtype) # Use r0's dtype
        H = np.zeros((m+1, m), dtype=r0.dtype) # Use r0's dtype
        
        # 第一正交基向量
        Q[:, 0] = r0 / r_norm
        
        for j in range(m):
            # 应用算子
            w = op.matvec(Q[:, j])
            
            # 正交化
            for i in range(j+1):
                H[i, j] = np.vdot(Q[:, i], w) # Use vdot for complex numbers
                w = w - H[i, j] * Q[:, i]
            
            # 归一化
            h_norm = np.linalg.norm(w)
            H[j+1, j] = h_norm
            
            if h_norm < 1e-12:  # 提前终止
                return Q[:, :j+1], H[:j+1, :j]
                
            if j < m:
                Q[:, j+1] = w / h_norm
            
        return Q, H
        
    def _solve_least_squares(self, H, e1):
        """使用QR分解求解最小二乘问题"""
        # 获取Hessenberg矩阵维度
        m = H.shape[1]
        h_height = H.shape[0]
        
        # 创建增广矩阵
        R = np.zeros((h_height, m+1), dtype=H.dtype) # Use H's dtype
        R[:, :m] = H
        R[:, m] = e1[:h_height]
        
        # 使用QR分解求解最小二乘问题
        Q, R_qr = np.linalg.qr(R, mode='complete')
        
        # 提取解
        # Correctly apply conjugate transpose for Q in QR decomposition
        y = np.linalg.solve(R_qr[:m, :m], Q[:m, :m].conj().T @ e1[:m])
        return y

class AdaptiveMultigridComplex:
    """
    自适应多重网格方法求解复数系统
    适用于求解复数系数的椭圆型偏微分方程
    """

    def __init__(self, nx, ny, nz=1, max_levels=5, tolerance=1e-8, max_iterations=100, dtype=np.complex128):
        """
        初始化自适应多重网格求解器

        参数:
        nx: x方向网格大小
        ny: y方向网格大小
        nz: z方向网格大小 (新增，不参与粗化)
        max_levels: 最大网格层数
        tolerance: 收敛容差
        max_iterations: 最大迭代次数
        dtype: 求解数据类型，默认为np.complex128 (新增)
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz # 存储nz
        self.max_levels = max_levels
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.dtype = dtype # 存储dtype
        self.convergence_history = []
        self.level_info = []
        
        # 创建GMRES光滑器实例
        self.gmres_smoother = GMRESSmoother(max_krylov=5, max_restarts=1, tol=0.1)

    def create_operator(self, nx, ny, nz, alpha=1.0, beta=1.0j):
        """
        创建矩阵向量算子
        """
        print(f"创建 {nx}x{ny}x{nz} 复数算子...")
        print(f"  系数: α = {alpha}, β = {beta}")
        return MatrixVectorOperator(nx, ny, nz, alpha, beta)

    def create_rhs(self, nx, ny, nz, func_type='sine'):
        """创建右端项"""
        hx = 1.0 / (nx + 1)
        hy = 1.0 / (ny + 1)
        hz = 1.0 / (nz + 1) # 新增z方向步长
        x = np.linspace(hx, 1-hx, nx)
        y = np.linspace(hy, 1-hy, ny)
        z = np.linspace(hz, 1-hz, nz) # 新增z坐标

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij') # Create 3D meshgrid

        if func_type == 'sine':
            # 复数正弦函数
            f = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y) * np.sin(2*np.pi*Z) * (1 + 1j)
        elif func_type == 'exponential':
            # 复数指数函数
            f = np.exp(X + 1j*Y + 1j*Z)
        else:
            # 默认函数
            f = np.ones((nx, ny, nz), dtype=self.dtype) # Adjust shape for 3D

        return f.flatten()

    def restrict(self, u_fine, nx_fine, ny_fine, nz_fine):
        """限制算子：细网格到粗网格 (z维度保持不变)"""
        if nx_fine < 2 or ny_fine < 2:
            return u_fine # 无法再粗化，直接返回

        nx_coarse = max(2, nx_fine // 2)
        ny_coarse = max(2, ny_fine // 2)
        nz_coarse = nz_fine # Z维度保持不变

        # Reshape fine grid solution to 3D for easier processing
        u_fine_3d = u_fine.reshape((nz_fine, ny_fine, nx_fine))
        u_coarse_3d = np.zeros((nz_coarse, ny_coarse, nx_coarse), dtype=self.dtype)

        # Apply 2D restriction slice by slice along the z-dimension
        for k in range(nz_fine):
            u_fine_slice = u_fine_3d[k, :, :]
            u_coarse_slice = np.zeros((ny_coarse, nx_coarse), dtype=self.dtype)

            # 全权重限制 (9点模板)
            for i in range(ny_coarse):
                for j in range(nx_coarse):
                    ii, jj = 2*i, 2*j # Adjust for cell-centered restriction
                    weight_sum = 0
                    value_sum = 0

                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = ii + di, jj + dj
                            if 0 <= ni < ny_fine and 0 <= nj < nx_fine:
                                # Standard full-weighting stencil
                                if di == 0 and dj == 0:
                                    weight = 1/4 # Center point
                                elif di == 0 or dj == 0:
                                    weight = 1/8 # Face neighbors
                                else:
                                    weight = 1/16 # Corner neighbors
                                weight_sum += weight
                                value_sum += weight * u_fine_slice[ni, nj]

                    u_coarse_slice[i, j] = value_sum / weight_sum if weight_sum > 0 else 0
            u_coarse_3d[k, :, :] = u_coarse_slice

        return u_coarse_3d.flatten()

    def prolongate(self, u_coarse, nx_fine, ny_fine, nz_fine):
        """延拓算子：粗网格到细网格 (z维度保持不变)"""
        nx_coarse = nx_fine // 2
        ny_coarse = ny_fine // 2
        nz_coarse = nz_fine # Z维度保持不变

        u_coarse_3d = u_coarse.reshape((nz_coarse, ny_coarse, nx_coarse))
        u_fine_3d = np.zeros((nz_fine, ny_fine, nx_fine), dtype=self.dtype)

        # Apply 2D prolongation slice by slice along the z-dimension
        for k in range(nz_fine):
            u_coarse_slice = u_coarse_3d[k, :, :]
            u_fine_slice = np.zeros((ny_fine, nx_fine), dtype=self.dtype)

            # 双线性插值
            for i in range(ny_fine):
                for j in range(nx_fine):
                    # Calculate position in coarse grid (relative to coarse grid indices)
                    i_c = i / 2.0
                    j_c = j / 2.0

                    # Find the four surrounding coarse grid points
                    i0, j0 = int(i_c), int(j_c)
                    i1 = min(i0 + 1, ny_coarse - 1)
                    j1 = min(j0 + 1, nx_coarse - 1)

                    # Interpolation weights
                    wx = i_c - i0
                    wy = j_c - j0

                    # Bilinear interpolation
                    u_fine_slice[i, j] = (1 - wx) * (1 - wy) * u_coarse_slice[i0, j0] + \
                                         (1 - wx) * wy * u_coarse_slice[i0, j1] + \
                                         wx * (1 - wy) * u_coarse_slice[i1, j0] + \
                                         wx * wy * u_coarse_slice[i1, j1]
            u_fine_3d[k, :, :] = u_fine_slice

        return u_fine_3d.flatten()

    def smooth(self, op, b, u, num_iterations=1, method='gmres'):
        """
        光滑算子 - 使用GMRES作为光滑器
        op: 矩阵向量算子对象
        b: 右端项
        u: 初始解
        num_iterations: 重启次数 (对于GMRES，这里实际由GMRESSmoother的max_restarts控制)
        """
        # 使用GMRES光滑器
        return self.gmres_smoother.smooth(op, b, u)

    def compute_residual(self, op, b, u):
        """计算残差 - 使用矩阵向量算子"""
        return b - op.matvec(u)

    def bistabcg_solver(self, op, b, x0=None, tol=1e-10, maxiter=1000):
        """
        双共轭梯度稳定法 (BiCGSTAB) 求解器
        用于求解线性系统: A*x = b
        
        参数:
        op: 矩阵向量算子 (实现 matvec 方法)
        b: 右端项
        x0: 初始解 (可选)
        tol: 容差
        maxiter: 最大迭代次数
        
        返回:
        x: 解向量
        info: 收敛信息 (0 表示成功)
        """
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
            rho1 = np.vdot(r0, r)  # 复数点积
            if abs(rho1) < np.finfo(rho1.dtype).eps: # Check for near-zero rho1
                return x, 2 # Breakdown

            beta = (rho1 / rho) * (alpha / omega)
            p = r + beta * (p - omega * v)
            v = op.matvec(p)
            
            denom_alpha = np.vdot(r0, v)
            if abs(denom_alpha) < np.finfo(denom_alpha.dtype).eps: # Check for near-zero denominator
                return x, 3 # Breakdown

            alpha = rho1 / denom_alpha
            s = r - alpha * v
            
            # Check for convergence after s calculation (typical BiCGSTAB)
            residual_norm = np.linalg.norm(s)
            if residual_norm < tol:
                return x + alpha * p, 0 # Converged
                
            t = op.matvec(s)
            
            denom_omega = np.vdot(t, t)
            if abs(denom_omega) < np.finfo(denom_omega.dtype).eps: # Check for near-zero denominator
                return x, 4 # Breakdown

            omega = np.vdot(t, s) / denom_omega
            x = x + alpha * p + omega * s
            r = s - omega * t
            
            # Check for convergence after updating x
            residual_norm = np.linalg.norm(r)
            if residual_norm < tol:
                return x, 0
                
            rho = rho1
            
        # 未收敛
        return x, 1

    def v_cycle(self, op_hierarchy, b_hierarchy, u_hierarchy, grid_params, level=0):
        """V-循环 - 使用矩阵向量算子"""
        current_level_idx = len(op_hierarchy) - 1 - level
        nx, ny, nz = grid_params[current_level_idx]
        print(f"V-循环 level {level}, 当前层索引: {current_level_idx}, 网格大小: {nx}x{ny}x{nz}")
        
        op = op_hierarchy[current_level_idx]
        b = b_hierarchy[current_level_idx]
        u = u_hierarchy[current_level_idx]

        # 如果是最粗网格，直接求解
        if current_level_idx == 0 or level >= self.max_levels - 1:
            print(f"    最粗网格直接求解...")
            # 计算前残差
            residual = self.compute_residual(op, b, u)
            residual_norm = np.linalg.norm(residual)
            print(f"    前残差范数: {residual_norm:.4e}")
            
            # 使用自定义的BiCGSTAB求解器
            u_coarse, info = self.bistabcg_solver(op, b, u, tol=1e-8, maxiter=1000) # Increased tolerance for inner solver
            if info != 0:
                print(f"    警告: 最粗网格求解未收敛! Info: {info}")
            u_hierarchy[current_level_idx] = u_coarse
            residual = self.compute_residual(op, b, u_hierarchy[current_level_idx])
            residual_norm = np.linalg.norm(residual)
            print(f"    残差范数: {residual_norm:.4e}")
            return u_hierarchy[current_level_idx]
            
        # 计算前光滑残差
        residual_before_smooth = self.compute_residual(op, b, u)
        residual_norm_before_smooth = np.linalg.norm(residual_before_smooth)
        print(f"    前光滑前残差范数: {residual_norm_before_smooth:.4e}")
        
        # 前光滑
        print(f"    前光滑...")
        u = self.smooth(op, b, u)
        u_hierarchy[current_level_idx] = u

        # 计算残差
        residual = self.compute_residual(op, b, u_hierarchy[current_level_idx])
        residual_norm = np.linalg.norm(residual)
        print(f"    前光滑后残差范数: {residual_norm:.4e}")

        # 限制残差到粗网格
        if current_level_idx > 0:
            r_coarse = self.restrict(residual, nx, ny, nz)
            b_hierarchy[current_level_idx - 1] = r_coarse
            u_hierarchy[current_level_idx - 1] = np.zeros_like(r_coarse, dtype=self.dtype)

            # 递归调用粗网格
            e_coarse = self.v_cycle(
                op_hierarchy, b_hierarchy, u_hierarchy, grid_params, level + 1)

            # 延拓误差修正
            nx_fine, ny_fine, nz_fine = grid_params[current_level_idx]
            e_fine = self.prolongate(e_coarse, nx_fine, ny_fine, nz_fine)
            u = u + e_fine
            u_hierarchy[current_level_idx] = u

        # 计算后光滑前残差
        residual_before_post_smooth = self.compute_residual(op, b, u)
        residual_norm_before_post_smooth = np.linalg.norm(residual_before_post_smooth)
        print(f"    后光滑前残差范数: {residual_norm_before_post_smooth:.4e}")
            
        # 后光滑
        print(f"    后光滑...")
        u = self.smooth(op, b, u)
        u_hierarchy[current_level_idx] = u
        residual = self.compute_residual(op, b, u_hierarchy[current_level_idx])
        residual_norm = np.linalg.norm(residual)
        print(f"    后光滑后残差范数: {residual_norm:.4e}")
        return u

    def adaptive_criterion(self, residual_norms):
        """自适应准则：决定是否需要调整网格层数"""
        if len(residual_norms) < 3:
            return False

        # 计算收敛率
        conv_rate = residual_norms[-1] / \
            residual_norms[-2] if residual_norms[-2] != 0 else 1

        # 如果收敛太慢，建议增加网格层数
        return conv_rate > 0.8

    def solve(self, alpha=1.0, beta=1.0j, func_type='sine'):
        """主求解函数"""
        print("="*60)
        print("开始自适应多重网格复数求解")
        print("="*60)

        start_time = time.time()

        # 设置网格层次结构
        grid_params = []
        current_nx, current_ny = self.nx, self.ny
        current_nz = self.nz # nz 维度不参与粗化

        print(f"构建网格层次结构:")
        while min(current_nx, current_ny) >= 4 and len(grid_params) < self.max_levels:
            grid_params.append((current_nx, current_ny, current_nz))
            print(f"  Level {len(grid_params)-1}: {current_nx}x{current_ny}x{current_nz}")
            current_nx = max(2, current_nx // 2)
            current_ny = max(2, current_ny // 2)

        num_levels = len(grid_params)
        print(f"总共 {num_levels} 层网格")

        # 创建各层矩阵向量算子和右端项
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

        # 反转层次（从细到粗）
        op_hierarchy.reverse()
        b_hierarchy.reverse()
        u_hierarchy.reverse()
        grid_params.reverse()

        print(f"\n开始多重网格迭代:")
        print("-" * 40)

        # 主迭代循环
        for iteration in range(self.max_iterations):
            print(f"\n迭代 {iteration + 1}:")

            # 执行V-循环
            u_hierarchy[-1] = self.v_cycle(op_hierarchy,
                                             b_hierarchy, u_hierarchy, grid_params)

            # 计算最细网格上的残差
            op_finest = op_hierarchy[-1]
            b_finest = b_hierarchy[-1]
            u_finest = u_hierarchy[-1]
            
            finest_residual = self.compute_residual(op_finest, b_finest, u_finest)
            residual_norm = np.linalg.norm(finest_residual)
            self.convergence_history.append(residual_norm)

            print(f"  迭代 {iteration + 1} 完成，残差范数: {residual_norm:.4e}")

            # 检查收敛
            if residual_norm < self.tolerance:
                print(f"  ✓ 收敛达到容差 {self.tolerance}")
                break

            # 自适应准则
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

        # 返回最细网格上的解，并reshape为三维
        return u_hierarchy[-1].reshape((self.nz, self.ny, self.nx))

    def verify_solution(self, solution, alpha=1.0, beta=1.0j, func_type='sine'):
        """验证解的正确性"""
        print("\n验证解的正确性:")
        print("-" * 30)

        # 重新创建矩阵向量算子和右端项
        op = self.create_operator(self.nx, self.ny, self.nz, alpha, beta)
        b = self.create_rhs(self.nx, self.ny, self.nz, func_type)
        u_flat = solution.flatten()

        # 计算 A*u - b
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

    def plot_results(self, solution):
        """可视化结果"""
        print("\n生成可视化图像...")

        # Plot 2D slices if nz > 1, otherwise plot original 2D views
        if self.nz > 1:
            # For 3D results, plot cross-sections (e.g., middle slice)
            mid_z_slice_idx = self.nz // 2
            solution_2d_slice = solution[mid_z_slice_idx, :, :]
            plot_title_suffix = f" (Z-slice at k={mid_z_slice_idx})"
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Adaptive Multigrid Complex Solution Results{plot_title_suffix}', fontsize=16)

            x = np.linspace(0, 1, self.nx)
            y = np.linspace(0, 1, self.ny)
            X, Y = np.meshgrid(x, y)

            # Real Part
            im1 = axes[0, 0].contourf(X, Y, np.real(solution_2d_slice), levels=20, cmap='RdBu_r')
            axes[0, 0].set_title('Solution - Real Part')
            axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')
            plt.colorbar(im1, ax=axes[0, 0])

            # Imaginary Part
            im2 = axes[0, 1].contourf(X, Y, np.imag(solution_2d_slice), levels=20, cmap='RdBu_r')
            axes[0, 1].set_title('Solution - Imaginary Part')
            axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('y')
            plt.colorbar(im2, ax=axes[0, 1])

            # Magnitude
            im3 = axes[0, 2].contourf(X, Y, np.abs(solution_2d_slice), levels=20, cmap='viridis')
            axes[0, 2].set_title('Solution - Magnitude')
            axes[0, 2].set_xlabel('x'); axes[0, 2].set_ylabel('y')
            plt.colorbar(im3, ax=axes[0, 2])

            # Convergence History
            axes[1, 0].semilogy(range(1, len(self.convergence_history) + 1),
                                 self.convergence_history, 'b-o', markersize=4)
            axes[1, 0].set_title('Convergence History')
            axes[1, 0].set_xlabel('Iteration'); axes[1, 0].set_ylabel('Residual Norm')
            axes[1, 0].grid(True)

            # Phase
            phase = np.angle(solution_2d_slice)
            im4 = axes[1, 1].contourf(X, Y, phase, levels=20, cmap='hsv')
            axes[1, 1].set_title('Solution - Phase')
            axes[1, 1].set_xlabel('x'); axes[1, 1].set_ylabel('y')
            plt.colorbar(im4, ax=axes[1, 1])

            # 3D surface plot (Magnitude of the slice)
            ax3d = fig.add_subplot(2, 3, 6, projection='3d')
            surf = ax3d.plot_surface(X, Y, np.abs(solution_2d_slice), cmap='viridis', alpha=0.8)
            ax3d.set_title('Solution Magnitude (2D Slice) - 3D View')
            ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('|u|')
            
        else: # Original 2D plotting for nz=1
            solution_2d = solution.reshape((self.ny, self.nx))
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Adaptive Multigrid Complex Solution Results', fontsize=16)

            x = np.linspace(0, 1, self.nx)
            y = np.linspace(0, 1, self.ny)
            X, Y = np.meshgrid(x, y)

            # Real Part
            im1 = axes[0, 0].contourf(X, Y, np.real(solution_2d), levels=20, cmap='RdBu_r')
            axes[0, 0].set_title('Solution - Real Part')
            axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')
            plt.colorbar(im1, ax=axes[0, 0])

            # Imaginary Part
            im2 = axes[0, 1].contourf(X, Y, np.imag(solution_2d), levels=20, cmap='RdBu_r')
            axes[0, 1].set_title('Solution - Imaginary Part')
            axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('y')
            plt.colorbar(im2, ax=axes[0, 1])

            # Magnitude
            im3 = axes[0, 2].contourf(X, Y, np.abs(solution_2d), levels=20, cmap='viridis')
            axes[0, 2].set_title('Solution - Magnitude')
            axes[0, 2].set_xlabel('x'); axes[0, 2].set_ylabel('y')
            plt.colorbar(im3, ax=axes[0, 2])

            # Convergence History
            axes[1, 0].semilogy(range(1, len(self.convergence_history) + 1),
                                 self.convergence_history, 'b-o', markersize=4)
            axes[1, 0].set_title('Convergence History')
            axes[1, 0].set_xlabel('Iteration'); axes[1, 0].set_ylabel('Residual Norm')
            axes[1, 0].grid(True)

            # Phase
            phase = np.angle(solution_2d)
            im4 = axes[1, 1].contourf(X, Y, phase, levels=20, cmap='hsv')
            axes[1, 1].set_title('Solution - Phase')
            axes[1, 1].set_xlabel('x'); axes[1, 1].set_ylabel('y')
            plt.colorbar(im4, ax=axes[1, 1])

            # 3D surface plot (Magnitude)
            ax3d = fig.add_subplot(2, 3, 6, projection='3d')
            surf = ax3d.plot_surface(X, Y, np.abs(solution_2d), cmap='viridis', alpha=0.8)
            ax3d.set_title('Solution Magnitude - 3D View')
            ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('|u|')

        plt.tight_layout()
        plt.show()
        solve_time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        plt.savefig(
            f"Adaptive_Multigrid_Complex_Solution_Results_{solve_time_str}.png", dpi=300)

        print("可视化完成!")


# 主程序
if __name__ == "__main__":
    print("自适应多重网格复数求解器演示")
    print("=" * 50)

    # 创建求解器实例 - 现在使用矩形网格 nx x ny x nz
    nx = 32  # x方向网格大小
    ny = 64  # y方向网格大小
    nz = 8   # z方向网格大小 (不参与粗化)
    
    # 可以自定义dtype
    solver = AdaptiveMultigridComplex(
        nx=nx, ny=ny, nz=nz, max_levels=5, tolerance=1e-8, max_iterations=1000, dtype=np.complex64) # Example with complex64

    # 测试不同的问题参数
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

        # 求解
        solution = solver.solve(
            alpha=case["alpha"], beta=case["beta"], func_type=case["func_type"])

        # 验证
        residual_norm, relative_error = solver.verify_solution(
            solution, alpha=case["alpha"], beta=case["beta"], func_type=case["func_type"]
        )

        # 可视化
        solver.plot_results(solution)

        # 性能统计
        print(f"\n性能统计:")
        print(f"网格大小: {nx}x{ny}x{nz}")
        print(f"未知数个数: {nx*ny*nz}")
        print(f"收敛迭代次数: {len(solver.convergence_history)}")
        print(f"最终残差: {solver.convergence_history[-1]:.2e}")

        # 重置收敛历史
        solver.convergence_history = []

    print(f"\n{'='*80}")
    print("所有测试完成!")
    print(f"{'='*80}")