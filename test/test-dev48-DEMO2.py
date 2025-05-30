import time
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import matplotlib.pyplot as plt
import numpy as np
np.Inf = np.inf


class AdaptiveMultigridComplex:
    """
    自适应多重网格方法求解复数系统
    适用于求解复数系数的椭圆型偏微分方程
    """

    def __init__(self, nx, ny, max_levels=5, tolerance=1e-8, max_iterations=100, dtype=np.complex64):
        """
        初始化自适应多重网格求解器

        参数:
        nx: x方向网格大小
        ny: y方向网格大小
        max_levels: 最大网格层数
        tolerance: 收敛容差
        max_iterations: 最大迭代次数
        """
        self.nx = nx
        self.ny = ny
        self.max_levels = max_levels
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.dtype = dtype
        self.convergence_history = []
        self.level_info = []
        
    @staticmethod
    def bicgstab(b, matvec, max_iter=100, tol=1e-6, x0=None):
        """
        BICGSTAB 迭代求解器
        参数:
        b: 右端项
        matvec: 矩阵向量乘法函数
        max_iter: 最大迭代次数
        tol: 收敛容差
        x0: 初始猜测
        """
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        print(f"norm of b: {np.linalg.norm(b)}")
        r = b - matvec(x)
        r0 = r.copy()
        rho = 1.0
        alpha = 1.0
        omega = 1.0
        v = np.zeros_like(b)
        p = np.zeros_like(b)
        
        residual_norm = np.linalg.norm(r)
        tol0 = residual_norm * tol
        
        for i in range(max_iter):
            rho1 = np.vdot(r0, r)
            if abs(rho1) < 1e-20:
                return x
                
            if i == 0:
                p = r.copy()
            else:
                beta = (rho1 / rho) * (alpha / omega)
                p = r + beta * (p - omega * v)
                
            v = matvec(p)
            alpha = rho1 / np.vdot(r0, v)
            s = r - alpha * v
            
            t = matvec(s)
            omega = np.vdot(t, s) / np.vdot(t, t)
            x = x + alpha * p + omega * s
            r = s - omega * t
            
            residual_norm = np.linalg.norm(r)
            if residual_norm < tol0:
                break
                
        return x

    def create_complex_matrix(self, nx, ny, alpha=1.0, beta=1.0j):
        """
        创建复数系数矩阵 (离散化的复数椭圆算子)
        -α∇²u - iβu = f
        返回矩阵和矩阵向量乘法函数
        """
        print(f"创建 {nx}x{ny} 复数系数矩阵...")
        print(f"  系数: α = {alpha}, β = {beta}")

        hx = 1.0 / (nx + 1)  # x方向网格间距
        hy = 1.0 / (ny + 1)  # y方向网格间距
        n = nx * ny

        # 主对角线: 2α(1/hx² + 1/hy²) + iβ
        main_diag = (2 * alpha * (1/hx**2 + 1/hy**2) + beta) * np.ones(n, dtype=self.dtype)

        # x方向上下对角线: -α/hx²
        off_diag_x = (-alpha / hx**2) * np.ones(n - 1, dtype=self.dtype)
        # 处理边界条件 (移除跨行的连接)
        for i in range(ny):
            for j in range(nx-1):
                idx = i * nx + j
                if j == nx-2:  # 每行最后一个元素的下一个元素在下一行
                    off_diag_x[idx] = 0

        # y方向上下对角线: -α/hy²
        off_diag_y = (-alpha / hy**2) * np.ones(n - nx, dtype=self.dtype)

        A = diags([off_diag_y, off_diag_x, main_diag, off_diag_x, off_diag_y],
                  [-nx, -1, 0, 1, nx], format='csr')
                  
        # 创建矩阵向量乘法函数
        def matvec(u):
            return A.dot(u)
            
        print(f"  矩阵创建完成: {A.shape}, 非零元素: {A.nnz}")
        return A, matvec

    def create_rhs(self, nx, ny, func_type='sine'):
        """创建右端项"""
        hx = 1.0 / (nx + 1)
        hy = 1.0 / (ny + 1)
        x = np.linspace(hx, 1-hx, nx)
        y = np.linspace(hy, 1-hy, ny)
        X, Y = np.meshgrid(x, y)

        if func_type == 'sine':
            # 复数正弦函数
            f = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y) * (1 + 1j)
        elif func_type == 'exponential':
            # 复数指数函数
            f = np.exp(X + 1j*Y)
        else:
            # 默认函数
            f = np.ones((ny, nx), dtype=self.dtype)

        return f.flatten()

    def restrict(self, u_fine, nx_fine, ny_fine):
        """限制算子：细网格到粗网格"""
        if nx_fine < 2 or ny_fine < 2:
            return u_fine
            
        nx_coarse = max(2, nx_fine // 2)
        ny_coarse = max(2, ny_fine // 2)

        u_fine_2d = u_fine.reshape((ny_fine, nx_fine))
        u_coarse_2d = np.zeros((ny_coarse, nx_coarse), dtype=self.dtype)

        # 全权重限制 (9点模板)
        for i in range(ny_coarse):
            for j in range(nx_coarse):
                ii, jj = 2*i+1, 2*j+1
                weight_sum = 0
                value_sum = 0

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = ii + di, jj + dj
                        if 0 <= ni < ny_fine and 0 <= nj < nx_fine:
                            if di == 0 and dj == 0:
                                weight = 4
                            elif di == 0 or dj == 0:
                                weight = 2
                            else:
                                weight = 1
                            weight_sum += weight
                            value_sum += weight * u_fine_2d[ni, nj]

                u_coarse_2d[i, j] = value_sum / weight_sum if weight_sum > 0 else 0

        return u_coarse_2d.flatten()

    def prolongate(self, u_coarse, nx_fine, ny_fine):
        """延拓算子：粗网格到细网格"""
        u_coarse_2d = u_coarse.reshape((ny_fine // 2, nx_fine // 2))
        u_fine_2d = np.zeros((ny_fine, nx_fine), dtype=self.dtype)

        # 双线性插值
        for i in range(ny_fine):
            for j in range(nx_fine):
                # 计算在粗网格中的位置
                i_coarse = (i * (ny_fine // 2)) / ny_fine
                j_coarse = (j * (nx_fine // 2)) / nx_fine

                i0, j0 = int(i_coarse), int(j_coarse)
                i1 = min(i0 + 1, (ny_fine // 2) - 1)
                j1 = min(j0 + 1, (nx_fine // 2) - 1)

                # 插值权重
                wi = i_coarse - i0
                wj = j_coarse - j0

                # 双线性插值
                u_fine_2d[i, j] = ((1-wi)*(1-wj)*u_coarse_2d[i0, j0] +
                                   wi*(1-wj)*u_coarse_2d[i1, j0] +
                                   (1-wi)*wj*u_coarse_2d[i0, j1] +
                                   wi*wj*u_coarse_2d[i1, j1])

        return u_fine_2d.flatten()

    def smooth(self, matvec, b, u, num_iterations=3):
        """光滑算子 - 使用BICGSTAB"""
        # 使用BICGSTAB进行光滑处理
        u = self.bicgstab(
            b, 
            matvec, 
            max_iter=num_iterations, 
            tol=1e-6,  # 较低精度，因为只是光滑步骤
            x0=u
        )
        return u

    def compute_residual(self, matvec, b, u):
        """计算残差"""
        return b - matvec(u)

    def v_cycle(self, matvec_hierarchy, b_hierarchy, u_hierarchy, grid_sizes, level=0):
        """V-循环"""
        current_level = len(matvec_hierarchy) - 1 - level
        nx, ny = grid_sizes[current_level]
        print(f"V-循环 level {level}, 当前层: {current_level}, 网格大小: {nx}x{ny}")
        
        matvec = matvec_hierarchy[current_level]
        b = b_hierarchy[current_level]
        u = u_hierarchy[current_level]

        # 如果是最粗网格，直接求解
        if current_level == 0 or level >= self.max_levels - 1:
            print(f"    最粗网格直接求解...")
            # 使用BICGSTAB求解
            u = self.bicgstab(
                b, 
                matvec, 
                max_iter=100, 
                tol=self.tolerance * 0.01,
                x0=u
            )
            residual = self.compute_residual(matvec, b, u)
            residual_norm = np.linalg.norm(residual)
            print(f"    残差范数: {residual_norm:.2e}")
            u_hierarchy[current_level] = u
            return u

        # 前光滑
        print(f"    前光滑...")
        u = self.smooth(matvec, b, u, num_iterations=5)
        u_hierarchy[current_level] = u

        # 计算残差
        residual = self.compute_residual(matvec, b, u)
        residual_norm = np.linalg.norm(residual)
        print(f"    残差范数: {residual_norm:.2e}")

        # 限制残差到粗网格
        if current_level > 0:
            r_coarse = self.restrict(residual, nx, ny)
            b_hierarchy[current_level - 1] = r_coarse
            u_hierarchy[current_level - 1] = np.zeros_like(r_coarse)

            # 递归调用粗网格
            e_coarse = self.v_cycle(
                matvec_hierarchy, b_hierarchy, u_hierarchy, grid_sizes, level + 1)

            # 延拓误差修正
            nx_fine, ny_fine = grid_sizes[current_level]
            e_fine = self.prolongate(e_coarse, nx_fine, ny_fine)
            u = u + e_fine
            u_hierarchy[current_level] = u

        # 后光滑
        print(f"    后光滑...")
        u = self.smooth(matvec, b, u, num_iterations=5)
        u_hierarchy[current_level] = u
        residual = self.compute_residual(matvec, b, u)
        residual_norm = np.linalg.norm(residual)
        print(f"    残差范数: {residual_norm:.2e}")
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
        grid_sizes = []
        current_nx, current_ny = self.nx, self.ny

        print(f"构建网格层次结构:")
        while min(current_nx, current_ny) >= 4 and len(grid_sizes) < self.max_levels:
            grid_sizes.append((current_nx, current_ny))
            print(f"  Level {len(grid_sizes)-1}: {current_nx}x{current_ny}")
            current_nx = max(2, current_nx // 2)
            current_ny = max(2, current_ny // 2)

        num_levels = len(grid_sizes)
        print(f"总共 {num_levels} 层网格")

        # 创建各层矩阵和右端项
        print(f"\n构建各层系统矩阵:")
        A_hierarchy = []  # 存储矩阵（用于验证）
        matvec_hierarchy = []  # 存储矩阵向量乘法函数
        b_hierarchy = []
        u_hierarchy = []

        for i, (nx, ny) in enumerate(grid_sizes):
            print(f"Level {i} ({nx}x{ny}):")
            A, matvec = self.create_complex_matrix(nx, ny, alpha, beta)
            b = self.create_rhs(nx, ny, func_type)
            u = np.zeros(nx * ny, dtype=self.dtype)

            A_hierarchy.append(A)
            matvec_hierarchy.append(matvec)
            b_hierarchy.append(b)
            u_hierarchy.append(u)

        # 反转层次（从细到粗）
        A_hierarchy.reverse()
        matvec_hierarchy.reverse()
        b_hierarchy.reverse()
        u_hierarchy.reverse()
        grid_sizes.reverse()

        print(f"\n开始多重网格迭代:")
        print("-" * 40)

        # 主迭代循环
        for iteration in range(self.max_iterations):
            print(f"\n迭代 {iteration + 1}:")

            # 执行V-循环
            u_hierarchy[-1] = self.v_cycle(matvec_hierarchy,
                                           b_hierarchy, u_hierarchy, grid_sizes)

            # 计算最细网格上的残差
            finest_matvec = matvec_hierarchy[-1]
            finest_residual = self.compute_residual(
                finest_matvec, b_hierarchy[-1], u_hierarchy[-1])
            residual_norm = np.linalg.norm(finest_residual)
            self.convergence_history.append(residual_norm)

            print(f"  迭代 {iteration + 1} 完成，残差范数: {residual_norm:.2e}")

            # 检查收敛
            if residual_norm < self.tolerance:
                print(f"  ✓ 收敛达到容差 {self.tolerance}")
                break

            # 自适应准则
            if self.adaptive_criterion(self.convergence_history):
                print(f"  注意: 收敛较慢，可能需要更多网格层")

        solve_time = time.time() - start_time

        print("\n" + "="*60)
        print("求解完成!")
        print(f"总迭代次数: {len(self.convergence_history)}")
        print(f"最终残差: {self.convergence_history[-1]:.2e}")
        print(f"求解时间: {solve_time:.4f} 秒")
        print("="*60)

        return u_hierarchy[-1].reshape((self.ny, self.nx))

    def verify_solution(self, solution, alpha=1.0, beta=1.0j, func_type='sine'):
        """验证解的正确性"""
        print("\n验证解的正确性:")
        print("-" * 30)

        # 重新计算矩阵和右端项
        A, matvec = self.create_complex_matrix(self.nx, self.ny, alpha, beta)
        b = self.create_rhs(self.nx, self.ny, func_type)
        u_flat = solution.flatten()

        # 计算 Au - b
        residual = b - matvec(u_flat)
        residual_norm = np.linalg.norm(residual)
        relative_error = residual_norm / np.linalg.norm(b)

        print(
            f"解的实部范围: [{np.real(solution).min():.4f}, {np.real(solution).max():.4f}]")
        print(
            f"解的虚部范围: [{np.imag(solution).min():.4f}, {np.imag(solution).max():.4f}]")
        print(
            f"解的模长范围: [{np.abs(solution).min():.4f}, {np.abs(solution).max():.4f}]")
        print(f"验证残差范数: {residual_norm:.2e}")
        print(f"相对误差: {relative_error:.2e}")

        if relative_error < 1e-6:
            print("✓ 解验证通过!")
        else:
            print("⚠ 解可能存在精度问题")

        return residual_norm, relative_error

    def plot_results(self, solution):
        """可视化结果"""
        print("\n生成可视化图像...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            'Adaptive Multigrid Complex Solution Results', fontsize=16)

        x = np.linspace(0, 1, self.nx)
        y = np.linspace(0, 1, self.ny)
        X, Y = np.meshgrid(x, y)

        # 解的实部
        im1 = axes[0, 0].contourf(X, Y, np.real(
            solution), levels=20, cmap='RdBu_r')
        axes[0, 0].set_title('Solution - Real Part')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 0])

        # 解的虚部
        im2 = axes[0, 1].contourf(X, Y, np.imag(
            solution), levels=20, cmap='RdBu_r')
        axes[0, 1].set_title('Solution - Imaginary Part')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0, 1])

        # 解的模长
        im3 = axes[0, 2].contourf(X, Y, np.abs(
            solution), levels=20, cmap='viridis')
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

        # 解的相位
        phase = np.angle(solution)
        im4 = axes[1, 1].contourf(X, Y, phase, levels=20, cmap='hsv')
        axes[1, 1].set_title('Solution - Phase')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        plt.colorbar(im4, ax=axes[1, 1])

        # 3D表面图（模长）
        ax3d = fig.add_subplot(2, 3, 6, projection='3d')
        surf = ax3d.plot_surface(X, Y, np.abs(
            solution), cmap='viridis', alpha=0.8)
        ax3d.set_title('Solution Magnitude - 3D View')
        ax3d.set_xlabel('x')
        ax3d.set_ylabel('y')
        ax3d.set_zlabel('|u|')

        plt.tight_layout()
        plt.show()
        solve_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        plt.savefig(
            f"Adaptive_Multigrid_Complex_Solution_Results_{solve_time}.png", dpi=300)

        print("可视化完成!")


# 主程序
if __name__ == "__main__":
    print("自适应多重网格复数求解器演示")
    print("=" * 50)

    # 创建求解器实例 - 现在使用矩形网格 nx x ny
    nx = 64  # x方向网格大小
    ny = 48  # y方向网格大小
    solver = AdaptiveMultigridComplex(
        nx=nx, ny=ny, max_levels=4, tolerance=1e-6, max_iterations=10000)

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
        print(f"网格大小: {nx}x{ny}")
        print(f"未知数个数: {nx*ny}")
        print(f"收敛迭代次数: {len(solver.convergence_history)}")
        print(f"最终残差: {solver.convergence_history[-1]:.2e}")

        # 重置收敛历史
        solver.convergence_history = []

    print(f"\n{'='*80}")
    print("所有测试完成!")
    print(f"{'='*80}")