import numpy as cp
import mg
from cupyx.scipy.linalg import expm
import os
import numpy as np
import cupy
# 定义矩阵乘法
def apply_clover(Vin, op):
    Vout = cp.zeros(Vin.shape, dtype=Vin.dtype)
    Vout = cp.einsum('...ab,...b->...a', op.clover, Vin)
    return Vout
def apply_hopping_x_p(Vin, op):
    Vout = cp.zeros(Vin.shape, dtype=Vin.dtype)
    Vout = (cp.einsum('...ab,...b->...a',
            op.hopping[0, :], cp.roll(Vin, -1, axis=0)))
    return Vout
def apply_hopping_x_m(Vin, op):
    Vout = cp.zeros(Vin.shape, dtype=Vin.dtype)
    Vout = (cp.einsum('...ab,...b->...a',
            op.hopping[1, :], cp.roll(Vin, 1, axis=0)))
    return Vout
def apply_hopping_y_p(Vin, op):
    Vout = cp.zeros(Vin.shape, dtype=Vin.dtype)
    Vout = (cp.einsum('...ab,...b->...a',
            op.hopping[2, :], cp.roll(Vin, -1, axis=1)))
    return Vout
def apply_hopping_y_m(Vin, op):
    Vout = cp.zeros(Vin.shape, dtype=Vin.dtype)
    Vout = (cp.einsum('...ab,...b->...a',
            op.hopping[3, :], cp.roll(Vin, 1, axis=1)))
    return Vout
def apply_hopping(Vin, op):
    Vout = cp.zeros(Vin.shape, dtype=Vin.dtype)
    Vout = apply_hopping_x_p(Vin, op) + apply_hopping_x_m(Vin, op) + \
        apply_hopping_y_p(Vin, op) + apply_hopping_y_m(Vin, op)
    return Vout
def apply_mat(Vin, op):
    Vin = apply_clover(Vin, op) + apply_hopping(Vin, op)
    return Vin
# 格点参数
class operator_para:
    nx = 0
    ny = 0
    nc = 0
    volume = 0
    if_fine = 0
    hopping = cp.zeros((4, nx, ny, nc, nc*2)).view(cp.complex128)
    clover = cp.zeros((nx, ny, nc, nc*2)).view(cp.complex128)
    def __init__(self, nx, ny, nc=2, U=0, if_fine=0):
        self.nx = nx
        self.ny = ny
        self.nc = nc
        self.U = U
        self.volume = nx*ny
        self.if_fine = if_fine
        self.hopping = cp.zeros((4, nx, ny, nc, nc*2)).view(cp.complex128)
        self.clover = cp.zeros((nx, ny, nc, nc*2)).view(cp.complex128)
        for i in range(0, self.nc):
            self.clover[:, :, i, i] = 1
        if self.if_fine != 0:
            print(self.hopping.shape)
            kappa = -0.137
            self.hopping[0, :] = kappa*cp.roll(self.U[0, :], -1, axis=0)  # x+
            self.hopping[1, :] = kappa * \
                cp.conj(self.U[0, :]).transpose(0, 1, 3, 2)  # x-
            self.hopping[2, :] = kappa*cp.roll(self.U[1, :], -1, axis=1)  # y+
            self.hopping[3, :] = kappa * \
                cp.conj(self.U[1, :]).transpose(0, 1, 3, 2)  # y-
        # for i in range(0,self.nc):
        #     self.hopping[0,:,:,i,i] = 1
        #     self.hopping[1,:,:,i,i] = 1
        #     self.hopping[2,:,:,i,i] = 1
        #     self.hopping[3,:,:,i,i] = 1
# 生成组态U
# SU(2)
# 定义 Pauli 矩阵
sigma1 = cp.array([[0, 1], [1, 0]], dtype=complex)
sigma2 = cp.array([[0, -1j], [1j, 0]], dtype=complex)
sigma3 = cp.array([[1, 0], [0, -1]], dtype=complex)
eye = cp.array([[1, 0], [0, 1]], dtype=complex)
pauli = [sigma1, sigma2, sigma3]
def generate_su2_matrix():
    """
    生成一个随机的 SU(2) 矩阵。
    """
    # 生成三个随机实数系数
    a = cp.random.randn(3)
    # 构造李代数元素 X
    X = sum(a[k] * pauli[k] for k in range(3))
    # 计算 U = exp(i X)
    X = cupy.asarray(X)
    U = expm(1j * X)
    return U
def generate_E_m_su2_u1():
    a = cp.random.randn(1)  # U(1)
    E_m_su2 = eye-sigma1
    return a*E_m_su2
def generate_E_p_su2_u1():
    a = cp.random.randn(1)  # U(1)
    E_m_su2 = eye+sigma1
    return a*E_m_su2
def generate_E_m_su2_u1_2():
    a = cp.random.randn(1)  # U(1)
    E_m_su2 = eye-sigma2
    return a*E_m_su2
def generate_E_p_su2_u1_2():
    a = cp.random.randn(1)  # U(1)
    E_m_su2 = eye+sigma2
    return a*E_m_su2
# SU(3)
# 定义 Gell-Mann 矩阵
lambda1 = cp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=cp.complex128)
lambda2 = cp.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=cp.complex128)
lambda3 = cp.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=cp.complex128)
lambda4 = cp.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=cp.complex128)
lambda5 = cp.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=cp.complex128)
lambda6 = cp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=cp.complex128)
lambda7 = cp.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=cp.complex128)
lambda8 = (1 / cp.sqrt(3)) * \
    cp.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=cp.complex128)
gell_mann = [lambda1, lambda2, lambda3,
             lambda4, lambda5, lambda6, lambda7, lambda8]
def generate_su3_matrix():
    # 随机选择系数
    a = cp.random.randn(8)
    # 构造 X = sum a_i * lambda_i
    X = sum(a[i] * gell_mann[i] for i in range(8))
    # 计算 U = exp(i X)
    X = cupy.asarray(X)
    U = expm(1j * X)
    return U
def generate_large_matrix(x, y, nc):
    """
    生成形状为 (x, y, 2, 2) 的矩阵，每个 (2, 2) 子矩阵是随机的 SU(2) 矩阵。
    参数：
        x (int): 第一维的大小
        y (int): 第二维的大小
    返回：
        numpy.ndarray: 形状为 (x, y, 2, 2) 的复数矩阵
    """
    # 初始化大矩阵
    large_matrix = cp.zeros((4, x, y, nc, nc), dtype=cp.complex128)
    # 为每个 (i, j) 位置生成 SU(2) 矩阵
    # for k in range(0,4):
    #     for i in range(x):
    #         for j in range(y):
    #             if(nc == 2):
    #                 large_matrix[k, i, j] = generate_su2_matrix().get()
    #             elif(nc==3):
    #                 large_matrix[k, i, j] = generate_su3_matrix().get()
    #             else:
    #                 print("generate_U error! nc size is not supported")
    if (nc == 2):
        # large_matrix[:, :, :] = generate_su2_matrix().get()
        large_matrix[0, :, :] = generate_E_m_su2_u1()
        large_matrix[1, :, :] = generate_E_p_su2_u1()
        large_matrix[2, :, :] = generate_E_m_su2_u1_2()
        large_matrix[3, :, :] = generate_E_p_su2_u1_2()
    elif (nc == 3):
        large_matrix[:, :, :] = generate_su3_matrix().get()
    return large_matrix
# 生成或读取组态U
def load_or_generate_U(nx, ny, nc, folder_path="."):
    """
    检查文件夹下是否有U_nx_ny_nc.npy文件，若存在则读取，否则生成并保存。
    参数：
    nx, ny, nc: 用于构造文件名和生成U的参数
    folder_path: 文件夹路径，默认为当前目录
    返回：
    U: CuPy complex128数组
    """
    # 构造文件名
    file_name = f"U_{nx}_{ny}_{nc}.npy"
    file_path = os.path.join(folder_path, file_name)
    # 检查文件是否存在
    if os.path.exists(file_path):
        print(f"找到文件 {file_name}，正在加载...")
        # 读取NumPy数组并转换为CuPy数组
        U_np = np.load(file_path)
        U = cp.array(U_np, dtype=cp.complex128)
    else:
        print(f"未找到文件 {file_name}，正在生成...")
        # 调用aaa函数生成U
        U = generate_large_matrix(nx, ny, nc)
        # 确保U是complex128类型
        U = U.astype(cp.complex128)
        # 将CuPy数组转换为NumPy数组以保存
        U_np = U
        # 保存到文件
        np.save(file_path, U_np)
        print(f"已保存文件 {file_name}")
    return U
