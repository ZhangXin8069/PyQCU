# import torch
# from pyqcu.ascend import dslash_parity
# from pyqcu.ascend import inverse
# dof = 12
# latt_size = (8, 8, 8, 8)
# latt_size = (4, 4, 4, 4)
# kappa = 0.125
# # dtype = torch.complex128
# dtype = torch.complex64
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# # Initialize lattice gauge theory
# wilson = dslash_parity.wilson_parity(
#     latt_size=latt_size,
#     kappa=kappa,
#     dtype=dtype,
#     device=device,
#     verbose=False
# )
# clover = dslash_parity.clover_parity(
#     latt_size=latt_size,
#     kappa=kappa,
#     dtype=dtype,
#     device=device,
#     verbose=False
# )
# U = wilson.generate_gauge_field(sigma=0.1, seed=42)
# null_vectors = torch.randn(dof, 4, 3, latt_size[3], latt_size[2], latt_size[1], latt_size[0],
#                            dtype=dtype, device=device)
# clover_term = clover.make_clover(U=U)


# def matvec(src: torch.Tensor, U: torch.Tensor = U) -> torch.Tensor:
#     return wilson.give_wilson(src, U)+clover.give_clover(clover=clover_term, src=src)
#     # return wilson.give_wilson(src, U)


# # 生成近似零空间向量
# result = inverse.give_null_vecs(
#     null_vecs=null_vectors,
#     matvec=matvec,
# )
import torch


def find_null_space_iterative(
    matvec: callable,
    dim: int,
    k: int = 1,
    tol: float = 1e-6,
    max_iter: int = 1000,
    verbose: bool = True
) -> torch.Tensor:
    """
    使用迭代法求零空间向量（对应最小特征值的特征向量）

    Args:
        matvec: 矩阵向量乘法函数 A(v) -> tensor
        dim: 向量维度
        k: 要寻找的零空间向量数量
        tol: 收敛容差
        max_iter: 最大迭代次数
        verbose: 是否打印进度

    Returns:
        null_vecs: 零空间向量 (k, dim)
    """
    # 初始化随机向量
    null_vecs = torch.randn(k, dim)
    null_vecs /= torch.norm(null_vecs, dim=1, keepdim=True)

    # 迭代改进
    for i in range(max_iter):
        # 计算残差 A·v
        residuals = torch.stack([matvec(v) for v in null_vecs])

        # 计算残差范数
        res_norms = torch.norm(residuals, dim=1)
        max_res = torch.max(res_norms).item()

        if verbose and (i % 10 == 0 or max_res < tol):
            print(f"Iter {i}: Max residual = {max_res:.4e}")

        # 检查收敛
        if max_res < tol:
            if verbose:
                print(f"Converged after {i} iterations")
            break

        # 更新向量: v = v - α·A·Aᵀ·v
        # 使用简单的梯度下降
        for j in range(k):
            grad = matvec(residuals[j])
            null_vecs[j] -= 0.1 * grad
            null_vecs[j] /= torch.norm(null_vecs[j])

    return null_vecs

# 示例使用


def matvec(v):
    """示例矩阵的矩阵向量乘法"""
    A = torch.tensor([[1.0, 0.5, 0.0],
                      [0.5, 1.0, 0.5],
                      [0.0, 0.5, 1.0]], dtype=torch.complex64)
    A += 1j*A
    return A @ v


null_vecs = find_null_space_iterative(matvec, dim=3, k=1, tol=1e-6)
print("零空间向量:", null_vecs.squeeze())
print("A·v 范数:", torch.norm(matvec(null_vecs.squeeze())).item())
