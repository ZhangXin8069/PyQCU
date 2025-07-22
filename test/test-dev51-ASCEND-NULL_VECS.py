import torch
from pyqcu.ascend import dslash_parity
from pyqcu.ascend import inverse


# 创建初始向量 (复数示例)
dof = 5
null_vectors = torch.randn(dof, 32, 32, dtype=torch.complex64)

# 定义算子 (示例)


def matvec(x):
    # 实际应用中替换为真实的矩阵向量乘
    return torch.fft.fft2(x).real * x


# 生成近似零空间向量
result = inverse.give_null_vecs(
    null_vecs=null_vectors,
    matvec=matvec,
    tol=1e-4
)
# 验证正交性
for i in range(dof):
    print(f"torch.norm(matvec(result[i])):{torch.norm(matvec(result[i]))}")
    for j in range(i+1, dof):
        dot = torch.vdot(
            result[i].flatten().conj(),
            result[j].flatten()
        )
        print(f"<v{i}, v{j}> = {dot.item():.2e}")
