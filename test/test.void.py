import numpy as np

# 创建一个 NumPy 数组
arr = np.array([1, 2, 3, 4, 5])

# 获取数组的指针
pointer = arr.__array_interface__['data'][0]

print(f"数组的指针地址: {pointer}")

# 获取数组的指针
pointer = arr.ctypes.data

print(f"数组的指针地址: {pointer}")

import cupy as cp

# 创建一个 CuPy 数组
arr = cp.array([1, 2, 3, 4, 5])
print(arr.data)
# 获取数组的指针
pointer = arr.data.ptr
print(type(pointer))
print(f"数组的指针地址: {pointer}")
