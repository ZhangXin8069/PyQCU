import numpy as np
import cupy as cp
import ctypes

# 创建一个 numpy 数组
arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

# 获取数据的 ctypes 指针
data_ptr = arr.ctypes.data
print("Numpy data pointer:", data_ptr)
print("type:", type(data_ptr))
# 获取指针的原始地址，并使用 c_void_p 包装
void_ptr = ctypes.c_void_p(data_ptr)
print("Wrapped void pointer:", void_ptr)
print("type:", type(void_ptr))
print("Numpy data pointer:", data_ptr)
print("Wrapped void pointer:", void_ptr)

print("###################")
# 创建一个 numpy 数组
arr = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)

# 获取数据的 ctypes 指针
data_ptr = arr.data.ptr
print("Numpy data pointer:", data_ptr)
print("type:", type(data_ptr))
# 获取指针的原始地址，并使用 c_void_p 包装
void_ptr = ctypes.c_void_p(data_ptr)
print("Wrapped void pointer:", void_ptr)
print("type:", type(void_ptr))
print("Numpy data pointer:", data_ptr)
print("Wrapped void pointer:", void_ptr)