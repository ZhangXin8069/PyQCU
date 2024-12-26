import numpy as np
import cupy as cp
import ctypes

# 从NumPy数组获取void*指针
numpy_array = np.array([1, 2, 3], dtype=np.int32)
numpy_ptr = numpy_array.ctypes.data
print(type(numpy_array))
print(type(numpy_ptr))
numpy_void_ptr = ctypes.c_void_p(numpy_ptr)
print(type(numpy_void_ptr))
print(numpy_void_ptr)
# 从CuPy数组获取void*指针
cupy_array = cp.array([1, 2, 3], dtype=cp.int32)
cupy_ptr = cupy_array.data.ptr
print(type(cupy_array))
print(type(cupy_ptr))
cupy_void_ptr = ctypes.c_void_p(cupy_ptr)
print(type(cupy_void_ptr))
print(cupy_void_ptr)