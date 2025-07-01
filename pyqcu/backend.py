class Backend:
    """
    Unified computational backend supporting multiple array libraries.
    
    Provides a consistent interface for array operations across NumPy, CuPy, 
    PyTorch, and JAX. Handles device management, data types, and random seeding.
    
    Args:
        lib (str): Computational backend ('numpy', 'cupy', 'torch', or 'jax')
        device (str): Execution device ('cpu', 'gpu', 'tpu', or specific like 'gpu:1')
        dtype (str): Data type ('float32', 'float64', 'complex64', etc.)
        seed (int, optional): Random seed for reproducible results
    """
    
    def __init__(self, lib='numpy', device='cpu', dtype='float32', seed=None):
        self.lib = lib.lower()
        self.device = device.lower()
        self.dtype_str = dtype
        self.seed = seed
        
        # Initialize backend components
        self._setup_lib()
        self._setup_dtype()
        self._setup_device()
        self.set_seed(seed)

    def _setup_lib(self):
        """Import and configure the selected computation library"""
        if self.lib == 'numpy':
            import numpy as xp
            self.xp = xp
        elif self.lib == 'cupy':
            import cupy as xp
            self.xp = xp
        elif self.lib == 'torch':
            import torch
            self.xp = torch
        elif self.lib == 'jax':
            import jax
            import jax.numpy as jnp
            self.xp = jnp
            self.jax = jax
            self._key = jax.random.PRNGKey(seed or 0)
        else:
            raise ValueError(f"Unsupported backend: {self.lib}")

    def _setup_dtype(self):
        """Configure data type mapping for the current backend"""
        if self.lib == 'torch':
            import torch
            self.dtype = {
                'float16': torch.float16,
                'float32': torch.float32,
                'float64': torch.float64,
                'complex32': torch.complex32,
                'complex64': torch.complex64,
                'complex128': torch.complex128,
                'int8': torch.int8,
                'int16': torch.int16,
                'int32': torch.int32,
                'int64': torch.int64,
                'uint8': torch.uint8,
                'bool': torch.bool
            }.get(self.dtype_str, torch.float32)
            
        elif self.lib == 'jax':
            self.dtype = {
                'float16': self.xp.float16,
                'float32': self.xp.float32,
                'float64': self.xp.float64,
                'complex32': self.xp.complex64,
                'complex64': self.xp.complex64,
                'complex128': self.xp.complex128,
                'int8': self.xp.int8,
                'int16': self.xp.int16,
                'int32': self.xp.int32,
                'int64': self.xp.int64,
                'uint8': self.xp.uint8,
                'bool': self.xp.bool_
            }.get(self.dtype_str, self.xp.float32)
            
        else:  # numpy/cupy
            self.dtype = self.dtype_str

    def _setup_device(self):
        """Configure execution device for hardware acceleration"""
        if self.lib == 'torch':
            import torch
            # Parse device identifier (e.g., 'gpu:2' -> ('gpu', 2))
            dev_type, _, dev_id = self.device.partition(':')
            dev_id = int(dev_id) if dev_id else 0
            
            if dev_type == 'gpu' and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            elif dev_type == 'npu' and not hasattr(torch, 'npu'):
                raise RuntimeError("NPU device requested but not supported in this PyTorch build")
                
            self.device_obj = torch.device(f'{dev_type}:{dev_id}' if dev_type != 'cpu' else 'cpu')
            
        elif self.lib == 'jax':
            # Parse device identifier (e.g., 'tpu:1')
            dev_type, _, dev_id = self.device.partition(':')
            dev_id = int(dev_id) if dev_id else 0
            
            # Get available devices of requested type
            try:
                devices = self.jax.devices(dev_type)
            except RuntimeError:
                devices = []
                
            if not devices:
                raise RuntimeError(f"No {dev_type.upper()} devices available")
            if dev_id >= len(devices):
                raise ValueError(f"Device index out of range. {dev_type.upper()} devices available: {len(devices)}")
                
            self.device_obj = devices[dev_id]
            
        else:  # numpy/cupy
            self.device_obj = None

    def set_seed(self, seed=None):
        """
        Set random seed for reproducible results
        
        Args:
            seed (int): Random seed value. Uses current seed if None.
        """
        if seed is not None:
            self.seed = seed
        if self.seed is None:
            return
            
        if self.lib == 'torch':
            import torch
            torch.manual_seed(self.seed)
            if self.device.startswith('cuda'):
                torch.cuda.manual_seed_all(self.seed)
        elif self.lib == 'numpy':
            import numpy as np
            np.random.seed(self.seed)
        elif self.lib == 'cupy':
            import cupy as cp
            cp.random.seed(self.seed)
        elif self.lib == 'jax':
            self._key = self.jax.random.PRNGKey(self.seed)

    # --------------------------
    # Array creation methods
    # --------------------------
    
    def array(self, data):
        """Create array from Python data structure"""
        if self.lib == 'torch':
            return self.xp.tensor(data, dtype=self.dtype, device=self.device_obj)
        elif self.lib == 'jax':
            arr = self.xp.array(data, dtype=self.dtype)
            return self.jax.device_put(arr, self.device_obj)
        return self.xp.array(data, dtype=self.dtype)

    def zeros(self, shape):
        """Create array filled with zeros"""
        if self.lib == 'torch':
            return self.xp.zeros(shape, dtype=self.dtype, device=self.device_obj)
        elif self.lib == 'jax':
            arr = self.xp.zeros(shape, dtype=self.dtype)
            return self.jax.device_put(arr, self.device_obj)
        return self.xp.zeros(shape, dtype=self.dtype)

    def ones(self, shape):
        """Create array filled with ones"""
        if self.lib == 'torch':
            return self.xp.ones(shape, dtype=self.dtype, device=self.device_obj)
        elif self.lib == 'jax':
            arr = self.xp.ones(shape, dtype=self.dtype)
            return self.jax.device_put(arr, self.device_obj)
        return self.xp.ones(shape, dtype=self.dtype)

    def eye(self, N, M=None):
        """Create identity matrix"""
        M = N if M is None else M
        if self.lib == 'torch':
            return self.xp.eye(N, M, dtype=self.dtype, device=self.device_obj)
        elif self.lib == 'jax':
            arr = self.xp.eye(N, M, dtype=self.dtype)
            return self.jax.device_put(arr, self.device_obj)
        return self.xp.eye(N, M, dtype=self.dtype)

    def linspace(self, start, stop, num):
        """Create linearly spaced values"""
        if self.lib == 'torch':
            return self.xp.linspace(start, stop, num, dtype=self.dtype, device=self.device_obj)
        elif self.lib == 'jax':
            arr = self.xp.linspace(start, stop, num, dtype=self.dtype)
            return self.jax.device_put(arr, self.device_obj)
        return self.xp.linspace(start, stop, num, dtype=self.dtype)

    def arange(self, start, stop=None, step=1):
        """Create values within an interval with fixed step"""
        if stop is None:
            stop = start
            start = 0
        if self.lib == 'torch':
            return self.xp.arange(start, stop, step, dtype=self.dtype, device=self.device_obj)
        elif self.lib == 'jax':
            arr = self.xp.arange(start, stop, step, dtype=self.dtype)
            return self.jax.device_put(arr, self.device_obj)
        return self.xp.arange(start, stop, step, dtype=self.dtype)

    def randn(self, *shape):
        """Create array with standard normal distribution"""
        if self.lib == 'torch':
            return self.xp.randn(*shape, dtype=self.dtype, device=self.device_obj)
        elif self.lib == 'jax':
            self._key, subkey = self.jax.random.split(self._key)
            arr = self.jax.random.normal(subkey, shape, dtype=self.dtype)
            return self.jax.device_put(arr, self.device_obj)
        return self.xp.random.randn(*shape).astype(self.dtype)

    def rand(self, *shape):
        """Create array with uniform distribution [0, 1)"""
        if self.lib == 'torch':
            return self.xp.rand(*shape, dtype=self.dtype, device=self.device_obj)
        elif self.lib == 'jax':
            self._key, subkey = self.jax.random.split(self._key)
            arr = self.jax.random.uniform(subkey, shape, dtype=self.dtype)
            return self.jax.device_put(arr, self.device_obj)
        return self.xp.random.rand(*shape).astype(self.dtype)

    def empty(self, shape):
        """Create uninitialized array (contents undefined)"""
        if self.lib == 'torch':
            return self.xp.empty(shape, dtype=self.dtype, device=self.device_obj)
        elif self.lib == 'jax':
            arr = self.xp.empty(shape, dtype=self.dtype)
            return self.jax.device_put(arr, self.device_obj)
        return self.xp.empty(shape, dtype=self.dtype)

    def full(self, shape, value):
        """Create array filled with specified value"""
        if self.lib == 'torch':
            return self.xp.full(shape, value, dtype=self.dtype, device=self.device_obj)
        elif self.lib == 'jax':
            arr = self.xp.full(shape, value, dtype=self.dtype)
            return self.jax.device_put(arr, self.device_obj)
        return self.xp.full(shape, value, dtype=self.dtype)

    # --------------------------
    # Array manipulation
    # --------------------------
    
    def transpose(self, x, axes=None):
        """Permute array dimensions"""
        if self.lib == 'torch':
            if axes is None:
                # Reverse all dimensions for matrix transpose
                return x.mT
            return x.permute(*axes)
        elif self.lib == 'jax':
            return self.xp.transpose(x, axes)
        return x.transpose(axes)

    def reshape(self, x, shape):
        """Reshape array without changing data"""
        return x.reshape(shape)

    def squeeze(self, x, axis=None):
        """Remove dimensions of size 1"""
        if self.lib == 'torch':
            return x.squeeze(dim=axis) if axis is not None else x.squeeze()
        return x.squeeze(axis=axis)

    def unsqueeze(self, x, axis):
        """Add new dimension at specified position"""
        if self.lib == 'torch':
            return x.unsqueeze(dim=axis)
        return self.xp.expand_dims(x, axis)

    def expand_dims(self, x, axis):
        """Alias for unsqueeze"""
        return self.unsqueeze(x, axis)

    def repeat(self, x, repeats, axis=None):
        """Repeat elements along specified axis"""
        if self.lib == 'torch':
            return x.repeat(repeats, dim=axis)
        return self.xp.repeat(x, repeats, axis=axis)

    def tile(self, x, reps):
        """Repeat array by tiling"""
        return self.xp.tile(x, reps)

    def concatenate(self, arrays, axis=0):
        """Join arrays along existing axis"""
        if self.lib == 'torch':
            return self.xp.cat(arrays, dim=axis)
        return self.xp.concatenate(arrays, axis=axis)

    def stack(self, arrays, axis=0):
        """Join arrays along new axis"""
        if self.lib == 'torch':
            return self.xp.stack(arrays, dim=axis)
        return self.xp.stack(arrays, axis=axis)

    def split(self, x, indices_or_sections, axis=0):
        """Split array into multiple sub-arrays"""
        if self.lib == 'torch':
            return self.xp.split(x, indices_or_sections, dim=axis)
        return self.xp.split(x, indices_or_sections, axis=axis)

    def where(self, condition, x, y):
        """Element-wise selection based on condition"""
        return self.xp.where(condition, x, y)

    def flip(self, x, axis=None):
        """Reverse array elements along specified axis"""
        if self.lib == 'torch':
            return self.xp.flip(x, dims=axis) if axis is not None else self.xp.flip(x)
        return self.xp.flip(x, axis=axis)

    def roll(self, x, shift, axis=None):
        """Roll array elements along specified axis"""
        return self.xp.roll(x, shift, axis=axis)

    # --------------------------
    # Mathematical operations
    # --------------------------
    
    def sum(self, x, axis=None, keepdims=False):
        """Sum of array elements"""
        if self.lib == 'torch':
            return x.sum(dim=axis, keepdim=keepdims)
        return x.sum(axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        """Arithmetic mean"""
        if self.lib == 'torch':
            return x.mean(dim=axis, keepdim=keepdims)
        return x.mean(axis=axis, keepdims=keepdims)

    def max(self, x, axis=None, keepdims=False):
        """Maximum values"""
        if self.lib == 'torch':
            return x.max(dim=axis, keepdim=keepdims).values
        return x.max(axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        """Minimum values"""
        if self.lib == 'torch':
            return x.min(dim=axis, keepdim=keepdims).values
        return x.min(axis=axis, keepdims=keepdims)

    def argmax(self, x, axis=None):
        """Indices of maximum values"""
        if self.lib == 'torch':
            return x.argmax(dim=axis)
        return x.argmax(axis=axis)

    def argmin(self, x, axis=None):
        """Indices of minimum values"""
        if self.lib == 'torch':
            return x.argmin(dim=axis)
        return x.argmin(axis=axis)

    def std(self, x, axis=None, keepdims=False):
        """Standard deviation"""
        if self.lib == 'torch':
            return x.std(dim=axis, unbiased=False, keepdim=keepdims)
        return x.std(axis=axis, keepdims=keepdims)

    def var(self, x, axis=None, keepdims=False):
        """Variance"""
        if self.lib == 'torch':
            return x.var(dim=axis, unbiased=False, keepdim=keepdims)
        return x.var(axis=axis, keepdims=keepdims)

    def abs(self, x):
        """Absolute value"""
        return self.xp.abs(x)

    def real(self, x):
        """Real part of complex array"""
        return x.real

    def imag(self, x):
        """Imaginary part of complex array"""
        return x.imag

    def conj(self, x):
        """Complex conjugate"""
        return self.xp.conj(x)

    def exp(self, x):
        """Exponential"""
        return self.xp.exp(x)

    def sin(self, x):
        """Sine function"""
        return self.xp.sin(x)

    def cos(self, x):
        """Cosine function"""
        return self.xp.cos(x)

    def tan(self, x):
        """Tangent function"""
        return self.xp.tan(x)

    def sinh(self, x):
        """Hyperbolic sine"""
        return self.xp.sinh(x)

    def cosh(self, x):
        """Hyperbolic cosine"""
        return self.xp.cosh(x)

    def tanh(self, x):
        """Hyperbolic tangent"""
        return self.xp.tanh(x)

    def sqrt(self, x):
        """Square root"""
        return self.xp.sqrt(x)

    def log(self, x):
        """Natural logarithm"""
        return self.xp.log(x)

    def log10(self, x):
        """Base-10 logarithm"""
        return self.xp.log10(x)

    def log2(self, x):
        """Base-2 logarithm"""
        return self.xp.log2(x)

    def clip(self, x, min_val, max_val):
        """Clip values to specified range"""
        if self.lib == 'torch':
            return x.clamp(min_val, max_val)
        return self.xp.clip(x, min_val, max_val)

    def sign(self, x):
        """Element-wise sign indicator"""
        return self.xp.sign(x)

    def round(self, x):
        """Round to nearest integer"""
        if self.lib == 'torch':
            return x.round()
        return self.xp.round(x)

    def floor(self, x):
        """Floor function"""
        return self.xp.floor(x)

    def ceil(self, x):
        """Ceiling function"""
        return self.xp.ceil(x)

    def matmul(self, a, b):
        """Matrix multiplication"""
        return self.xp.matmul(a, b)

    def dot(self, a, b):
        """Dot product"""
        return self.xp.dot(a, b)

    def einsum(self, subscripts, *operands):
        """Einstein summation convention"""
        return self.xp.einsum(subscripts, *operands)

    def power(self, base, exponent):
        """Element-wise exponentiation"""
        return self.xp.power(base, exponent)

    def add(self, a, b):
        """Element-wise addition"""
        return a + b

    def subtract(self, a, b):
        """Element-wise subtraction"""
        return a - b

    def multiply(self, a, b):
        """Element-wise multiplication"""
        return a * b

    def divide(self, a, b):
        """Element-wise division"""
        return a / b

    # --------------------------
    # Type and device conversion
    # --------------------------
    
    def astype(self, x, dtype_str):
        """Convert array to different data type"""
        if self.lib == 'torch':
            import torch
            dtype_map = {
                'float16': torch.float16,
                'float32': torch.float32,
                'float64': torch.float64,
                'complex64': torch.complex64,
                'complex128': torch.complex128,
                'int8': torch.int8,
                'int16': torch.int16,
                'int32': torch.int32,
                'int64': torch.int64,
                'uint8': torch.uint8,
                'bool': torch.bool
            }
            return x.to(dtype=dtype_map[dtype_str])
            
        elif self.lib == 'jax':
            dtype_map = {
                'float16': self.xp.float16,
                'float32': self.xp.float32,
                'float64': self.xp.float64,
                'complex64': self.xp.complex64,
                'complex128': self.xp.complex128,
                'int8': self.xp.int8,
                'int16': self.xp.int16,
                'int32': self.xp.int32,
                'int64': self.xp.int64,
                'uint8': self.xp.uint8,
                'bool': self.xp.bool_
            }
            return x.astype(dtype_map[dtype_str])
            
        return x.astype(dtype_str)

    def to_device(self, x, device):
        """
        Transfer array to specified device
        
        Args:
            x: Input array
            device: Target device identifier (e.g., 'gpu:1')
            
        Returns:
            Array transferred to target device
        """
        if self.lib == 'torch':
            dev_type, _, dev_id = device.partition(':')
            dev_id = int(dev_id) if dev_id else 0
            return x.to(device=f'{dev_type}:{dev_id}')
            
        elif self.lib == 'jax':
            dev_type, _, dev_id = device.partition(':')
            dev_id = int(dev_id) if dev_id else 0
            devices = self.jax.devices(dev_type)
            if dev_id >= len(devices):
                raise ValueError(f"Device index out of range. {dev_type.upper()} devices available: {len(devices)}")
            return self.jax.device_put(x, devices[dev_id])
            
        return x  # numpy/cupy don't support device transfer

    def to_numpy(self, x):
        """Convert array to NumPy ndarray on CPU"""
        if self.lib == 'torch':
            return x.detach().cpu().numpy()
        elif self.lib == 'cupy':
            return self.xp.asnumpy(x)
        elif self.lib == 'jax':
            return self.jax.device_get(x)
        return x

    def move(self, x):
        """
        Convert array to current backend configuration
        
        Ensures array uses the currently configured dtype and device
        """
        if self.lib == 'torch':
            return x.to(self.device_obj, dtype=self.dtype)
        elif self.lib == 'cupy':
            return self.xp.asarray(x, dtype=self.dtype)
        elif self.lib == 'jax':
            arr = x.astype(self.dtype)
            return self.jax.device_put(arr, self.device_obj)
        else:  # numpy
            import numpy as np
            return np.array(x, dtype=self.dtype)

    # --------------------------
    # Utility methods
    # --------------------------
    
    def info(self):
        """Print current backend configuration"""
        print(f"[Backend] lib={self.lib}, dtype={self.dtype_str}, device={self.device}")

    def is_array(self, x):
        """Check if object is an array in current backend"""
        if self.lib == 'torch':
            return self.xp.is_tensor(x)
        elif self.lib == 'numpy':
            return isinstance(x, self.xp.ndarray)
        elif self.lib == 'cupy':
            return isinstance(x, self.xp.ndarray)
        elif self.lib == 'jax':
            return isinstance(x, self.jax.Array)
        return False

    def get_device_info(self):
        """Get information about current execution device"""
        if self.lib == 'torch' and self.device.startswith('cuda'):
            import torch
            return {
                "device": self.device,
                "name": torch.cuda.get_device_name(self.device_obj),
                "memory": torch.cuda.get_device_properties(self.device_obj).total_memory,
                "capability": torch.cuda.get_device_capability(self.device_obj)
            }
        elif self.lib == 'jax':
            return {
                "device": str(self.device_obj),
                "platform": self.device_obj.platform,
                "device_kind": self.device_obj.device_kind,
                "process_index": self.device_obj.process_index
            }
        return {"device": self.device}

    def shape(self, x):
        """Get array dimensions"""
        return x.shape

    def ndim(self, x):
        """Get number of array dimensions"""
        return x.ndim

    def size(self, x):
        """Get total number of elements"""
        return x.size

    def dtype_of(self, x):
        """Get data type of array as string"""
        if self.lib == 'torch':
            return str(x.dtype).split('.')[-1]
        return x.dtype.name

    def device_of(self, x):
        """Get device of array as string"""
        if self.lib == 'torch':
            return str(x.device)
        elif self.lib == 'jax':
            return str(x.device())
        return 'cpu'
        
    def data_ptr(self, x):
        """
        Get the memory address pointer of the array data
        
        Args:
            x: Input array
            
        Returns:
            int: Memory address of the array data
            
        Note:
            - NumPy: Returns .ctypes.data value
            - CuPy: Returns .data.ptr value
            - PyTorch: Returns .data_ptr() value
            - JAX: Not supported (raises NotImplementedError)
        """
        if self.lib == 'numpy':
            return x.ctypes.data
        elif self.lib == 'cupy':
            return x.data.ptr
        elif self.lib == 'torch':
            return x.data_ptr()
        elif self.lib == 'jax':
            raise NotImplementedError("data_ptr() not supported for JAX arrays")
        else:
            raise ValueError(f"Unsupported backend: {self.lib}")