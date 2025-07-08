# %%
import cupy as cp
import numpy as np
from pyqcu.cuda import define
from pyqcu.cuda import io
from pyqcu.cuda import qcu
from pyqcu.cuda.set import params, argv, set_ptrs
import h5py
print('My rank is ', define.rank)
# %%
gauge_filename = f"quda_wilson-dslash-gauge_-{params[define._LAT_X_]}-{params[define._LAT_Y_]}-{params  [define._LAT_Z_]}-{params[define._LAT_T_]}-{params[define._LAT_XYZT_]}-{params[define._GRID_X_]}-{params[define._GRID_Y_]}-{params[define._GRID_Z_]}-{params[define._GRID_T_]}-{params[define._PARITY_]}-{params[define._NODE_RANK_]}-{params[define._NODE_SIZE_]}-{params[define._DAGGER_]}-f.bin"
print("Gauge filename:", gauge_filename)
gauge = cp.fromfile(gauge_filename, dtype=cp.complex64,
                    count=params[define._LAT_XYZT_]*define._LAT_DCC_)
gauge = io.gauge2ccdptzyx(gauge, params)
print("Gauge data:", gauge.data)
print("Gauge shape:", gauge.shape)
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
params[define._GRID_T_] = 2
params[define._GRID_Z_] = 1
params[define._GRID_Y_] = 1
params[define._GRID_X_] = 4
print("Params:", params)
def xxxtzyx2grid_xxxtzyx(input_array, params):
    print(f"Input Array Shape: {input_array.shape}")
    lat_t = params[define._LAT_T_]
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = int(params[define._LAT_X_]/define._LAT_P_)
    grid_t = params[define._GRID_T_]
    grid_z = params[define._GRID_Z_]
    grid_y = params[define._GRID_Y_]
    grid_x = params[define._GRID_X_]
    rank = params[define._NODE_RANK_]
    size = params[define._NODE_SIZE_]
    grid_index_t, grid_index_z, grid_index_y, grid_index_x = np.argwhere(np.arange(size).reshape(
        grid_t, grid_z, grid_y, grid_x) == rank)[0]
    print(
        f"Grid Index T: {grid_index_t}, Grid Index Z: {grid_index_z}, Grid Index Y: {grid_index_y}, Grid Index X: {grid_index_x}")
    grid_lat_t = lat_t//grid_t
    grid_lat_z = lat_z//grid_z
    grid_lat_y = lat_y//grid_y
    grid_lat_x = lat_x//grid_x
    dest = input_array[...,
                       grid_index_t*grid_lat_t:grid_index_t*grid_lat_t+grid_lat_t,
                       grid_index_z*grid_lat_z:grid_index_z*grid_lat_z+grid_lat_z,
                       grid_index_y*grid_lat_y:grid_index_y*grid_lat_y+grid_lat_y,
                       grid_index_x*grid_lat_x:grid_index_x*grid_lat_x+grid_lat_x]
    print(f"Dest Shape: {dest.shape}")
    return dest
def grid_xxxtzyx2hdf5_xxxtzyx(input_array, params, file_name='xxxtzyx.h5'):
    print(f"Input Array Shape: {input_array.shape}")
    dtype = input_array.dtype
    prefix_shape = input_array.shape[:-define._LAT_D_]
    lat_t = params[define._LAT_T_]
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = int(params[define._LAT_X_]/define._LAT_P_)
    grid_t = params[define._GRID_T_]
    grid_z = params[define._GRID_Z_]
    grid_y = params[define._GRID_Y_]
    grid_x = params[define._GRID_X_]
    rank = params[define._NODE_RANK_]
    size = params[define._NODE_SIZE_]
    grid_index_t, grid_index_z, grid_index_y, grid_index_x = np.argwhere(np.arange(size).reshape(
        grid_t, grid_z, grid_y, grid_x) == rank)[0]
    print(
        f"Grid Index T: {grid_index_t}, Grid Index Z: {grid_index_z}, Grid Index Y: {grid_index_y}, Grid Index X: {grid_index_x}")
    grid_lat_t = lat_t//grid_t
    grid_lat_z = lat_z//grid_z
    grid_lat_y = lat_y//grid_y
    grid_lat_x = lat_x//grid_x
    print(
        f"Grid Lat T: {grid_lat_t}, Grid Lat Z: {grid_lat_z}, Grid Lat Y: {grid_lat_y}, Grid Lat X: {grid_lat_x}")
    with h5py.File(file_name, 'w', driver='mpio', comm=define.comm) as f:
        dset = f.create_dataset('data', shape=(
            *prefix_shape, lat_t, lat_z, lat_y, lat_x), dtype=dtype)
        dset[...,
             grid_index_t*grid_lat_t:grid_index_t*grid_lat_t+grid_lat_t,
             grid_index_z*grid_lat_z:grid_index_z*grid_lat_z+grid_lat_z,
             grid_index_y*grid_lat_y:grid_index_y*grid_lat_y+grid_lat_y,
             grid_index_x*grid_lat_x:grid_index_x*grid_lat_x+grid_lat_x] = input_array.get()
        print(f"Dest Shape: {dset.shape}")
        print(f"Rank {rank}: Data is saved to {file_name}")
def hdf5_xxxtzyx2grid_xxxtzyx(params, file_name='xxxtzyx.h5'):
    with h5py.File('parallel_example.h5', 'r', driver='mpio', comm=define.comm) as f:
        lat_t = params[define._LAT_T_]
        lat_z = params[define._LAT_Z_]
        lat_y = params[define._LAT_Y_]
        lat_x = int(params[define._LAT_X_]/define._LAT_P_)
        grid_t = params[define._GRID_T_]
        grid_z = params[define._GRID_Z_]
        grid_y = params[define._GRID_Y_]
        grid_x = params[define._GRID_X_]
        rank = params[define._NODE_RANK_]
        size = params[define._NODE_SIZE_]
        grid_index_t, grid_index_z, grid_index_y, grid_index_x = np.argwhere(np.arange(size).reshape(
            grid_t, grid_z, grid_y, grid_x) == rank)[0]
        print(
            f"Grid Index T: {grid_index_t}, Grid Index Z: {grid_index_z}, Grid Index Y: {grid_index_y}, Grid Index X: {grid_index_x}")
        grid_lat_t = lat_t//grid_t
        grid_lat_z = lat_z//grid_z
        grid_lat_y = lat_y//grid_y
        grid_lat_x = lat_x//grid_x
        print(
            f"Grid Lat T: {grid_lat_t}, Grid Lat Z: {grid_lat_z}, Grid Lat Y: {grid_lat_y}, Grid Lat X: {grid_lat_x}")
        all_dset = f['data']
        print(f"All Dset Shape: {all_dset.shape}")
        dtype = all_dset.dtype
        prefix_shape = all_dset.shape[:-define._LAT_D_]
        dest = all_dset[...,
                        grid_index_t*grid_lat_t:grid_index_t*grid_lat_t+grid_lat_t,
                        grid_index_z*grid_lat_z:grid_index_z*grid_lat_z+grid_lat_z,
                        grid_index_y*grid_lat_y:grid_index_y*grid_lat_y+grid_lat_y,
                        grid_index_x*grid_lat_x:grid_index_x*grid_lat_x+grid_lat_x]
        print(f"Dest Shape: {dest.shape}")
        return cp.asarray(dest)
# %%
# %%
# %%
print("params[define._NODE_RANK_]", params[define._NODE_RANK_])
print("params[define._NODE_SIZE_]", params[define._NODE_SIZE_])
_ = xxxtzyx2grid_xxxtzyx(gauge, params)
# %%
grid_xxxtzyx2hdf5_xxxtzyx(_, params)
# %%
# qcu.applyEndQcu(set_ptrs, params)
