import pyqcu.define as define
import numpy as np
import cupy as cp
import h5py
def array2xxx(array):
    return array.reshape(array.size)
def gauge2ccdptzyx(gauge, params):
    dtype = gauge.dtype
    lat_t = params[define._LAT_T_]
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = int(params[define._LAT_X_]/define._LAT_P_)
    lat_d = define._LAT_D_
    lat_p = define._LAT_P_
    lat_c = define._LAT_C_
    dest_shape = (lat_c, lat_c, lat_d, lat_p, lat_t, lat_z, lat_y, lat_x)
    dest = gauge.reshape(dest_shape)
    U = dest[:, :, 0, 0, 0, 0, 0, 0].reshape(define._LAT_C_ * define._LAT_C_)
    _U = cp.array([0.0+0.0j]*9, dtype=dtype)
    _U[6] = (U[1] * U[5] - U[2] * U[4]).conj()
    _U[7] = (U[2] * U[3] - U[0] * U[5]).conj()
    _U[8] = (U[0] * U[4] - U[1] * U[3]).conj()
    print("U:", U)
    print("_U:", _U)
    print("Gauge:", gauge.size)
    return dest
def gauge2dptzyxcc(gauge, params):
    dtype = gauge.dtype
    lat_t = params[define._LAT_T_]
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = int(params[define._LAT_X_]/define._LAT_P_)
    lat_d = define._LAT_D_
    lat_p = define._LAT_P_
    lat_c = define._LAT_C_
    dest_shape = (lat_d, lat_p, lat_t, lat_z, lat_y, lat_x, lat_c, lat_c)
    dest = gauge.reshape(dest_shape)
    U = dest[0, 0, 0, 0, 0, 0, :, :].reshape(define._LAT_C_ * define._LAT_C_)
    _U = cp.array([0.0+0.0j]*9, dtype=dtype)
    _U[6] = (U[1] * U[5] - U[2] * U[4]).conj()
    _U[7] = (U[2] * U[3] - U[0] * U[5]).conj()
    _U[8] = (U[0] * U[4] - U[1] * U[3]).conj()
    print("U:", U)
    print("_U:", _U)
    print("Gauge:", gauge.size)
    return dest
def gauge2tzyxdcc(gauge, params):
    dtype = gauge.dtype
    lat_t = params[define._LAT_T_]
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = params[define._LAT_X_]
    lat_d = define._LAT_D_
    lat_c = define._LAT_C_
    dest_shape = (lat_t, lat_z, lat_y, lat_x, lat_d, lat_c, lat_c)
    dest = gauge.reshape(dest_shape)
    U = dest[0, 0, 0, 0, 0, :, :].reshape(define._LAT_C_ * define._LAT_C_)
    _U = cp.array([0.0+0.0j]*9, dtype=dtype)
    _U[6] = (U[1] * U[5] - U[2] * U[4]).conj()
    _U[7] = (U[2] * U[3] - U[0] * U[5]).conj()
    _U[8] = (U[0] * U[4] - U[1] * U[3]).conj()
    print("U:", U)
    print("_U:", _U)
    print("Gauge:", gauge.size)
    return dest
def fermion2sctzyx(fermion, params):
    lat_s = define._LAT_S_
    lat_c = define._LAT_C_
    lat_t = params[define._LAT_T_]
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = int(params[define._LAT_X_]/define._LAT_P_)
    dest_shape = (lat_s, lat_c, lat_t, lat_z, lat_y, lat_x)
    dest = fermion.reshape(dest_shape)
    return dest
def fermion2tzyxsc(fermion, params):
    lat_s = define._LAT_S_
    lat_c = define._LAT_C_
    lat_t = params[define._LAT_T_]
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = int(params[define._LAT_X_]/define._LAT_P_)
    dest_shape = (lat_t, lat_z, lat_y, lat_x, lat_s, lat_c)
    dest = fermion.reshape(dest_shape)
    return dest
def fermion2psctzyx(fermion, params):
    lat_p = define._LAT_P_
    lat_s = define._LAT_S_
    lat_c = define._LAT_C_
    lat_t = params[define._LAT_T_]
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = int(params[define._LAT_X_]/define._LAT_P_)
    dest_shape = (lat_p, lat_s, lat_c, lat_t, lat_z, lat_y, lat_x)
    dest = fermion.reshape(dest_shape)
    return dest
def fermion2ptzyxsc(fermion, params):
    lat_p = define._LAT_P_
    lat_s = define._LAT_S_
    lat_c = define._LAT_C_
    lat_t = params[define._LAT_T_]
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = int(params[define._LAT_X_]/define._LAT_P_)
    dest_shape = (lat_p, lat_t, lat_z, lat_y, lat_x, lat_s, lat_c)
    dest = fermion.reshape(dest_shape)
    return dest
def laplacian_gauge2ccdzyx(laplacian_gauge, params):
    lat_d = define._LAT_D_
    lat_c = define._LAT_C_
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = params[define._LAT_X_]
    dest_shape = (lat_c, lat_c, lat_d, lat_z, lat_y, lat_x)
    dest = laplacian_gauge.reshape(dest_shape)
    return dest
def laplacian_gauge2dzyxcc(laplacian_gauge, params):
    lat_d = define._LAT_D_
    lat_c = define._LAT_C_
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = params[define._LAT_X_]
    dest_shape = (lat_d, lat_z, lat_y, lat_x, lat_c, lat_c)
    dest = laplacian_gauge.reshape(dest_shape)
    return dest
def laplacian2czyx(laplacian, params):
    lat_c = define._LAT_C_
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = params[define._LAT_X_]
    dest_shape = (lat_c, lat_z, lat_y, lat_x)
    dest = laplacian.reshape(dest_shape)
    return dest
def laplacian2zyxc(laplacian, params):
    lat_c = define._LAT_C_
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = params[define._LAT_X_]
    dest_shape = (lat_z, lat_y, lat_x, lat_c)
    dest = laplacian.reshape(dest_shape)
    return dest
def eigenvectors2esctzyx(eigenvectors, params):
    lat_e = params[define._LAT_E_]
    lat_s = define._LAT_S_
    lat_c = define._LAT_C_
    lat_t = params[define._LAT_T_]
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = int(params[define._LAT_X_]/define._LAT_P_)
    dest_shape = (lat_e, lat_s, lat_c, lat_t, lat_z, lat_y, lat_x)
    dest = eigenvectors.reshape(dest_shape)
    return dest
def ccdptzyx2dptzyxcc(gauge):
    dest = gauge.transpose(2, 3, 4, 5, 6, 7, 0, 1)
    return dest
def dptzyxcc2ccdptzyx(gauge):
    dest = gauge.transpose(6, 7, 0, 1, 2, 3, 4, 5)
    return dest
def sctzyx2tzyxsc(fermion):
    dest = fermion.transpose(2, 3, 4, 5, 0, 1)
    return dest
def tzyxsc2sctzyx(fermion):
    dest = fermion.transpose(4, 5, 0, 1, 2, 3)
    return dest
def psctzyx2ptzyxsc(fermion):
    dest = fermion.transpose(0, 3, 4, 5, 6, 1, 2)
    return dest
def ptzyxsc2psctzyx(fermion):
    dest = fermion.transpose(0, 5, 6, 1, 2, 3, 4)
    return dest
def ccdzyx2dzyxcc(laplacian_gauge):
    dest = laplacian_gauge.transpose(2, 3, 4, 5, 0, 1)
    return dest
def dzyxcc2ccdzyx(laplacian_gauge):
    dest = laplacian_gauge.transpose(4, 5, 0, 1, 2, 3)
    return dest
def czyx2zyxc(laplacian):
    dest = laplacian.transpose(1, 2, 3, 0)
    return dest
def zyxc2czyx(laplacian):
    dest = laplacian.transpose(3, 0, 1, 2)
    return dest
def xxxtzyx2pxxxtzyx(input_array):
    shape = input_array.shape
    dtype = input_array.dtype
    prefix_shape = shape[:-define._LAT_D_]
    t, z, y, x = shape[-define._LAT_D_:]
    indices = cp.indices((t, z, y, x))
    sums = indices[define._X_] + indices[define._Y_] + \
        indices[define._Z_] + indices[define._T_]
    even_mask = (sums % define._LAT_P_ == 0)
    odd_mask = ~even_mask
    splited_array = cp.zeros(
        (define._LAT_P_, *prefix_shape, t, z, y, x//define._LAT_P_), dtype=dtype)
    splited_array[define._EVEN_] = input_array[..., even_mask].reshape(
        *prefix_shape, t, z, y, x//define._LAT_P_)
    splited_array[define._ODD_] = input_array[..., odd_mask].reshape(
        *prefix_shape, t, z, y, x//define._LAT_P_)
    print(f"Splited Array Shape: {splited_array.shape}")
    return splited_array
def pxxxtzyx2xxxtzyx(input_array):
    shape = input_array.shape
    dtype = input_array.dtype
    prefix_shape = shape[1:-define._LAT_D_]
    t, z, y, x = shape[-define._LAT_D_:]
    x *= define._LAT_P_
    indices = cp.indices((t, z, y, x))
    sums = indices[define._X_] + indices[define._Y_] + \
        indices[define._Z_] + indices[define._T_]
    even_mask = (sums % define._LAT_P_ == 0)
    odd_mask = ~even_mask
    restored_array = cp.zeros((*prefix_shape, t, z, y, x), dtype=dtype)
    restored_array[..., even_mask] = input_array[define._EVEN_].reshape(
        (*prefix_shape, -1))
    restored_array[..., odd_mask] = input_array[define._ODD_].reshape(
        (*prefix_shape, -1))
    print(f"Restored Array Shape: {restored_array.shape}")
    return restored_array
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
        dest = f.create_dataset('data', shape=(
            *prefix_shape, lat_t, lat_z, lat_y, lat_x), dtype=dtype)
        dest[...,
             grid_index_t*grid_lat_t:grid_index_t*grid_lat_t+grid_lat_t,
             grid_index_z*grid_lat_z:grid_index_z*grid_lat_z+grid_lat_z,
             grid_index_y*grid_lat_y:grid_index_y*grid_lat_y+grid_lat_y,
             grid_index_x*grid_lat_x:grid_index_x*grid_lat_x+grid_lat_x] = input_array.get()
        print(f"Dest Shape: {dest.shape}")
        print(f"Rank {rank}: Data is saved to {file_name}")
def hdf5_xxxtzyx2grid_xxxtzyx(params, file_name='xxxtzyx.h5'):
    with h5py.File(file_name, 'r', driver='mpio', comm=define.comm) as f:
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
        all_dest = f['data']
        print(f"All Dest Shape: {all_dest.shape}")
        dest = all_dest[...,
                        grid_index_t*grid_lat_t:grid_index_t*grid_lat_t+grid_lat_t,
                        grid_index_z*grid_lat_z:grid_index_z*grid_lat_z+grid_lat_z,
                        grid_index_y*grid_lat_y:grid_index_y*grid_lat_y+grid_lat_y,
                        grid_index_x*grid_lat_x:grid_index_x*grid_lat_x+grid_lat_x]
        print(f"Dest Shape: {dest.shape}")
        return cp.asarray(dest)
def xxx2hdf5_xxx(input_array, params=None, file_name='xxx.h5'):
    print(f"Input Array Shape: {input_array.shape}")
    dtype = input_array.dtype
    shape = input_array.shape
    with h5py.File(file_name, 'w', driver='mpio', comm=define.comm) as f:
        dest = f.create_dataset('data', shape=shape, dtype=dtype)
        dest[...] = input_array.get()
        print(f"Dest Shape: {dest.shape}")
        print(f"Data is saved to {file_name}")
def hdf5_xxx2xxx(params=None, file_name='xxx.h5'):
    with h5py.File(file_name, 'r', driver='mpio', comm=define.comm) as f:
        all_dest = f['data']
        dest = all_dest[...]
        print(f"Dest Shape: {dest.shape}")
        return cp.asarray(dest)
def xxxtzyx2mg_xxxtzyx(input_array, params):
    print(f"Input Array Shape: {input_array.shape}")
    prefix_shape = input_array.shape[:-define._LAT_D_]
    lat_t = params[define._LAT_T_]
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = int(params[define._LAT_X_]/define._LAT_P_)
    mg_t = params[define._MG_T_]
    mg_z = params[define._MG_Z_]
    mg_y = params[define._MG_Y_]
    mg_x = params[define._MG_X_]
    mg_lat_t = lat_t//mg_t
    mg_lat_z = lat_z//mg_z
    mg_lat_y = lat_y//mg_y
    mg_lat_x = lat_x//mg_x
    dest = input_array.reshape(
        *prefix_shape, mg_t, mg_lat_t, mg_z, mg_lat_z, mg_y, mg_lat_y, mg_x, mg_lat_x)
    print(f"Dest Shape: {dest.shape}")
    return dest
def xxx2eTZYX(input_array, params):
    print(f"Input Array Shape: {input_array.shape}")
    lat_e = params[define._LAT_E_]
    mg_t = params[define._MG_T_]
    mg_z = params[define._MG_Z_]
    mg_y = params[define._MG_Y_]
    mg_x = params[define._MG_X_]
    dest = input_array.reshape(
        lat_e, mg_t, mg_z, mg_y, mg_x)
    print(f"Dest Shape: {dest.shape}")
    return dest
def xxx2escTZYX(input_array, params):
    print(f"Input Array Shape: {input_array.shape}")
    lat_s = define._LAT_S_
    lat_c = define._LAT_C_
    lat_e = params[define._LAT_E_]
    mg_t = params[define._MG_T_]
    mg_z = params[define._MG_Z_]
    mg_y = params[define._MG_Y_]
    mg_x = params[define._MG_X_]
    dest = input_array.reshape(
        lat_e, lat_s, lat_c, mg_t, mg_z, mg_y, mg_x)
    print(f"Dest Shape: {dest.shape}")
    return dest
def xxx2scTZYX(input_array, params):
    print(f"Input Array Shape: {input_array.shape}")
    lat_s = define._LAT_S_
    lat_c = define._LAT_C_
    mg_t = params[define._MG_T_]
    mg_z = params[define._MG_Z_]
    mg_y = params[define._MG_Y_]
    mg_x = params[define._MG_X_]
    dest = input_array.reshape(
        lat_s, lat_c, mg_t, mg_z, mg_y, mg_x)
    print(f"Dest Shape: {dest.shape}")
    return dest
