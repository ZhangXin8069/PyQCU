import pyqcu.define as define
import cupy as cp


def array2xxx(array):
    return array.reshape(array.size)


def gauge2ccdptzyx(gauge, params):
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
    _U = cp.array([0.0+0.0j]*9, dtype=cp.complex64)
    _U[6] = (U[1] * U[5] - U[2] * U[4]).conj()
    _U[7] = (U[2] * U[3] - U[0] * U[5]).conj()
    _U[8] = (U[0] * U[4] - U[1] * U[3]).conj()
    print("U:", U)
    print("_U:", _U)
    print("Gauge:", gauge.size)
    return dest


def gauge2dptzyxcc(gauge, params):
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
    _U = cp.array([0.0+0.0j]*9, dtype=cp.complex64)
    _U[6] = (U[1] * U[5] - U[2] * U[4]).conj()
    _U[7] = (U[2] * U[3] - U[0] * U[5]).conj()
    _U[8] = (U[0] * U[4] - U[1] * U[3]).conj()
    print("U:", U)
    print("_U:", _U)
    print("Gauge:", gauge.size)
    return dest


def gauge2tzyxdcc(gauge, params):
    lat_t = params[define._LAT_T_]
    lat_z = params[define._LAT_Z_]
    lat_y = params[define._LAT_Y_]
    lat_x = params[define._LAT_X_]
    lat_d = define._LAT_D_
    lat_c = define._LAT_C_
    dest_shape = (lat_t, lat_z, lat_y, lat_x, lat_d, lat_c, lat_c)
    dest = gauge.reshape(dest_shape)
    U = dest[0, 0, 0, 0, 0, :, :].reshape(define._LAT_C_ * define._LAT_C_)
    _U = cp.array([0.0+0.0j]*9, dtype=cp.complex64)
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


def split_vector_by_parity(input_array):
    shape = input_array.shape
    prefix_shape = shape[:-4]
    t, z, y, x = shape[-4:]
    indices = cp.indices((t, z, y, x))
    sums = indices[0] + indices[1] + indices[2] + indices[3]
    even_mask = (sums % 2 == 0)
    odd_mask = ~even_mask
    even_part = input_array[..., even_mask].reshape(
        *prefix_shape, t, z, y, x//2)
    odd_part = input_array[..., odd_mask].reshape(*prefix_shape, t, z, y, x//2)

    print(f"Reshaped Even Elements Shape: {even_part.shape}")
    print(f"Reshaped Odd Elements Shape: {odd_part.shape}")
    return even_part, odd_part


def reverse_split_vector(even_part, odd_part):
    shape = even_part.shape
    prefix_shape = shape[:-4]
    t, z, y, x = shape[-4:]
    x *= 2
    indices = cp.indices((t, z, y, x))
    sums = indices[0] + indices[1] + indices[2] + indices[3]
    even_mask = (sums % 2 == 0)
    odd_mask = ~even_mask
    restored_array = cp.zeros((*prefix_shape, t, z, y, x))
    restored_array[..., even_mask] = even_part.reshape((*prefix_shape, -1))
    restored_array[..., odd_mask] = odd_part.reshape((*prefix_shape, -1))

    print(f"Restored Array Shape: {restored_array.shape}")
    return restored_array
