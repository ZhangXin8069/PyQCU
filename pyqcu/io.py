import define
import cupy as cp


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
    U = gauge[:, :, 0, 0, 0, 0, 0, 0].reshape(define._LAT_C_ * define._LAT_C_)
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
