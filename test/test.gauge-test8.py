import cupy as cp
from pyqcu.cuda import define
from pyqcu.cuda import io
from pyqcu.cuda import qcu
from pyqcu.cuda.tmp_set import params, argv, set_ptrs
#############################
print("Remmber to delete _*-test8.h5!!!")
gauge = io.get_or_give_gauge(params, "_gauge-test8.h5")
fermion_in = io.get_or_give_fermion_in(params, "_fermion_in-test8.h5")
wilson_fermion_out = io.get_or_give_fermion_out(params, "_wilson_fermion_out-test8.h5")
clover_fermion_out = io.get_or_give_fermion_out(params, "_clover_fermion_out-test8.h5")
#############################
gauge = cp.zeros_like(gauge)
fermion_in = cp.ones_like(fermion_in)
wilson_fermion_out = cp.zeros_like(fermion_in)
clover_fermion_out = cp.zeros_like(fermion_in)
#############################
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyGaussGaugeQcu(gauge, set_ptrs, params)
qcu.applyEndQcu(set_ptrs, params)
#############################
io.grid_xxxtzyx2hdf5_xxxtzyx(gauge, params, "_gauge-test8.h5")
io.grid_xxxtzyx2hdf5_xxxtzyx(fermion_in, params, "_fermion_in-test8.h5")
io.grid_xxxtzyx2hdf5_xxxtzyx(wilson_fermion_out, params, "_wilson_fermion_out-test8.h5")
io.grid_xxxtzyx2hdf5_xxxtzyx(clover_fermion_out, params, "_clover_fermion_out-test8.h5")
#############################
