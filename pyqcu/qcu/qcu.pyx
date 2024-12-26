cimport qcu
import numpy
cimport numpy
import ctypes
def testWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    c_void_p_fermion_out = ctypes.c_void_p(_fermion_out)
    c_void_p_fermion_in = ctypes.c_void_p(_fermion_in)
    c_void_p_gauge = ctypes.c_void_p(_gauge)
    c_void_p_params = ctypes.c_void_p(_params)
    c_void_p_argv = ctypes.c_void_p(_argv)
    fermion_out = <void*>c_void_p_fermion_out
    fermion_in = <void*>c_void_p_fermion_in
    gauge = <void*>c_void_p_gauge
    params = <void*>c_void_p_params
    argv = <void*>c_void_p_argv
    qcu.testWilsonDslashQcu(fermion_out, fermion_in, gauge, params, argv)
def applyWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    c_void_p_fermion_out = ctypes.c_void_p(_fermion_out)
    c_void_p_fermion_in = ctypes.c_void_p(_fermion_in)
    c_void_p_gauge = ctypes.c_void_p(_gauge)
    c_void_p_params = ctypes.c_void_p(_params)
    c_void_p_argv = ctypes.c_void_p(_argv)
    fermion_out = <void*>c_void_p_fermion_out
    fermion_in = <void*>c_void_p_fermion_in
    gauge = <void*>c_void_p_gauge
    params = <void*>c_void_p_params
    argv = <void*>c_void_p_argv
    qcu.applyWilsonDslashQcu(fermion_out, fermion_in, gauge, params, argv)
def testCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    c_void_p_fermion_out = ctypes.c_void_p(_fermion_out)
    c_void_p_fermion_in = ctypes.c_void_p(_fermion_in)
    c_void_p_gauge = ctypes.c_void_p(_gauge)
    c_void_p_params = ctypes.c_void_p(_params)
    c_void_p_argv = ctypes.c_void_p(_argv)
    fermion_out = <void*>c_void_p_fermion_out
    fermion_in = <void*>c_void_p_fermion_in
    gauge = <void*>c_void_p_gauge
    params = <void*>c_void_p_params
    argv = <void*>c_void_p_argv
    qcu.testCloverDslashQcu(fermion_out, fermion_in, gauge, params, argv)
def applyCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    c_void_p_fermion_out = ctypes.c_void_p(_fermion_out)
    c_void_p_fermion_in = ctypes.c_void_p(_fermion_in)
    c_void_p_gauge = ctypes.c_void_p(_gauge)
    c_void_p_params = ctypes.c_void_p(_params)
    c_void_p_argv = ctypes.c_void_p(_argv)
    fermion_out = <void*>c_void_p_fermion_out
    fermion_in = <void*>c_void_p_fermion_in
    gauge = <void*>c_void_p_gauge
    params = <void*>c_void_p_params
    argv = <void*>c_void_p_argv
    qcu.applyCloverDslashQcu(fermion_out, fermion_in, gauge, params, argv)
def applyBistabCgQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    c_void_p_fermion_out = ctypes.c_void_p(_fermion_out)
    c_void_p_fermion_in = ctypes.c_void_p(_fermion_in)
    c_void_p_gauge = ctypes.c_void_p(_gauge)
    c_void_p_params = ctypes.c_void_p(_params)
    c_void_p_argv = ctypes.c_void_p(_argv)
    fermion_out = <void*>c_void_p_fermion_out
    fermion_in = <void*>c_void_p_fermion_in
    gauge = <void*>c_void_p_gauge
    params = <void*>c_void_p_params
    argv = <void*>c_void_p_argv
    qcu.applyBistabCgQcu(fermion_out, fermion_in, gauge, params, argv)
def applyCgQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    c_void_p_fermion_out = ctypes.c_void_p(_fermion_out)
    c_void_p_fermion_in = ctypes.c_void_p(_fermion_in)
    c_void_p_gauge = ctypes.c_void_p(_gauge)
    c_void_p_params = ctypes.c_void_p(_params)
    c_void_p_argv = ctypes.c_void_p(_argv)
    fermion_out = <void*>c_void_p_fermion_out
    fermion_in = <void*>c_void_p_fermion_in
    gauge = <void*>c_void_p_gauge
    params = <void*>c_void_p_params
    argv = <void*>c_void_p_argv
    qcu.applyCgQcu(fermion_out, fermion_in, gauge, params, argv)
def applyGmresIrQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    c_void_p_fermion_out = ctypes.c_void_p(_fermion_out)
    c_void_p_fermion_in = ctypes.c_void_p(_fermion_in)
    c_void_p_gauge = ctypes.c_void_p(_gauge)
    c_void_p_params = ctypes.c_void_p(_params)
    c_void_p_argv = ctypes.c_void_p(_argv)
    fermion_out = <void*>c_void_p_fermion_out
    fermion_in = <void*>c_void_p_fermion_in
    gauge = <void*>c_void_p_gauge
    params = <void*>c_void_p_params
    argv = <void*>c_void_p_argv
    qcu.applyGmresIrQcu(fermion_out, fermion_in, gauge, params, argv)