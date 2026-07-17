#include "../include/qcu.h"
#include "../python/pyqcu.h"
#pragma optimize(5)
using namespace qcu;
void applyMultigridRestrictQcu(long long _coarse_out, long long _fine_in,
                                long long _null_vecs, long long _set_ptrs,
                                long long _params) {
  cudaDeviceSynchronize();
  void *coarse_out = (void *)_coarse_out;
  void *fine_in = (void *)_fine_in;
  void *null_vecs = (void *)_null_vecs;
  void *set_ptrs = (void *)_set_ptrs;
  int *params = (int *)_params;
  int set_index = params[_SET_INDEX_];
  int data_type = params[_DATA_TYPE_];
  int E = params[_MG_LEVEL1_E_];
  int e = params[_MG_NUM_LEVEL_]; // repurposed: fine DOF for this restrict call
  int Xf = params[_LAT_X_];
  int Yf = params[_LAT_Y_];
  int Zf = params[_LAT_Z_];
  int Tf = params[_LAT_T_];
  int Xc = params[_MG_LEVEL1_X_];
  int Yc = params[_MG_LEVEL1_Y_];
  int Zc = params[_MG_LEVEL1_Z_];
  int Tc = params[_MG_LEVEL1_T_];
  if (data_type == _LAT_C64_) {
    LatticeSet<float> *set_ptr =
        static_cast<LatticeSet<float> *>(
            (void *)(static_cast<long long *>(set_ptrs)[set_index]));
    LatticeMultigridRestrict<float> _restrict;
    _restrict.give(set_ptr);
    _restrict.run(coarse_out, fine_in, null_vecs, E, e, Xf, Yf, Zf, Tf, Xc, Yc,
                  Zc, Tc);
  } else if (data_type == _LAT_C128_) {
    LatticeSet<double> *set_ptr =
        static_cast<LatticeSet<double> *>(
            (void *)(static_cast<long long *>(set_ptrs)[set_index]));
    LatticeMultigridRestrict<double> _restrict;
    _restrict.give(set_ptr);
    _restrict.run(coarse_out, fine_in, null_vecs, E, e, Xf, Yf, Zf, Tf, Xc, Yc,
                  Zc, Tc);
  } else {
    printf("data_type error in applyMultigridRestrictQcu\n");
  }
  cudaDeviceSynchronize();
}
void applyMultigridProLongQcu(long long _fine_out, long long _coarse_in,
                               long long _null_vecs, long long _set_ptrs,
                               long long _params) {
  cudaDeviceSynchronize();
  void *fine_out = (void *)_fine_out;
  void *coarse_in = (void *)_coarse_in;
  void *null_vecs = (void *)_null_vecs;
  void *set_ptrs = (void *)_set_ptrs;
  int *params = (int *)_params;
  int set_index = params[_SET_INDEX_];
  int data_type = params[_DATA_TYPE_];
  int E = params[_MG_LEVEL1_E_];
  int e = params[_MG_NUM_LEVEL_]; // repurposed: fine DOF for this prolong call
  int Xf = params[_LAT_X_];
  int Yf = params[_LAT_Y_];
  int Zf = params[_LAT_Z_];
  int Tf = params[_LAT_T_];
  int Xc = params[_MG_LEVEL1_X_];
  int Yc = params[_MG_LEVEL1_Y_];
  int Zc = params[_MG_LEVEL1_Z_];
  int Tc = params[_MG_LEVEL1_T_];
  if (data_type == _LAT_C64_) {
    LatticeSet<float> *set_ptr =
        static_cast<LatticeSet<float> *>(
            (void *)(static_cast<long long *>(set_ptrs)[set_index]));
    LatticeMultigridProLong<float> _prolong;
    _prolong.give(set_ptr);
    _prolong.run(fine_out, coarse_in, null_vecs, E, e, Xf, Yf, Zf, Tf, Xc, Yc,
                 Zc, Tc);
  } else if (data_type == _LAT_C128_) {
    LatticeSet<double> *set_ptr =
        static_cast<LatticeSet<double> *>(
            (void *)(static_cast<long long *>(set_ptrs)[set_index]));
    LatticeMultigridProLong<double> _prolong;
    _prolong.give(set_ptr);
    _prolong.run(fine_out, coarse_in, null_vecs, E, e, Xf, Yf, Zf, Tf, Xc, Yc,
                 Zc, Tc);
  } else {
    printf("data_type error in applyMultigridProLongQcu\n");
  }
  cudaDeviceSynchronize();
}
void applyMultigridCoarseDslashQcu(long long _fermion_out,
                                    long long _fermion_in, long long _hopping,
                                    long long _sitting, long long _set_ptrs,
                                    long long _params) {
  cudaDeviceSynchronize();
  void *fermion_out = (void *)_fermion_out;
  void *fermion_in = (void *)_fermion_in;
  void *hopping = (void *)_hopping;
  void *sitting = (void *)_sitting;
  void *set_ptrs = (void *)_set_ptrs;
  int *params = (int *)_params;
  int set_index = params[_SET_INDEX_];
  int data_type = params[_DATA_TYPE_];
  int E = params[_MG_NUM_LEVEL_];
  int X = params[_MG_LEVEL1_X_];
  int Y = params[_MG_LEVEL1_Y_];
  int Z = params[_MG_LEVEL1_Z_];
  int T = params[_MG_LEVEL1_T_];
  if (data_type == _LAT_C64_) {
    LatticeSet<float> *set_ptr =
        static_cast<LatticeSet<float> *>(
            (void *)(static_cast<long long *>(set_ptrs)[set_index]));
    LatticeMultigridCoarseDslash<float> _coarse_dslash;
    _coarse_dslash.give(set_ptr);
    _coarse_dslash.run(fermion_out, fermion_in, hopping, sitting, E, X, Y, Z, T);
  } else if (data_type == _LAT_C128_) {
    LatticeSet<double> *set_ptr =
        static_cast<LatticeSet<double> *>(
            (void *)(static_cast<long long *>(set_ptrs)[set_index]));
    LatticeMultigridCoarseDslash<double> _coarse_dslash;
    _coarse_dslash.give(set_ptr);
    _coarse_dslash.run(fermion_out, fermion_in, hopping, sitting, E, X, Y, Z, T);
  } else {
    printf("data_type error in applyMultigridCoarseDslashQcu\n");
  }
  cudaDeviceSynchronize();
}
