#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
void applyCloverDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params)
{
  void *fermion_out = (void *)_fermion_out;
  void *fermion_in = (void *)_fermion_in;
  void *gauge = (void *)_gauge;
  void *set_ptrs = (void *)_set_ptrs;
  void *params = (void *)_params;
  int set_index = static_cast<int *>(params)[_SET_INDEX_];
  LatticeSet<T> *set_ptr = static_cast<LatticeSet<T> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_clover_dslash
  // dptzyxcc2ccdptzyx<T>(gauge, &_set);
  // tzyxsc2sctzyx<T>(fermion_in, &_set);
  // tzyxsc2sctzyx<T>(fermion_out, &_set);
  LatticeWilsonDslash<T> _wilson_dslash;
  LatticeCloverDslash<T> _clover_dslash;
  _wilson_dslash.give(set_ptr);
  _clover_dslash.give(set_ptr);
  _clover_dslash.init();
  {
    // wilson dslash
    _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
  }
  {
    // make clover
    _clover_dslash.make(gauge);
  }
  {
    // inverse clover
    _clover_dslash.inverse();
  }
  {
    // give clover
    _clover_dslash.give(fermion_out);
  }
  // ccdptzyx2dptzyxcc<T>(gauge, &_set);
  // sctzyx2tzyxsc<T>(fermion_in, &_set);
  // sctzyx2tzyxsc<T>(fermion_out, &_set);
  _clover_dslash.end();
}