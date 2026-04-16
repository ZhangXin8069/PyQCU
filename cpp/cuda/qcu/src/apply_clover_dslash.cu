#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
void applyCloverDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params)
{
  cudaDeviceSynchronize();
  void *fermion_out = (void *)_fermion_out;
  void *fermion_in = (void *)_fermion_in;
  void *gauge = (void *)_gauge;
  void *set_ptrs = (void *)_set_ptrs;
  void *params = (void *)_params;
  int set_index = static_cast<int *>(params)[_SET_INDEX_];
  int data_type = static_cast<int *>(params)[_DATA_TYPE_];
  if (data_type == _LAT_C64_)
  {
    LatticeSet<float> *set_ptr = static_cast<LatticeSet<float> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_clover_dslash
    // dptzyxcc2ccdptzyx<float>(gauge, &_set);
    // tzyxsc2sctzyx<float>(fermion_in, &_set);
    // tzyxsc2sctzyx<float>(fermion_out, &_set);
    LatticeWilsonDslash<float> _wilson_dslash;
    LatticeCloverDslash<float> _clover_dslash;
    _wilson_dslash.give(set_ptr);
    _clover_dslash.give(set_ptr);
    _clover_dslash.init();
    if (set_ptr->host_params[_VERBOSE_])
    {
      {
        // wilson dslash
        _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
      }
      {
        // make clover
        _clover_dslash.make_test(gauge);
      }
      {
        // inverse clover
        _clover_dslash.inverse_test();
      }
      {
        // give clover
        _clover_dslash.give_test(fermion_out);
      }
    }
    else
    {
      {
        // wilson dslash
        _wilson_dslash.run(fermion_out, fermion_in, gauge);
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
    }
    // ccdptzyx2dptzyxcc<float>(gauge, &_set);
    // sctzyx2tzyxsc<float>(fermion_in, &_set);
    // sctzyx2tzyxsc<float>(fermion_out, &_set);
    _clover_dslash.end();
  }
  else if (data_type == _LAT_C128_)
  {
    LatticeSet<double> *set_ptr = static_cast<LatticeSet<double> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_clover_dslash
    // dptzyxcc2ccdptzyx<double>(gauge, &_set);
    // tzyxsc2sctzyx<double>(fermion_in, &_set);
    // tzyxsc2sctzyx<double>(fermion_out, &_set);
    LatticeWilsonDslash<double> _wilson_dslash;
    LatticeCloverDslash<double> _clover_dslash;
    _wilson_dslash.give(set_ptr);
    _clover_dslash.give(set_ptr);
    _clover_dslash.init();
    if (set_ptr->host_params[_VERBOSE_])
    {
      {
        // wilson dslash
        _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
      }
      {
        // make clover
        _clover_dslash.make_test(gauge);
      }
      {
        // inverse clover
        _clover_dslash.inverse_test();
      }
      {
        // give clover
        _clover_dslash.give_test(fermion_out);
      }
    }
    else
    {
      {
        // wilson dslash
        _wilson_dslash.run(fermion_out, fermion_in, gauge);
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
    }
    // ccdptzyx2dptzyxcc<double>(gauge, &_set);
    // sctzyx2tzyxsc<double>(fermion_in, &_set);
    // sctzyx2tzyxsc<double>(fermion_out, &_set);
    _clover_dslash.end();
  }
  else
  {
    printf("data_type error\n");
  }
  cudaDeviceSynchronize();
}