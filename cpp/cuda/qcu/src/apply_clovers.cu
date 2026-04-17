#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
void applyCloversQcu(long long _clover, long long _clover_inv, long long _gauge, long long _set_ptrs, long long _params)
{
  cudaDeviceSynchronize();
  void *clover = (void *)_clover;
  void *clover_inv = (void *)_clover_inv;
  void *gauge = (void *)_gauge;
  void *set_ptrs = (void *)_set_ptrs;
  void *params = (void *)_params;
  int set_index = static_cast<int *>(params)[_SET_INDEX_];
  int data_type = static_cast<int *>(params)[_DATA_TYPE_];
  if (data_type == _LAT_C64_)
  {
    LatticeSet<float> *set_ptr = static_cast<LatticeSet<float> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_clover_dslash
    // dptzyxcc2ccdptzyx<float>(gauge, &_set);
    LatticeCloverDslash<float> _clover_dslash;
    LatticeCloverDslash<float> _clover_dslash_inv;
    _clover_dslash.give(set_ptr);
    _clover_dslash.init(clover);
    _clover_dslash_inv.give(set_ptr);
    _clover_dslash_inv.init(clover_inv);
    if (set_ptr->host_params[_VERBOSE_])
    {
      {
        // make clover
        _clover_dslash.make_test(gauge);
      }
      {
        // make clover
        _clover_dslash_inv.make_test(gauge);
      }
      {
        // inverse clover
        _clover_dslash_inv.inverse_test();
      }
    }
    else
    {
      {
        // make clover
        _clover_dslash.make(gauge);
      }
      {
        // make clover
        _clover_dslash_inv.make(gauge);
      }
      {
        // inverse clover
        _clover_dslash_inv.inverse();
      }
    }
    // ccdptzyx2dptzyxcc<float>(gauge, &_set);
    _clover_dslash.end();
    _clover_dslash_inv.end();
  }
  else if (data_type == _LAT_C128_)
  {
    LatticeSet<double> *set_ptr = static_cast<LatticeSet<double> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_clover_dslash
    // dptzyxcc2ccdptzyx<double>(gauge, &_set);
    LatticeCloverDslash<double> _clover_dslash;
    LatticeCloverDslash<double> _clover_dslash_inv;
    _clover_dslash.give(set_ptr);
    _clover_dslash.init(clover);
    _clover_dslash_inv.give(set_ptr);
    _clover_dslash_inv.init(clover_inv);
    if (set_ptr->host_params[_VERBOSE_])
    {
      {
        // make clover
        _clover_dslash.make_test(gauge);
      }
      {
        // make clover
        _clover_dslash_inv.make_test(gauge);
      }
      {
        // inverse clover
        _clover_dslash_inv.inverse_test();
      }
    }
    else
    {
      {
        // make clover
        _clover_dslash.make(gauge);
      }
      {
        // make clover
        _clover_dslash_inv.make(gauge);
      }
      {
        // inverse clover
        _clover_dslash_inv.inverse();
      }
    }
    // ccdptzyx2dptzyxcc<double>(gauge, &_set);
    _clover_dslash.end();
    _clover_dslash_inv.end();
  }
  else
  {
    printf("data_type error\n");
  }
  cudaDeviceSynchronize();
}