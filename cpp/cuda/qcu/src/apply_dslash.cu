#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
void applyDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _clover, long long _set_ptrs, long long _params)
{
  cudaDeviceSynchronize();
  void *fermion_out = (void *)_fermion_out;
  void *fermion_in = (void *)_fermion_in;
  void *gauge = (void *)_gauge;
  void *clover = (void *)_clover;
  void *set_ptrs = (void *)_set_ptrs;
  void *params = (void *)_params;
  int set_index = static_cast<int *>(params)[_SET_INDEX_];
  int data_type = static_cast<int *>(params)[_DATA_TYPE_];
  if (data_type == _LAT_C64_)
  {
    LatticeSet<float> *set_ptr = static_cast<LatticeSet<float> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_clover_dslash
    // dptzyxcc2ccdptzyx<float>(gauge, &_set);
    LatticeWilsonDslash<float> _wilson_dslash;
    _wilson_dslash.give(set_ptr);
    {
      // wilson dslash
      if (set_ptr->host_params[_VERBOSE_])
      {
        _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
      }
      else
      {
        _wilson_dslash.run(fermion_out, fermion_in, gauge);
      }
    }
    {
      // clover dslash
      if (set_ptr->host_params[_VERBOSE_])
      {
        printf("clover:%p\n", clover);
        printf("long long clover:%lld\n", (long long)clover);
      }
      if ((long long)clover)
      {
        LatticeCloverDslash<float> _clover_dslash;
        _clover_dslash.give(set_ptr);
        _clover_dslash.init(clover);
        // give clover
        _clover_dslash.give(fermion_out);
        _clover_dslash.end();
      }
    }
    // ccdptzyx2dptzyxcc<float>(gauge, &_set);
  }
  else if (data_type == _LAT_C128_)
  {
    LatticeSet<double> *set_ptr = static_cast<LatticeSet<double> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_clover_dslash
    // dptzyxcc2ccdptzyx<double>(gauge, &_set);
    LatticeWilsonDslash<double> _wilson_dslash;
    _wilson_dslash.give(set_ptr);
    {
      // wilson dslash
      if (set_ptr->host_params[_VERBOSE_])
      {
        _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
      }
      else
      {
        _wilson_dslash.run(fermion_out, fermion_in, gauge);
      }
    }
    {
      // clover dslash
      if (set_ptr->host_params[_VERBOSE_])
      {
        printf("clover:%p\n", clover);
        printf("long long clover:%lld\n", (long long)clover);
      }
      if ((long long)clover)
      {
        LatticeCloverDslash<double> _clover_dslash;
        _clover_dslash.give(set_ptr);
        _clover_dslash.init(clover);
        // give clover
        _clover_dslash.give(fermion_out);
        _clover_dslash.end();
      }
    }
    // ccdptzyx2dptzyxcc<double>(gauge, &_set);
  }
  else
  {
    printf("data_type error\n");
  }
  cudaDeviceSynchronize();
}