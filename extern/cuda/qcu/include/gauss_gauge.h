#ifndef _GAUSS_GAUGE_H
#define _GAUSS_GAUGE_H
namespace qcu
{
  template <typename T>
  __global__ void make_gauss_gauge(void *device_U, void *device_params, T sigma);
}
#endif