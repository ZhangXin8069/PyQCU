#ifndef _GAUSS_GAUGE_H
#define _GAUSS_GAUGE_H
namespace qcu
{
  template <typename T>
  __global__ void give_random_stzyx(void *device_random_stzyx, unsigned long seed);
  template <typename T>
  __global__ void _make_gauss_gauge(void *device_U, void *device_random_stzyx, void *device_params, T sigma);
  template <typename T>
  make_gauss_gauge(void *device_U, void *set_ptr);
}
#endif