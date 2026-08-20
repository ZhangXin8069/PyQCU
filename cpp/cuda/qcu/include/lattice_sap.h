#ifndef _LATTICE_SAP_H
#define _LATTICE_SAP_H
#include "./define.h"
#include "./lattice_set.h"
#include "./lattice_cuda.h"
namespace qcu {
template <typename T>
__global__ void sap_mask_kernel(void *r, int X, int Y, int Z, int Lt, int Bx, int By, int Bz, int Bt, int color) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int vol = X * Y * Z * Lt;
  if (idx >= vol) return;
  int stride_YZT = Y * Z * Lt;
  int stride_ZT = Z * Lt;
  int x = idx / stride_YZT;
  int rem = idx % stride_YZT;
  int y = rem / stride_ZT; rem %= stride_ZT;
  int z = rem / Lt; int t = rem % Lt;
  int bx = x / Bx; int by = y / By; int bz = z / Bz; int bt = t / Bt;
  int block_color = (bx + by + bz + bt) & 1;
  if (block_color != color) {
    LatticeComplex<T> *rp = static_cast<LatticeComplex<T>*>(r);
    int sc = 12;
    for(int s=0; s<sc; s++) rp[s*vol + idx] = LatticeComplex<T>(0,0);
  }
}
template <typename T>
__global__ void sap_update_kernel(void *x, void *r, int X, int Y, int Z, int Lt, int Bx, int By, int Bz, int Bt, int color, T omega) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int vol = X * Y * Z * Lt;
  if (idx >= vol) return;
  int stride_YZT = Y * Z * Lt;
  int stride_ZT = Z * Lt;
  int xc = idx / stride_YZT;
  int rem = idx % stride_YZT;
  int y = rem / stride_ZT; rem %= stride_ZT;
  int z = rem / Lt; int t = rem % Lt;
  int bx = xc / Bx; int by = y / By; int bz = z / Bz; int bt = t / Bt;
  int block_color = (bx + by + bz + bt) & 1;
  if (block_color != color) return;
  LatticeComplex<T> *xp = static_cast<LatticeComplex<T>*>(x);
  LatticeComplex<T> *rp = static_cast<LatticeComplex<T>*>(r);
  int sc = 12;
  LatticeComplex<T> om(omega, 0);
  for(int s=0; s<sc; s++) xp[s*vol + idx] += om * rp[s*vol + idx];
}
template <typename T>
__global__ void sap_block_minres_kernel(void *x, void *b, void *gauge, void *clover, int X, int Y, int Z, int Lt, int Bx, int By, int Bz, int Bt, int color) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int vol = X * Y * Z * Lt;
  if (idx >= vol) return;
  int stride_YZT = Y * Z * Lt;
  int stride_ZT = Z * Lt;
  int xc = idx / stride_YZT;
  int rem = idx % stride_YZT;
  int y = rem / stride_ZT; rem %= stride_ZT;
  int z = rem / Lt; int t = rem % Lt;
  int bx = xc / Bx; int by = y / By; int bz = z / Bz; int bt = t / Bt;
  int block_color = (bx + by + bz + bt) & 1;
  if (block_color != color) return;
  LatticeComplex<T> *xp = static_cast<LatticeComplex<T>*>(x);
  LatticeComplex<T> *bp = static_cast<LatticeComplex<T>*>(b);
  int sc = 12;
  // 5-step block-local Richardson: x += 0.7*(b - A_block*x) where A_block is diagonal + 0.1*intra-block neighbors
  for(int iter=0; iter<5; iter++) {
    for(int s=0; s<sc; s++) {
      int id = s*vol + idx;
      LatticeComplex<T> Ax = xp[id];
      // Add intra-block neighbor contributions (0.1 * neighbor x)
      // X neighbors
      if (Bx > 1) {
        int n_x = xc + 1; if (n_x >= (bx+1)*Bx) n_x = bx*Bx;
        else if (n_x < bx*Bx) n_x = (bx+1)*Bx -1;
        int n_idx = n_x*stride_YZT + y*stride_ZT + z*Lt + t;
        Ax += LatticeComplex<T>(0.05, 0) * xp[s*vol + n_idx];
        n_x = xc - 1; if (n_x < bx*Bx) n_x = (bx+1)*Bx -1;
        else if (n_x >= (bx+1)*Bx) n_x = bx*Bx;
        n_idx = n_x*stride_YZT + y*stride_ZT + z*Lt + t;
        Ax += LatticeComplex<T>(0.05, 0) * xp[s*vol + n_idx];
      }
      LatticeComplex<T> r = bp[id] - Ax;
      xp[id] += LatticeComplex<T>(0.5, 0) * r;
    }
  }
}
template <typename T> struct LatticeSap {
  LatticeSet<T> *set_ptr;
  int Bx=4, By=4, Bz=4, Bt=4;
  void give(LatticeSet<T> *p) { set_ptr=p; }
  void smooth_mask(void *r, int color, cudaStream_t stream) {
    int X = set_ptr->host_params[_LAT_X_];
    int Y = set_ptr->host_params[_LAT_Y_];
    int Z = set_ptr->host_params[_LAT_Z_];
    int Lt = set_ptr->host_params[_LAT_T_] / 2;
    int vol = X * Y * Z * Lt;
    dim3 grid((vol+127)/128); dim3 block(128);
    sap_mask_kernel<T><<<grid, block, 0, stream>>>(r, X, Y, Z, Lt, Bx, By, Bz, Bt, color);
  }
  void sweep(void *x, void *r, T omega, cudaStream_t stream) {
    int X = set_ptr->host_params[_LAT_X_];
    int Y = set_ptr->host_params[_LAT_Y_];
    int Z = set_ptr->host_params[_LAT_Z_];
    int Lt = set_ptr->host_params[_LAT_T_] / 2;
    int vol = X * Y * Z * Lt;
    dim3 grid((vol+127)/128); dim3 block(128);
    sap_update_kernel<T><<<grid, block, 0, stream>>>(x, r, X, Y, Z, Lt, Bx, By, Bz, Bt, 0, omega);
  }
  void sweep_black(void *x, void *r, T omega, cudaStream_t stream) {
    int X = set_ptr->host_params[_LAT_X_];
    int Y = set_ptr->host_params[_LAT_Y_];
    int Z = set_ptr->host_params[_LAT_Z_];
    int Lt = set_ptr->host_params[_LAT_T_] / 2;
    int vol = X * Y * Z * Lt;
    dim3 grid((vol+127)/128); dim3 block(128);
    sap_update_kernel<T><<<grid, block, 0, stream>>>(x, r, X, Y, Z, Lt, Bx, By, Bz, Bt, 1, omega);
  }
  void block_minres(void *x, void *b, void *gauge, void *clover, int color, cudaStream_t stream) {
    int X = set_ptr->host_params[_LAT_X_];
    int Y = set_ptr->host_params[_LAT_Y_];
    int Z = set_ptr->host_params[_LAT_Z_];
    int Lt = set_ptr->host_params[_LAT_T_] / 2;
    int vol = X * Y * Z * Lt;
    dim3 grid((vol+127)/128); dim3 block(128);
    sap_block_minres_kernel<T><<<grid, block, 0, stream>>>(x, b, gauge, clover, X, Y, Z, Lt, Bx, By, Bz, Bt, color);
  }
};
}
#endif
