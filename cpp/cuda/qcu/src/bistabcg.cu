#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu {
template <typename T> __global__ void bistabcg_give_1beta(void *device_vals) {
  LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
  vals[_beta_] =
      (vals[_rho_] / vals[_rho_prev_]) * (vals[_alpha_] / vals[_omega_]);
}
template <typename T>
__global__ void bistabcg_give_1beta_rp(void *device_vals) {
  LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
  vals[_beta_] =
      (vals[_rho_] / vals[_rho_prev_]) * (vals[_alpha_] / vals[_omega_]);
  vals[_rho_prev_] = vals[_rho_];
}
template <typename T>
__global__ void bistabcg_give_1rho_prev(void *device_vals) {
  LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
  vals[_rho_prev_] = vals[_rho_];
}
template <typename T> __global__ void bistabcg_give_1alpha(void *device_vals) {
  LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
  vals[_alpha_] = vals[_rho_] / vals[_tmp0_];
}
template <typename T> __global__ void bistabcg_give_1omega(void *device_vals) {
  LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
  vals[_omega_] = vals[_tmp0_] / vals[_tmp1_];
}
template <typename T> __global__ void bistabcg_give_1diff2(void *device_vals) {
  LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
  vals[_diff2_tmp_] = vals[_diff2_tmp_] / vals[_norm2_tmp_];
}
template <typename T>
__global__ void bistabcg_give_b_e(void *device_b_e, void *device_ans_e,
                                  void *device_vec0, T kappa,
                                  void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *b_e = (static_cast<LatticeComplex<T> *>(device_b_e) + idx);
  LatticeComplex<T> *ans_e =
      (static_cast<LatticeComplex<T> *>(device_ans_e) + idx);
  LatticeComplex<T> *vec0 =
      (static_cast<LatticeComplex<T> *>(device_vec0) + idx);
  int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    b_e[i] = ans_e[i] - vec0[i] * kappa; // b_e=ans_e-kappa*D_eo(ans_o)
  }
}
template <typename T>
__global__ void bistabcg_give_b_o(void *device_b_o, void *device_ans_o,
                                  void *device_vec1, T kappa,
                                  void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *b_o = (static_cast<LatticeComplex<T> *>(device_b_o) + idx);
  LatticeComplex<T> *ans_o =
      (static_cast<LatticeComplex<T> *>(device_ans_o) + idx);
  LatticeComplex<T> *vec1 =
      (static_cast<LatticeComplex<T> *>(device_vec1) + idx);
  int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    b_o[i] = ans_o[i] - vec1[i] * kappa; // b_o=ans_o-kappa*D_oe(ans_e)
  }
}
template <typename T>
__global__ void bistabcg_give_b__o(void *device_b__o, void *device_b_o,
                                   void *device_vec0, T kappa,
                                   void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *b__o =
      (static_cast<LatticeComplex<T> *>(device_b__o) + idx);
  LatticeComplex<T> *b_o = (static_cast<LatticeComplex<T> *>(device_b_o) + idx);
  LatticeComplex<T> *vec0 =
      (static_cast<LatticeComplex<T> *>(device_vec0) + idx);
  int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    b__o[i] = b_o[i] + vec0[i] * kappa; // b__o=b_o+kappa*D_oe(b_e)
  }
}
template <typename T>
__global__ void bistabcg_give_dest_o(void *device_dest_o, void *device_src_o,
                                     void *device_vec1, T kappa,
                                     void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *dest_o =
      (static_cast<LatticeComplex<T> *>(device_dest_o) + idx);
  LatticeComplex<T> *src_o =
      (static_cast<LatticeComplex<T> *>(device_src_o) + idx);
  LatticeComplex<T> *vec1 =
      (static_cast<LatticeComplex<T> *>(device_vec1) + idx);
  int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    // NOTE: kappa^2 is correct for odd-preconditioned Clover-Wilson system.
    // The preconditioned system is M_oo - kappa^2 * M_oe * M_ee^{-1} * M_eo.
    // After solving for x_o, the even-site solution is:
    //   x_e = M_ee^{-1} (b_e - kappa * M_eo * x_o)
    //   dest_o = ans_o - kappa^2 * tmp1   (where tmp1 = M_oe * x_e / kappa)
    // The extra kappa comes from the Wilson hopping term's kappa factor.
    dest_o[i] = src_o[i] - vec1[i] * kappa * kappa; // kappa^2 in odd-preconditioned system
  }
}
template <typename T>
__global__ void bistabcg_give_rr(void *device_r, void *device_b__o,
                                 void *device_r_tilde, void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *r = (static_cast<LatticeComplex<T> *>(device_r) + idx);
  LatticeComplex<T> *b__o =
      (static_cast<LatticeComplex<T> *>(device_b__o) + idx);
  LatticeComplex<T> *r_tilde =
      (static_cast<LatticeComplex<T> *>(device_r_tilde) + idx);
  int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    r[i] = b__o[i] - r[i];
    r_tilde[i] = r[i];
  }
}
template <typename T>
__global__ void bistabcg_give_p(void *device_p, void *device_r, void *device_v,
                                void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *p = (static_cast<LatticeComplex<T> *>(device_p) + idx);
  LatticeComplex<T> *r = (static_cast<LatticeComplex<T> *>(device_r) + idx);
  LatticeComplex<T> *v = (static_cast<LatticeComplex<T> *>(device_v) + idx);
  LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
  LatticeComplex<T> beta;
  beta = vals[_beta_];
  LatticeComplex<T> omega;
  omega = vals[_omega_];
  int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    p[i] = r[i] + (p[i] - v[i] * omega) * beta;
  }
}
template <typename T>
__global__ void bistabcg_give_s(void *device_s, void *device_r, void *device_v,
                                void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *s = (static_cast<LatticeComplex<T> *>(device_s) + idx);
  LatticeComplex<T> *r = (static_cast<LatticeComplex<T> *>(device_r) + idx);
  LatticeComplex<T> *v = (static_cast<LatticeComplex<T> *>(device_v) + idx);
  LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
  LatticeComplex<T> alpha;
  alpha = vals[_alpha_];
  int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    s[i] = r[i] - v[i] * alpha;
  }
}
template <typename T>
__global__ void bistabcg_give_x_o(void *device_x_o, void *device_p,
                                  void *device_s, void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *x_o = (static_cast<LatticeComplex<T> *>(device_x_o) + idx);
  LatticeComplex<T> *p = (static_cast<LatticeComplex<T> *>(device_p) + idx);
  LatticeComplex<T> *s = (static_cast<LatticeComplex<T> *>(device_s) + idx);
  LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
  LatticeComplex<T> alpha;
  alpha = vals[_alpha_];
  LatticeComplex<T> omega;
  omega = vals[_omega_];
  int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    x_o[i] = x_o[i] + p[i] * alpha + s[i] * omega;
  }
}
template <typename T>
__global__ void bistabcg_give_r_x(void *device_r, void *device_x_o,
                                  void *device_p, void *device_s,
                                  void *device_tt, void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *r = (static_cast<LatticeComplex<T> *>(device_r) + idx);
  LatticeComplex<T> *x_o =
      (static_cast<LatticeComplex<T> *>(device_x_o) + idx);
  LatticeComplex<T> *p = (static_cast<LatticeComplex<T> *>(device_p) + idx);
  LatticeComplex<T> *s = (static_cast<LatticeComplex<T> *>(device_s) + idx);
  LatticeComplex<T> *t = (static_cast<LatticeComplex<T> *>(device_tt) + idx);
  LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
  const LatticeComplex<T> alpha = vals[_alpha_];
  const LatticeComplex<T> omega = vals[_omega_];
  int _ = int(vals[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    r[i] = s[i] - t[i] * omega;
    x_o[i] = x_o[i] + p[i] * alpha + s[i] * omega;
  }
}
template <typename T>
__global__ void bistabcg_give_r(void *device_r, void *device_s, void *device_tt,
                                void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *r = (static_cast<LatticeComplex<T> *>(device_r) + idx);
  LatticeComplex<T> *s = (static_cast<LatticeComplex<T> *>(device_s) + idx);
  LatticeComplex<T> *t = (static_cast<LatticeComplex<T> *>(device_tt) + idx);
  LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
  LatticeComplex<T> omega;
  omega = vals[_omega_];
  int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    r[i] = s[i] - t[i] * omega;
  }
}
template <typename T, int block_size>
__global__ void bistabcg_dot_complete(
    const LatticeComplex<T> *left, const LatticeComplex<T> *right, int n,
    int block_count, LatticeComplex<T> *partials, LatticeComplex<T> *result,
    unsigned int *completion_counter) {
  using complex_data = typename LatticeComplex<T>::_data_type;
  __shared__ complex_data shared[block_size];
  __shared__ int is_last_block;

  const int block = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const int index = block * block_size + thread;
  const int stride = block_count * block_size;
  LatticeComplex<T> sum((T)0, (T)0);
  for (int i = index; i < n; i += stride)
    sum += left[i].conj() * right[i];

  shared[thread].x = sum.real();
  shared[thread].y = sum.imag();
  __syncthreads();
  for (int width = block_size / 2; width > 0; width >>= 1) {
    if (thread < width) {
      shared[thread].x += shared[thread + width].x;
      shared[thread].y += shared[thread + width].y;
    }
    __syncthreads();
  }

  if (thread == 0) {
    partials[block] = LatticeComplex<T>(shared[0].x, shared[0].y);
    __threadfence();
    const unsigned int ticket = atomicInc(
        completion_counter, static_cast<unsigned int>(block_count - 1));
    is_last_block = ticket == static_cast<unsigned int>(block_count - 1);
  }
  __syncthreads();
  if (!is_last_block) return;

  sum = LatticeComplex<T>((T)0, (T)0);
  for (int i = thread; i < block_count; i += block_size)
    sum += partials[i];
  shared[thread].x = sum.real();
  shared[thread].y = sum.imag();
  __syncthreads();
  for (int width = block_size / 2; width > 0; width >>= 1) {
    if (thread < width) {
      shared[thread].x += shared[thread + width].x;
      shared[thread].y += shared[thread + width].y;
    }
    __syncthreads();
  }
  if (thread == 0)
    *result = LatticeComplex<T>(shared[0].x, shared[0].y);
}
template <typename T, int block_size>
__global__ void bistabcg_dot_pair_complete(
    const LatticeComplex<T> *left, const LatticeComplex<T> *right0,
    const LatticeComplex<T> *right1, int n, int block_count,
    LatticeComplex<T> *partials, LatticeComplex<T> *result0,
    LatticeComplex<T> *result1, unsigned int *completion_counter) {
  using complex_data = typename LatticeComplex<T>::_data_type;
  __shared__ complex_data shared0[block_size];
  __shared__ complex_data shared1[block_size];
  __shared__ int is_last_block;

  const int block = static_cast<int>(blockIdx.x);
  const int thread = static_cast<int>(threadIdx.x);
  const int index = block * block_size + thread;
  const int stride = block_count * block_size;
  LatticeComplex<T> sum0((T)0, (T)0);
  LatticeComplex<T> sum1((T)0, (T)0);
  for (int i = index; i < n; i += stride) {
    const LatticeComplex<T> conjugate_left = left[i].conj();
    sum0 += conjugate_left * right0[i];
    sum1 += conjugate_left * right1[i];
  }
  shared0[thread].x = sum0.real();
  shared0[thread].y = sum0.imag();
  shared1[thread].x = sum1.real();
  shared1[thread].y = sum1.imag();
  __syncthreads();
  for (int width = block_size / 2; width > 0; width >>= 1) {
    if (thread < width) {
      shared0[thread].x += shared0[thread + width].x;
      shared0[thread].y += shared0[thread + width].y;
      shared1[thread].x += shared1[thread + width].x;
      shared1[thread].y += shared1[thread + width].y;
    }
    __syncthreads();
  }

  if (thread == 0) {
    partials[block] = LatticeComplex<T>(shared0[0].x, shared0[0].y);
    partials[block_count + block] =
        LatticeComplex<T>(shared1[0].x, shared1[0].y);
    __threadfence();
    const unsigned int ticket = atomicInc(
        completion_counter, static_cast<unsigned int>(block_count - 1));
    is_last_block = ticket == static_cast<unsigned int>(block_count - 1);
  }
  __syncthreads();
  if (!is_last_block) return;

  sum0 = LatticeComplex<T>((T)0, (T)0);
  sum1 = LatticeComplex<T>((T)0, (T)0);
  for (int i = thread; i < block_count; i += block_size) {
    sum0 += partials[i];
    sum1 += partials[block_count + i];
  }
  shared0[thread].x = sum0.real();
  shared0[thread].y = sum0.imag();
  shared1[thread].x = sum1.real();
  shared1[thread].y = sum1.imag();
  __syncthreads();
  for (int width = block_size / 2; width > 0; width >>= 1) {
    if (thread < width) {
      shared0[thread].x += shared0[thread + width].x;
      shared0[thread].y += shared0[thread + width].y;
      shared1[thread].x += shared1[thread + width].x;
      shared1[thread].y += shared1[thread + width].y;
    }
    __syncthreads();
  }
  if (thread == 0) {
    *result0 = LatticeComplex<T>(shared0[0].x, shared0[0].y);
    *result1 = LatticeComplex<T>(shared1[0].x, shared1[0].y);
  }
}
template <typename T>
__global__ void bistabcg_give_diff2(void *device_x, void *device_ans,
                                    void *device_vec, void *device_vals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *x = (static_cast<LatticeComplex<T> *>(device_x) + idx);
  LatticeComplex<T> *ans = (static_cast<LatticeComplex<T> *>(device_ans) + idx);
  LatticeComplex<T> *vec = (static_cast<LatticeComplex<T> *>(device_vec) + idx);
  int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
  for (int i = 0; i < _LAT_SC_ * _; i += _) {
    vec[i] = x[i] - ans[i];
  }
}
//@@@CUDA_TEMPLATE_FOR_DEVICE@@@
template __global__ void bistabcg_give_1beta<double>(void *device_vals);
template __global__ void bistabcg_give_1beta_rp<double>(void *device_vals);
template __global__ void bistabcg_give_1rho_prev<double>(void *device_vals);
template __global__ void bistabcg_give_1alpha<double>(void *device_vals);
template __global__ void bistabcg_give_1omega<double>(void *device_vals);
template __global__ void bistabcg_give_1diff2<double>(void *device_vals);
template __global__ void
bistabcg_give_b_e<double>(void *device_b_e, void *device_ans_e,
                          void *device_vec0, double kappa, void *device_vals);
template __global__ void
bistabcg_give_b_o<double>(void *device_b_o, void *device_ans_o,
                          void *device_vec1, double kappa, void *device_vals);
template __global__ void
bistabcg_give_b__o<double>(void *device_b__o, void *device_b_o,
                           void *device_vec0, double kappa, void *device_vals);
template __global__ void bistabcg_give_dest_o<double>(void *device_dest_o,
                                                      void *device_src_o,
                                                      void *device_vec1,
                                                      double kappa,
                                                      void *device_vals);
template __global__ void bistabcg_give_rr<double>(void *device_r,
                                                  void *device_b__o,
                                                  void *device_r_tilde,
                                                  void *device_vals);
template __global__ void bistabcg_give_p<double>(void *device_p, void *device_r,
                                                 void *device_v,
                                                 void *device_vals);
template __global__ void bistabcg_give_s<double>(void *device_s, void *device_r,
                                                 void *device_v,
                                                 void *device_vals);
template __global__ void bistabcg_give_x_o<double>(void *device_x_o,
                                                   void *device_p,
                                                   void *device_s,
                                                   void *device_vals);
template __global__ void bistabcg_give_r_x<double>(
    void *device_r, void *device_x_o, void *device_p, void *device_s,
    void *device_tt, void *device_vals);
template __global__ void bistabcg_give_r<double>(void *device_r, void *device_s,
                                                 void *device_tt,
                                                 void *device_vals);
template __global__ void bistabcg_dot_complete<double, 256>(
    const LatticeComplex<double> *left, const LatticeComplex<double> *right,
    int n, int block_count, LatticeComplex<double> *partials,
    LatticeComplex<double> *result, unsigned int *completion_counter);
template __global__ void bistabcg_dot_pair_complete<double, 256>(
    const LatticeComplex<double> *left, const LatticeComplex<double> *right0,
    const LatticeComplex<double> *right1, int n, int block_count,
    LatticeComplex<double> *partials, LatticeComplex<double> *result0,
    LatticeComplex<double> *result1, unsigned int *completion_counter);
template __global__ void bistabcg_give_diff2<double>(void *device_x,
                                                     void *device_ans,
                                                     void *device_vec,
                                                     void *device_vals);
//@@@CUDA_TEMPLATE_FOR_DEVICE@@@
template __global__ void bistabcg_give_1beta<float>(void *device_vals);
template __global__ void bistabcg_give_1beta_rp<float>(void *device_vals);
template __global__ void bistabcg_give_1rho_prev<float>(void *device_vals);
template __global__ void bistabcg_give_1alpha<float>(void *device_vals);
template __global__ void bistabcg_give_1omega<float>(void *device_vals);
template __global__ void bistabcg_give_1diff2<float>(void *device_vals);
template __global__ void
bistabcg_give_b_e<float>(void *device_b_e, void *device_ans_e,
                         void *device_vec0, float kappa, void *device_vals);
template __global__ void
bistabcg_give_b_o<float>(void *device_b_o, void *device_ans_o,
                         void *device_vec1, float kappa, void *device_vals);
template __global__ void
bistabcg_give_b__o<float>(void *device_b__o, void *device_b_o,
                          void *device_vec0, float kappa, void *device_vals);
template __global__ void
bistabcg_give_dest_o<float>(void *device_dest_o, void *device_src_o,
                            void *device_vec1, float kappa, void *device_vals);
template __global__ void bistabcg_give_rr<float>(void *device_r,
                                                 void *device_b__o,
                                                 void *device_r_tilde,
                                                 void *device_vals);
template __global__ void bistabcg_give_p<float>(void *device_p, void *device_r,
                                                void *device_v,
                                                void *device_vals);
template __global__ void bistabcg_give_s<float>(void *device_s, void *device_r,
                                                void *device_v,
                                                void *device_vals);
template __global__ void bistabcg_give_x_o<float>(void *device_x_o,
                                                  void *device_p,
                                                  void *device_s,
                                                  void *device_vals);
template __global__ void bistabcg_give_r_x<float>(
    void *device_r, void *device_x_o, void *device_p, void *device_s,
    void *device_tt, void *device_vals);
template __global__ void bistabcg_give_r<float>(void *device_r, void *device_s,
                                                void *device_tt,
                                                void *device_vals);
template __global__ void bistabcg_dot_complete<float, 256>(
    const LatticeComplex<float> *left, const LatticeComplex<float> *right,
    int n, int block_count, LatticeComplex<float> *partials,
    LatticeComplex<float> *result, unsigned int *completion_counter);
template __global__ void bistabcg_dot_pair_complete<float, 256>(
    const LatticeComplex<float> *left, const LatticeComplex<float> *right0,
    const LatticeComplex<float> *right1, int n, int block_count,
    LatticeComplex<float> *partials, LatticeComplex<float> *result0,
    LatticeComplex<float> *result1, unsigned int *completion_counter);
template __global__ void bistabcg_give_diff2<float>(void *device_x,
                                                    void *device_ans,
                                                    void *device_vec,
                                                    void *device_vals);
} // namespace qcu
