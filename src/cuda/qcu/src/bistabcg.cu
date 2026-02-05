#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu
{
  template <typename T>
  __global__ void bistabcg_give_1beta(void *device_vals)
  {
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    vals[_beta_] = (vals[_rho_] / vals[_rho_prev_]) * (vals[_alpha_] / vals[_omega_]);
  }
  template <typename T>
  __global__ void bistabcg_give_1rho_prev(void *device_vals)
  {
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    vals[_rho_prev_] = vals[_rho_];
  }
  template <typename T>
  __global__ void bistabcg_give_1alpha(void *device_vals)
  {
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    vals[_alpha_] = vals[_rho_] / vals[_tmp0_];
  }
  template <typename T>
  __global__ void bistabcg_give_1omega(void *device_vals)
  {
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    vals[_omega_] = vals[_tmp0_] / vals[_tmp1_];
  }
  template <typename T>
  __global__ void bistabcg_give_1diff(void *device_vals)
  {
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    vals[_diff_tmp_] = vals[_diff_tmp_] / vals[_norm2_tmp_];
  }
  template <typename T>
  __global__ void bistabcg_give_b_e(void *device_b_e, void *device_ans_e,
                                    void *device_vec0, T kappa,
                                    void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *b_e = (static_cast<LatticeComplex<T> *>(device_b_e) + idx);
    LatticeComplex<T> *ans_e = (static_cast<LatticeComplex<T> *>(device_ans_e) + idx);
    LatticeComplex<T> *vec0 = (static_cast<LatticeComplex<T> *>(device_vec0) + idx);
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      b_e[i] = ans_e[i] - vec0[i] * kappa; // b_e=ans_e-kappa*D_eo(ans_o)
    }
  }
  template <typename T>
  __global__ void bistabcg_give_b_o(void *device_b_o, void *device_ans_o,
                                    void *device_vec1, T kappa,
                                    void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *b_o = (static_cast<LatticeComplex<T> *>(device_b_o) + idx);
    LatticeComplex<T> *ans_o = (static_cast<LatticeComplex<T> *>(device_ans_o) + idx);
    LatticeComplex<T> *vec1 = (static_cast<LatticeComplex<T> *>(device_vec1) + idx);
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      b_o[i] = ans_o[i] - vec1[i] * kappa; // b_o=ans_o-kappa*D_oe(ans_e)
    }
  }
  template <typename T>
  __global__ void bistabcg_give_b__o(void *device_b__o, void *device_b_o,
                                     void *device_vec0, T kappa,
                                     void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *b__o = (static_cast<LatticeComplex<T> *>(device_b__o) + idx);
    LatticeComplex<T> *b_o = (static_cast<LatticeComplex<T> *>(device_b_o) + idx);
    LatticeComplex<T> *vec0 = (static_cast<LatticeComplex<T> *>(device_vec0) + idx);
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      b__o[i] = b_o[i] + vec0[i] * kappa; // b__o=b_o+kappa*D_oe(b_e)
    }
  }
  template <typename T>
  __global__ void bistabcg_give_dest_o(void *device_dest_o, void *device_src_o,
                                       void *device_vec1, T kappa,
                                       void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *dest_o = (static_cast<LatticeComplex<T> *>(device_dest_o) + idx);
    LatticeComplex<T> *src_o = (static_cast<LatticeComplex<T> *>(device_src_o) + idx);
    LatticeComplex<T> *vec1 = (static_cast<LatticeComplex<T> *>(device_vec1) + idx);
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      dest_o[i] = src_o[i] - vec1[i] * kappa * kappa; // dest_o=ans_o-kappa^2*tmp1
    }
  }
  template <typename T>
  __global__ void bistabcg_give_rr(void *device_r, void *device_b__o,
                                   void *device_r_tilde, void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *r = (static_cast<LatticeComplex<T> *>(device_r) + idx);
    LatticeComplex<T> *b__o = (static_cast<LatticeComplex<T> *>(device_b__o) + idx);
    LatticeComplex<T> *r_tilde =
        (static_cast<LatticeComplex<T> *>(device_r_tilde) + idx);
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      r[i] = b__o[i] - r[i];
      r_tilde[i] = r[i];
    }
  }
  template <typename T>
  __global__ void bistabcg_give_p(void *device_p, void *device_r, void *device_v,
                                  void *device_vals)
  {
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
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      p[i] = r[i] + (p[i] - v[i] * omega) * beta;
    }
  }
  template <typename T>
  __global__ void bistabcg_give_s(void *device_s, void *device_r, void *device_v,
                                  void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *s = (static_cast<LatticeComplex<T> *>(device_s) + idx);
    LatticeComplex<T> *r = (static_cast<LatticeComplex<T> *>(device_r) + idx);
    LatticeComplex<T> *v = (static_cast<LatticeComplex<T> *>(device_v) + idx);
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    LatticeComplex<T> alpha;
    alpha = vals[_alpha_];
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      s[i] = r[i] - v[i] * alpha;
    }
  }
  template <typename T>
  __global__ void bistabcg_give_x_o(void *device_x_o, void *device_p,
                                    void *device_s, void *device_vals)
  {
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
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      x_o[i] = x_o[i] + p[i] * alpha + s[i] * omega;
    }
  }
  template <typename T>
  __global__ void bistabcg_give_r(void *device_r, void *device_s, void *device_tt,
                                  void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *r = (static_cast<LatticeComplex<T> *>(device_r) + idx);
    LatticeComplex<T> *s = (static_cast<LatticeComplex<T> *>(device_s) + idx);
    LatticeComplex<T> *t = (static_cast<LatticeComplex<T> *>(device_tt) + idx);
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T> *>(device_vals);
    LatticeComplex<T> omega;
    omega = vals[_omega_];
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      r[i] = s[i] - t[i] * omega;
    }
  }
  template <typename T>
  __global__ void bistabcg_give_diff(void *device_x, void *device_ans,
                                     void *device_vec, void *device_vals)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *x = (static_cast<LatticeComplex<T> *>(device_x) + idx);
    LatticeComplex<T> *ans = (static_cast<LatticeComplex<T> *>(device_ans) + idx);
    LatticeComplex<T> *vec = (static_cast<LatticeComplex<T> *>(device_vec) + idx);
    int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
    for (int i = 0; i < _LAT_SC_ * _; i += _)
    {
      vec[i] = x[i] - ans[i];
    }
  }
  //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
  template __global__ void bistabcg_give_1beta<double>(void *device_vals);
  template __global__ void bistabcg_give_1rho_prev<double>(void *device_vals);
  template __global__ void bistabcg_give_1alpha<double>(void *device_vals);
  template __global__ void bistabcg_give_1omega<double>(void *device_vals);
  template __global__ void bistabcg_give_1diff<double>(void *device_vals);
  template __global__ void bistabcg_give_b_e<double>(void *device_b_e, void *device_ans_e,
                                                     void *device_vec0, double kappa,
                                                     void *device_vals);
  template __global__ void bistabcg_give_b_o<double>(void *device_b_o, void *device_ans_o,
                                                     void *device_vec1, double kappa,
                                                     void *device_vals);
  template __global__ void bistabcg_give_b__o<double>(void *device_b__o, void *device_b_o,
                                                      void *device_vec0, double kappa,
                                                      void *device_vals);
  template __global__ void bistabcg_give_dest_o<double>(void *device_dest_o, void *device_src_o,
                                                        void *device_vec1, double kappa,
                                                        void *device_vals);
  template __global__ void bistabcg_give_rr<double>(void *device_r, void *device_b__o,
                                                    void *device_r_tilde, void *device_vals);
  template __global__ void bistabcg_give_p<double>(void *device_p, void *device_r, void *device_v,
                                                   void *device_vals);
  template __global__ void bistabcg_give_s<double>(void *device_s, void *device_r, void *device_v,
                                                   void *device_vals);
  template __global__ void bistabcg_give_x_o<double>(void *device_x_o, void *device_p,
                                                     void *device_s, void *device_vals);
  template __global__ void bistabcg_give_r<double>(void *device_r, void *device_s, void *device_tt,
                                                   void *device_vals);
  template __global__ void bistabcg_give_diff<double>(void *device_x, void *device_ans,
                                                      void *device_vec, void *device_vals);
  //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
  template __global__ void bistabcg_give_1beta<float>(void *device_vals);
  template __global__ void bistabcg_give_1rho_prev<float>(void *device_vals);
  template __global__ void bistabcg_give_1alpha<float>(void *device_vals);
  template __global__ void bistabcg_give_1omega<float>(void *device_vals);
  template __global__ void bistabcg_give_1diff<float>(void *device_vals);
  template __global__ void bistabcg_give_b_e<float>(void *device_b_e, void *device_ans_e,
                                                    void *device_vec0, float kappa,
                                                    void *device_vals);
  template __global__ void bistabcg_give_b_o<float>(void *device_b_o, void *device_ans_o,
                                                    void *device_vec1, float kappa,
                                                    void *device_vals);
  template __global__ void bistabcg_give_b__o<float>(void *device_b__o, void *device_b_o,
                                                     void *device_vec0, float kappa,
                                                     void *device_vals);
  template __global__ void bistabcg_give_dest_o<float>(void *device_dest_o, void *device_src_o,
                                                       void *device_vec1, float kappa,
                                                       void *device_vals);
  template __global__ void bistabcg_give_rr<float>(void *device_r, void *device_b__o,
                                                   void *device_r_tilde, void *device_vals);
  template __global__ void bistabcg_give_p<float>(void *device_p, void *device_r, void *device_v,
                                                  void *device_vals);
  template __global__ void bistabcg_give_s<float>(void *device_s, void *device_r, void *device_v,
                                                  void *device_vals);
  template __global__ void bistabcg_give_x_o<float>(void *device_x_o, void *device_p,
                                                    void *device_s, void *device_vals);
  template __global__ void bistabcg_give_r<float>(void *device_r, void *device_s, void *device_tt,
                                                  void *device_vals);
  template __global__ void bistabcg_give_diff<float>(void *device_x, void *device_ans,
                                                     void *device_vec, void *device_vals);
}
