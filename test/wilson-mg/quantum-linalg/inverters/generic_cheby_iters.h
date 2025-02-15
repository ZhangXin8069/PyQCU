// Copyright (c) 2018 Evan S Weinberg
// Taken from the article "The Chebyshev iteration revisited"
// 
// Solves lhs = A^(-1) rhs with Richardson iterations.
// For an operator satisfying the positive half-plane condition,
// the Richardson iteration converges if the relaxation
// parameter, alpha, satisfies
// alpha < 2 Re(lambda)/[ Re(lambda)^2 + Im(lambda)^2]
// for the eigenvalue lambda with the largest magnitude.

#ifndef QLINALG_INVERTER_CHEBY_ITERS
#define QLINALG_INVERTER_CHEBY_ITERS

#include <string>
#include <sstream>
#include <complex>

using std::complex;
using std::stringstream;

#include "inverter_struct.h"
#include "../verbosity/verbosity.h"
#include "../blas/generic_vector.h"

// Solve Ax = b with Chebyshev Iterations
template <typename T>
inversion_info minv_vector_cheby_iters(T *phi, T *phi0, int size, int max_iter, double res, typename RealReducer<T>::type lambda_min, typename RealReducer<T>::type lambda_max, int check_freq, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verbosity = nullptr)
{
  using Real = typename RealReducer<T>::type;

  // Iterators.
  int k;

  // Vectors
  T *x = phi;
  T *b = phi0;
  T *v;
  T *Av;
  T *r;

  // Alg name:
  stringstream ss;
  ss << "ChebyIters(" << lambda_min << "," << lambda_max << ")";

  // Scalars
  Real alpha = 0.5*(lambda_min + lambda_max);
  Real c = 0.5*(lambda_max - lambda_min);
  Real cdiv2sq = 0.25*c*c; // needed in the alg
  Real psi = 0;
  Real omega_old = 0;
  Real omega = 0;

  // More scalars.
  Real truersq, rsq, bsqrt;
  inversion_info invif;

  // Allocate memory.
  v = allocate_vector<T>(size);
  Av = allocate_vector<T>(size);
  r = allocate_vector<T>(size);
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(b, size));

  // Form residual.
  zero_vector(Av, size);
  (*matrix_vector)(Av, x, extra_info); invif.ops_count++;
  caxpbyz(1.0, b, -1.0, Av, r, size);

  if (max_iter > 0) {
    // n = 0, explicitly unrolled.
    k = 1;
    psi = 0;
    omega = 1.0/alpha;
    copy_vector(v, r, size);
    caxpy(omega, v, x, size);
    zero_vector(Av, size);
    (*matrix_vector)(Av, v, extra_info); invif.ops_count++;
    caxpy(-omega, Av, r, size);
  }

  // n = 1, explicitly unrolled
  if (max_iter > 1) {
    k = 2;
    psi = -0.5*c*c/(alpha*alpha);
    omega = 1.0/(alpha + psi*alpha);
    cxpay(r, -psi, v, size); // v = r - psi v
    caxpy(omega, v, x, size); // x += omega v
    zero_vector(Av, size);
    (*matrix_vector)(Av, v, extra_info); invif.ops_count++;
    caxpy(-omega, Av, r, size); // r -= omega A v
  }

  // Iterate for the rest
  for (k = 3; k <= max_iter; k++) {
    psi = -cdiv2sq * omega * omega; // psi = -(c/2)^2 omega^2
    omega_old = omega;
    omega = 1.0/(alpha - cdiv2sq*omega_old); // omega = (alpha - (c/2)^2 omega)^{-1}

    cxpay(r, -psi, v, size); // v = r - psi v
    caxpy(omega, v, x, size); // x += omega v
    zero_vector(Av, size);
    (*matrix_vector)(Av, v, extra_info); invif.ops_count++;
    caxpy(-omega, Av, r, size); // r -= omega A v

    if (k%check_freq == 0) {
      rsq = norm2sq(r, size);

      print_verbosity_resid(verbosity, ss.str(), k, invif.ops_count, sqrt(rsq)/bsqrt); 

      // Check convergence. 
      if (sqrt(rsq) < res*bsqrt || k == max_iter) {
        break;
      }
    }
  }
    
  if(k >= max_iter) {
    invif.success = false;
  }
  else
  {
     invif.success = true;
  }
  
  // Check true residual.
  zero_vector(Av, size);
  (*matrix_vector)(Av, x, extra_info); invif.ops_count++;
  truersq = diffnorm2sq(Av, b, size);
  
  // Free all the things!
  deallocate_vector(&v);
  deallocate_vector(&Av);
  deallocate_vector(&r);

  print_verbosity_summary(verbosity, ss.str(), invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  invif.resSq = truersq;
  invif.iter = k; // idk
  invif.name = ss.str();

  return invif; // Convergence 
} 

#endif
