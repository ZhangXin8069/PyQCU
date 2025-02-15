// Copyright (c) 2018 Evan S Weinberg
// CG inverter.
// Solves lhs = A^(-1) rhs with Chebyshev basis CA-CG.
// Paper: Communication avoiding multigrid preconditioned conjugate gradient method
// for extreme scale multiphase CFD simulations. Note the typo on line 14.

// Modified Chebyshev recursion relation:
// T_0(x) = 1
// m = 2/(lambda_max - lambda_min)
// b = -(lambda_max + lambda_min)/(lambda_max - lambda_min)
// T_1(x) = m x + b
// T_{n+1}(x) = 2(mx+b) T_n(x) - T_{n-1}(x)

// Assumes the matrix is Hermitian (symmetric) positive definite.



#ifndef QLINALG_INVERTER_CHEBY_CA_CG
#define QLINALG_INVERTER_CHEBY_CA_CG

#include <string>
#include <sstream>
#include <complex>

using std::complex;
using std::stringstream;


#ifndef QLINALG_FCN_POINTER
#define QLINALG_FCN_POINTER
typedef void (*matrix_op_real)(double*,double*,void*);
typedef void (*matrix_op_cplx)(complex<double>*,complex<double>*,void*);
#endif

#include "inverter_struct.h"
#include "../verbosity/verbosity.h"

#ifndef QLINALG_EIGEN_MATRIX
#define QLINALG_EIGEN_MATRIX
typedef Matrix<double, Dynamic, Dynamic, ColMajor> rSquareMatrix;
typedef Matrix<double, Dynamic, 1> rVector;
#endif

template <typename T>
inversion_info minv_vector_cheby_ca_cg(T *phi, T *phi0, int size, int max_iter, double eps, typename RealReducer<T>::type lambda_min, typename RealReducer<T>::type lambda_max, int s, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb)
{

  int k;

  stringstream ss;
  ss << "Cheby-CA-CG(s=" << s << ")";

  // Initialize vectors.
  T *Svec[s];
  T *ASvec[s];
  T *Qvec[s];
  T *AQvec[s];
  T *Tvec[s];
  using Real = typename RealReducer<T>::type;
  using rSquareMatrix = Matrix<Real, Dynamic, Dynamic, ColMajor>;
  using rVector = Matrix<Real, Dynamic, 1>;
  rVector a(s), g(s);
  rSquareMatrix QAQ(s,s), QAS(s,s), B(s,s);
  //rSquareMatrix Rsq(s,s), RsqNew(s,s);

  // Factors which map the linear operator onto [-1,1]
  Real m = 2./(lambda_max-lambda_min);
  Real b = -(lambda_max+lambda_min)/(lambda_max-lambda_min);

  Real rsq, bsqrt, truersq;
  inversion_info invif;

  // Allocate memory.
  for (int i = 0; i < s; i++) {
    Svec[i] = allocate_vector<T>(size); zero_vector(Svec[i], size);
    ASvec[i] = allocate_vector<T>(size); zero_vector(ASvec[i], size);
    Qvec[i] = allocate_vector<T>(size); zero_vector(Qvec[i], size);
    AQvec[i] = allocate_vector<T>(size); zero_vector(AQvec[i], size);
    Tvec[i] = allocate_vector<T>(size); zero_vector(Tvec[i], size);
  }

  // Initialize values.
  rsq = 0.0; bsqrt = 0.0; truersq = 0.0; k=0;
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // 1. Compute r = b - Ax
  (*matrix_vector)(Tvec[0], phi, extra_info); invif.ops_count++;
  cxpayz(phi0, -1.0, Tvec[0], Svec[0], size);

  // 2. Compute S = (T_0 r, T_1 r, ..., T_{s-1} r)

  // Get A T_0 r
  (*matrix_vector)(ASvec[0], Svec[0], extra_info); invif.ops_count++;

  // Get T_1 r = (m A + b) T_0 r - r
  caxpbyz(m, ASvec[0], b, Svec[0], Svec[1], size);

  // Get A T_1 r
  (*matrix_vector)(ASvec[1], Svec[1], extra_info); invif.ops_count++;

  // Dig into the recursion relation
  for (int i = 2; i < s; i++) {
    caxpbypczw(2.*m, ASvec[i-1], 2.*b, Svec[i-1], -1., Svec[i-2], Svec[i], size);
    (*matrix_vector)(ASvec[i], Svec[i], extra_info); invif.ops_count++;
  }

  // End 2.

  // 3. Q = S, AQ = AS
  for (int i = 0; i < s; i++) {
    copy_vector(Qvec[i], Svec[i], size);
    copy_vector(AQvec[i], ASvec[i], size);
  }

  // 4: for k = 0, 1, 2, ... until convergence do...
  for(k = 0; k< max_iter; k+=s) {

    // 5. Compute Q^dag A Q
    for (int i = 0; i < s; i++) {
      for (int j = 0; j < s; j++) {
        QAQ(i,j) = re_dot(Qvec[i],AQvec[j],size);
      }
    }

    // 6. Compute Q^dag r
    for (int i = 0; i < s; i++) {
      g(i) = re_dot(Qvec[i], Svec[0], size);
    }

    // 7. a = QAQ^{-1} g
    a = QAQ.fullPivLu().solve(g);

    // 8. x += Q a
    for (int i = 0; i < s; i++) {
      caxpy(a(i), Qvec[i], phi, size);
    }

    // 9. Explicitly compute the residual
    //zero_vector(Tvec[0], size);
    //(*matrix_vector)(Tvec[0], phi, extra_info); invif.ops_count++;
    //cxpayz(phi0, -1.0, Tvec[0], Svec[0], size);

    // 9. r -= A Q a
    for (int i = 0; i < s; i++) {
      caxpy(-a(i), AQvec[i], Svec[0], size);
    }


    // Check convergence
    rsq = norm2sq(Svec[0], size);
    print_verbosity_resid(verb, ss.str(), k+s, invif.ops_count, sqrt(rsq)/bsqrt);
    if (sqrt(rsq) < eps*bsqrt || k == max_iter - s) {
      break;
    }

    // 10. Compute S = (T_0 r, T_1 r, ..., T_{s-1} r)

    // Get A T_0 r
    zero_vector(ASvec[0], size);
    (*matrix_vector)(ASvec[0], Svec[0], extra_info); invif.ops_count++;

    // Get T_1 r = (m A + b) T_0 r - r
    caxpbyz(m, ASvec[0], b, Svec[0], Svec[1], size);

    // Get A T_1 r
    zero_vector(ASvec[1], size);
    (*matrix_vector)(ASvec[1], Svec[1], extra_info); invif.ops_count++;

    // Dig into the recursion relation
    for (int i = 2; i < s; i++) {
      caxpbypczw(2.*m, ASvec[i-1], 2.*b, Svec[i-1], -1., Svec[i-2], Svec[i], size);
      zero_vector(ASvec[i], size);
      (*matrix_vector)(ASvec[i], Svec[i], extra_info); invif.ops_count++;
    }

    // End 10.

    // 11. Compute Q^dag A S
    for (int i = 0; i < s; i++) {
      for (int j = 0; j < s; j++) {
        QAS(i,j) = re_dot(Qvec[i], ASvec[j],size);
      }
    }

    // 12. B = (QAQ)^{-1} QAS
    B = QAQ.fullPivLu().solve(QAS);

    // 13. Q = S - Q B
    for (int i = 0; i < s; i++) {
      copy_vector(Tvec[i], Svec[i], size);
    }
    for (int i = 0; i < s; i++) {
      for (int j = 0; j < s; j++) {
        caxpy(-B(i,j), Qvec[i], Tvec[j], size);
      }
    }
    for (int i = 0; i < s; i++) {
      copy_vector(Qvec[i], Tvec[i], size);
    }

    // 14. AQ = AS - AQ B
    for (int i = 0; i < s; i++) {
      copy_vector(Tvec[i], ASvec[i], size);
    }
    for (int i = 0; i < s; i++) {
      for (int j = 0; j < s; j++) {
        caxpy(-B(i,j), AQvec[i], Tvec[j], size);
      }
    }
    for (int i = 0; i < s; i++) {
      copy_vector(AQvec[i], Tvec[i], size);
    }

    // 15. End for
  } 
    
  if(k >= max_iter-s) {
    invif.success = false;
  }
  else
  {
     invif.success = true;
  }
  
  k++; 
  
  zero_vector(Svec[0], size);
  (*matrix_vector)(Svec[0],phi,extra_info); invif.ops_count++;
  truersq = diffnorm2sq(Svec[0], phi0, size);
  
  // Free all the things!
  for (int i = 0; i < s; i++) {
    deallocate_vector(&Svec[i]);
    deallocate_vector(&ASvec[i]);
    deallocate_vector(&Qvec[i]);
    deallocate_vector(&AQvec[i]);
    deallocate_vector(&Tvec[i]);
  }
  
  print_verbosity_summary(verb, ss.str(), invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  
  invif.resSq = truersq;
  invif.iter = k+s-1;
  invif.name = ss.str();
  return invif; // Convergence 
}


// Performs CG(restart_freq) with restarts when restart_freq is hit.
// This may be sloppy, but it works.
template <typename T>
inversion_info minv_vector_cheby_ca_cg_restart(T *phi, T *phi0, int size, int max_iter, double res, int s, typename RealReducer<T>::type lambda_max, int restart_freq, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb)
{
  int iter; // counts total number of iterations.
  int ops_count; 
  inversion_info invif;
  using Real = typename RealReducer<T>::type;

  Real bsqrt = sqrt(norm2sq(phi0, size));
  
  inversion_verbose_struct verb_rest;
  shuffle_verbosity_restart(&verb_rest, verb);
  
  stringstream ss;
  ss << "Cheby-CA-CG(s=" << s << "," << restart_freq << ")";
  
  iter = 0; ops_count = 0; 
  do
  {
    invif = minv_vector_cheby_ca_cg(phi, phi0, size, min(max_iter, restart_freq), res, s, lambda_max, matrix_vector, extra_info, &verb_rest);
    iter += invif.iter;
    ops_count += invif.ops_count; 
    
    print_verbosity_restart(verb, ss.str(), iter, ops_count, sqrt(invif.resSq)/bsqrt);
  }
  while (iter < max_iter && invif.success == false && sqrt(invif.resSq)/bsqrt > res);
  
  invif.iter = iter; invif.ops_count = ops_count; 
  
  print_verbosity_summary(verb, ss.str(), invif.success, iter, invif.ops_count, sqrt(invif.resSq)/bsqrt);
  
  invif.name = ss.str();
  // invif.resSq is good.
  if (sqrt(invif.resSq)/bsqrt > res)
  {
    invif.success = false;
  }
  else
  {
    invif.success = true;
  }
  
  return invif;
}

// Version where lambda_min = 0.
template <typename T>
inversion_info minv_vector_cheby_ca_cg(T *phi, T *phi0, int size, int max_iter, double eps, typename RealReducer<T>::type lambda_max, int s, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb)
{
  return minv_vector_cheby_ca_cg(phi, phi0, size, max_iter, eps, 0., lambda_max, s, matrix_vector, extra_info, verb);
}


#endif 