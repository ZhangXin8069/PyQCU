// Copyright (c) 2018 Evan S Weinberg
// CG inverter.
// Solves lhs = A^(-1) rhs with CA-CG.
// https://research.nvidia.com/sites/default/files/pubs/2016-04_S-Step-and-Communication-Avoiding/nvr-2016-003.pdf
// Assumes the matrix is Hermitian (symmetric) positive definite.



#ifndef QLINALG_INVERTER_CA_CG
#define QLINALG_INVERTER_CA_CG

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


template <typename T>
inversion_info minv_vector_ca_cg(T *phi, T *phi0, int size, int max_iter, double eps, int s, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb)
{

  int k;

  stringstream ss;
  ss << "CA-CG(s=" << s << ")";

  // Initialize vectors.
  T *Svec[s+1];
  T **ASvec = Svec+1; // because we're using a power basis
  T *Qvec[s];
  T *AQvec[s];
  T *Tvec[s];
  using Real = typename RealReducer<T>::type;
  using rSquareMatrix = Matrix<Real, Dynamic, Dynamic, ColMajor>;
  using rVector = Matrix<Real, Dynamic, 1>;
  rVector a(s), g(s);
  rSquareMatrix QAQ(s,s), QAS(s,s), B(s,s);
  //rSquareMatrix Rsq(s,s), RsqNew(s,s);

  Real rsq, bsqrt, truersq;
  inversion_info invif;

  // Allocate memory.
  for (int i = 0; i < s; i++) {
    Qvec[i] = allocate_vector<T>(size); zero_vector(Qvec[i], size);
    AQvec[i] = allocate_vector<T>(size); zero_vector(AQvec[i], size);
    Tvec[i] = allocate_vector<T>(size); zero_vector(Tvec[i], size);
  }
  for (int i = 0; i <= s; i++) {
    Svec[i] = allocate_vector<T>(size); zero_vector(Svec[i], size);
  }

  // Initialize values.
  rsq = 0.0; bsqrt = 0.0; truersq = 0.0; k=0;
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // 1. Compute r = b - Ax
  (*matrix_vector)(Tvec[0], phi, extra_info); invif.ops_count++;
  cxpayz(phi0, -1.0, Tvec[0], Svec[0], size);

  // 2. Compute S = (r, Ar, ..., A^{s-1} r)
  for (int i = 0; i < s; i++) {
    (*matrix_vector)(ASvec[i], Svec[i], extra_info); invif.ops_count++;
  }

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
    for (int i = 0; i < s; i++) {
      zero_vector(ASvec[i], size);
      (*matrix_vector)(ASvec[i], Svec[i], extra_info); invif.ops_count++;
    }

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
    deallocate_vector(&Qvec[i]);
    deallocate_vector(&AQvec[i]);
    deallocate_vector(&Tvec[i]);
  }
  for (int i = 0; i <= s; i++) {
    deallocate_vector(&Svec[i]);
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
inversion_info minv_vector_ca_cg_restart(T *phi, T *phi0, int size, int max_iter, double res, int s, int restart_freq, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb = nullptr)
{
  int iter; // counts total number of iterations.
  int ops_count; 
  inversion_info invif;
  using Real = typename RealReducer<T>::type;

  Real bsqrt = sqrt(norm2sq(phi0, size));
  
  inversion_verbose_struct verb_rest;
  shuffle_verbosity_restart(&verb_rest, verb);
  
  stringstream ss;
  ss << "CA-CG(s=" << s << "," << restart_freq << ")";
  
  iter = 0; ops_count = 0; 
  do
  {
    invif = minv_vector_ca_cg(phi, phi0, size, min(max_iter, restart_freq), res, s, matrix_vector, extra_info, &verb_rest);
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



#endif 

// Old implementation
/*
template <typename T>
inversion_info minv_vector_ca_cg(T *phi, T *phi0, int size, int max_iter, double eps, int s, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb = nullptr)
{

  int k;

  stringstream ss;
  ss << "CA-CG(s=" << s << ")";

  // Initialize vectors.
  T *Tvec[s+1];
  T *Rvec[s];
  T *Qvec[s];
  T *Pvec[s], *Ptmpvec[s];
  using Real = typename RealReducer<T>::type;
  using rSquareMatrix = Matrix<Real, Dynamic, Dynamic, ColMajor>;
  using rVector = Matrix<Real, Dynamic, 1>;
  rVector a(s), g(s);
  rSquareMatrix B(s,s), C(s,s), W(s,s);
  //rSquareMatrix Rsq(s,s), RsqNew(s,s);

  Real rsq, bsqrt, truersq;
  inversion_info invif;

  // Allocate memory.
  for (int i = 0; i <= s; i++) {
    Tvec[i] = allocate_vector<T>(size);
    Rvec[i] = Tvec[i];
    if (i > 0) { Qvec[i-1] = Tvec[i]; }
    zero_vector(Tvec[i], size);
  }

  for (int i = 0; i < s; i++) {
    Pvec[i] = allocate_vector<T>(size);
    Ptmpvec[i] = allocate_vector<T>(size);
    zero_vector(Pvec[i], size);
    zero_vector(Ptmpvec[i], size);
  }

  // Initialize values.
  rsq = 0.0; bsqrt = 0.0; truersq = 0.0; k=0;
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // 2. Compute r = b - Ax
  (*matrix_vector)(Pvec[0], phi, extra_info); invif.ops_count++;
  cxpayz(phi0, -1.0, Pvec[0], Rvec[0], size);

  // 3: for i = 0, s, 2s ... until convergence do
  for(k = 0; k< max_iter; k+=s) {

    // 4: Compute T = [r_k, ..., A^s r_k]
    for (int i = 0; i < s; i++) {
      (*matrix_vector)(Tvec[i+1], Tvec[i], extra_info); invif.ops_count++;
    }

    // 5: Let R_i = [r_k, Ar_k, ..., A^{s-1} r_k] 
    // no op: taken care of by aliasing.

    // 6: Let Q_i = [Ar_k, A^2 r_k, ..., A^s r_k]
    // no op: taken care of by aliasing.

    // 7: if i == 0 then
    if (k == 0) {
      // 8: Set P = R
      for (int i = 0; i < s; i++) {
        copy_vector(Pvec[i], Rvec[i], size);
      }
    } else { // 9: else

      // 10: Compute C_i = -Q_i^dag P [may need to switch?]
      for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
          C(i,j) = -re_dot(Pvec[i], Qvec[j],size);
        }
      }

      // 11: Solve W_{i-1} B_i = C_i
      B = W.fullPivLu().solve(C); //Rsq.fullPivLu().solve(RsqNew); 
      
      // 12: Compute P = R + P B
      for (int i = 0; i < s; i++) {
        copy_vector(Ptmpvec[i], Rvec[i], size);
      }
      for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
          caxpy(B(i,j), Pvec[i], Ptmpvec[j], size);
        }
      }
      for (int i = 0; i < s; i++) {
        copy_vector(Pvec[i], Ptmpvec[i], size);
      }
 
      // 13: end if
    }

    // 14: Compute W = Q^dag P;
    for (int i = 0; i < s; i++) {
      for (int j = 0; j < s; j++) {
        W(i,j) = re_dot(Pvec[i],Qvec[j],size);
      }
    }

    // 15: Compute g = P^dag r_i
    for (int i = 0; i < s; i++) {
      g(i) = re_dot(Pvec[i], Rvec[0], size);
    }

    // 16: Solve W a = g
    a = W.fullPivLu().solve(g);

    // 17: Compute x_{k+s} = P_i a_i
    for (int i = 0; i < s; i++) {
      caxpy(a(i), Pvec[i], phi, size);
    }

    // 18: Compute r_{k+s} = b - A x_{k+s}
    zero_vector(Ptmpvec[0], size);
    (*matrix_vector)(Ptmpvec[0], phi, extra_info); invif.ops_count++;
    cxpayz(phi0, -1.0, Ptmpvec[0], Rvec[0], size);

    // 19: Check convergence
    rsq = norm2sq(Rvec[0], size);
    print_verbosity_resid(verb, ss.str(), k+s, invif.ops_count, sqrt(rsq)/bsqrt);
    if (sqrt(rsq) < eps*bsqrt || k == max_iter - s) {
      break;
    }

  } 
    
  if(k == max_iter-s) {
    invif.success = false;
  }
  else
  {
     invif.success = true;
  }
  
  k++; 
  
  zero_vector(Ptmpvec[0], size);
  (*matrix_vector)(Ptmpvec[0],phi,extra_info); invif.ops_count++;
  truersq = diffnorm2sq(Ptmpvec[0], phi0, size);
  
  // Free all the things!
  for (int i = 0; i <= s; i++) {
    deallocate_vector(&Tvec[i]);
    if (i < s) {
      deallocate_vector(&Pvec[i]);
      deallocate_vector(&Ptmpvec[i]);
    }
  }

  
  print_verbosity_summary(verb, ss.str(), invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  
  invif.resSq = truersq;
  invif.iter = k+s-1;
  invif.name = ss.str();
  return invif; // Convergence 
}*/

