// Copyright (c) 2017 Evan S Weinberg
// Transpose free QMR (TFQMR) inverter
// Solves lhs = A^(-1) rhs with tfqmr
// Makes no assumptions about the matrix.
// Based on MATLAB code from 
// https://www.mathworks.com/matlabcentral/fileexchange/2198-iterative-methods-for-linear-and-nonlinear-equations?focused=6124557&tab=function

// See licensing at bottom.

#ifndef QLINALG_INVERTER_TFQMR
#define QLINALG_INVERTER_TFQMR

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


// Solves lhs = A^(-1) rhs using tfqmr
template <typename T>
inversion_info minv_vector_tfqmr(T  *phi, T  *phi0, int size, int max_iter, double eps, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb = nullptr)
{
  // tfqmr solutions to Mphi = b 
  using Real = typename RealReducer<T>::type;

  // Initialize vectors.
  T* u[2];
  T* y[2];
  T *d, *v, *w, *r; 
  int j,k,m;
  T alpha, beta, eta, sigma, rho, rho_new;
  Real tau, theta, c; 
  Real bsqrt, truersq;
  inversion_info invif;

  // Convergence test.
  bool converged = false;

  // Allocate memory.
  u[0] = allocate_vector<T>(size);
  u[1] = allocate_vector<T>(size);
  y[0] = allocate_vector<T>(size);
  y[1] = allocate_vector<T>(size);
  d = allocate_vector<T>(size);
  v = allocate_vector<T>(size);
  w = allocate_vector<T>(size);
  r = allocate_vector<T>(size);
  
  // Zero vectors. 
  zero_vector(u[0], size); zero_vector(u[1], size);
  zero_vector(y[0], size); zero_vector(y[1], size);
  zero_vector(d, size);
  zero_vector(v, size);
  zero_vector(w, size); 
  zero_vector(r, size);

  // Initialize values.
  alpha = beta = eta = sigma = rho = rho_new = 0.0;
  tau = theta = c = 0;
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));

  // 1. r = b - Ax. 
  // Take advantage of initial guess in phi. Use v as a temp.
  (*matrix_vector)(v, phi, extra_info); invif.ops_count++;
  cxpayz(phi0, -1.0, v, r, size); 

  // 2. Don't worry about it.

  // 3. w = r. y[0] = r.
  copy_vector(w, r, size);
  copy_vector(y[0], r, size);

  // 4. v = A y[0]. u[0] = v;
  zero_vector(v, size);
  (*matrix_vector)(v, y[0], extra_info); invif.ops_count++;
  copy_vector(u[0], v, size);

  // 5. rho = norm2 of r, tau = norm of r.
  rho = norm2sq(r,size);
  tau = sqrt(real(rho)); // rho is uniquely real here
  
  // 6. iterate till convergence
  for(k = 0; k< max_iter; k++) {
    
    // 7. sigma = <r, v>.
    sigma = dot(r, v, size);
    if (abs(sigma) == 0.) // breakdown.
      break;

    // 8. alpha = rho/sigma.
    alpha = rho/sigma;

    // 9. Inner iteration.
    for (j = 0; j < 2; j++)
    {
      // 10. Only need to compute y[1] and u[1] every other time.
      if (j == 1)
      {
        // 11. y[1] = y[0] - alpha*v
        cxpayz(y[0], -alpha, v, y[1], size);

        // 12. u[1] = A y[1].
        zero_vector(u[1], size);
        (*matrix_vector)(u[1], y[1], extra_info); invif.ops_count++;
      }

      // 13. Used to set a pessimistic bound on r.
      // Should check if wasted iterations are worth carrying around r.
      m = 2*k+j;

      // 14. w -= alpha*u[j]
      caxpy(-alpha, u[j], w, size);

      // 15. d = y[j] + (theta*theta*eta/alpha) * d... wut.
      cxpay(y[j], theta*theta*eta/alpha, d, size);

      // 16. theta = |w|/tau, c = 1/sqrt(1 + theta*theta),
      //     tau = tau*theta*c, eta = c*c*alpha. okay.
      theta = sqrt(norm2sq(w, size))/tau;
      c = 1.0/sqrt(1.0 + theta*theta);
      tau = tau*theta*c;
      eta = c*c*alpha; 

      // 18. phi += eta*d. 
      // (Could add r -= eta* A d, we'd need to carry around A*d,
      //   and we'd need to update A*d with u[1].)
      caxpy(eta, d, phi, size);

      // 19. Check (pessimistic) error bound.
      print_verbosity_resid(verb, "TFQMR", k+1, invif.ops_count, tau*sqrt((double)m+1)/bsqrt); 
      if (tau*sqrt((double)m+1) < eps*bsqrt)
      {
        converged = true;
        break; // converged!
      }

    } // end inner iteration.

    if (converged) { break; }

    // 20. Check for breakdown.
    if (abs(rho) == 0.)
      break;

    // 21. rho_new = dot(r,w), beta = rho_new/rho; rho = rho_new.
    rho_new = dot(r, w, size);
    beta = rho_new/rho;
    rho = rho_new;

    // 22. y[0] = w + beta*y[1]
    cxpayz(w, beta, y[1], y[0], size);

    // 23. u[0] = A y[0].
    zero_vector(u[0], size);
    (*matrix_vector)(u[0], y[0], extra_info); invif.ops_count++;

    // 24. v = u[0] + beta*u[1] + beta*beta*v;
    caxpbypcz(T(1.0), u[0], beta, u[1], beta*beta, v, size);
    
  }

  if (sigma == 0.0 || rho == 0.0) // breakdown.
  {
    invif.success = false; 
  }
  else if(k == max_iter-1) {
    //printf("CG: Failed to converge iter = %d, rsq = %e\n", k,rsq);
    invif.success = false;
  }
  else
  {
     //printf("CG: Converged in %d iterations.\n", k);
     invif.success = true;
  }
	k++; 
  
  // Check the true residual.
  zero_vector(v, size);
  (*matrix_vector)(v,phi,extra_info); invif.ops_count++; 
  truersq = diffnorm2sq(v, phi0, size);
  
  // Free all the things!
  deallocate_vector(&u[0]);
  deallocate_vector(&u[1]);
  deallocate_vector(&y[0]);
  deallocate_vector(&y[1]);
  deallocate_vector(&d);
  deallocate_vector(&v);
  deallocate_vector(&w);
  deallocate_vector(&r);
  
  print_verbosity_summary(verb, "TFQMR", invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);

  //  printf("# CG: Converged iter = %d, rsq = %e, truersq = %e\n",k,rsq,truersq);
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = "TFQMR";
  return invif; // Convergence 

} 


// Performs tfqmr(restart_freq) with restarts when restart_freq is hit.
template <typename T>
inversion_info minv_vector_tfqmr_restart(T *phi, T *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb = nullptr)
{
  using Real = typename RealReducer<T>::type;

  int iter; // counts total number of iterations.
  int ops_count;
  inversion_info invif;
  Real bsqrt = sqrt(norm2sq(phi0, size));
  
  inversion_verbose_struct verb_rest;
  shuffle_verbosity_restart(&verb_rest, verb);
  
  stringstream ss;
  ss << "TFQMR(" << restart_freq << ")";
  
  iter = 0; ops_count = 0; 
  do
  {
    invif = minv_vector_tfqmr(phi, phi0, size, min(max_iter, restart_freq), res, matrix_vector, extra_info, &verb_rest);
    iter += invif.iter; ops_count += invif.ops_count;
    
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

/*


Copyright (c) 2016, C.T. Kelley
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

*/
