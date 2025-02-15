// Copyright (c) 2017 Evan S Weinberg
// BiCGStab inverter.
// Solves lhs = A^(-1) rhs with bicgstab
// Makes no assumptions about the matrix.

#ifndef QLINALG_INVERTER_BICGSTAB
#define QLINALG_INVERTER_BICGSTAB

#include <string>
#include <sstream>
#include <complex>
using std::complex;
using std::stringstream;

#include "inverter_struct.h"
#include "../verbosity/verbosity.h"


// Solves lhs = A^(-1) rhs using bicgstab
template <typename T>
inversion_info minv_vector_bicgstab(T *phi, T *phi0, int size, int max_iter, double eps, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb = nullptr)
{
  using Real = typename RealReducer<T>::type;
// BICGSTAB solutions to Mphi = b 
  //  see www.mcs.anl.gov/papers/P3039-0912.pdf
  // "Analysis and Practical Use of Flexible BiCGStab

  // Initialize vectors.
  T *r, *r0, *p, *Ap, *s, *As; 
  T rho, rhoNew, alpha, beta, omega;
  Real rsq, bsqrt, truersq; 
  int k;
  inversion_info invif;

  // Allocate memory.
  r = allocate_vector<T>(size);
  r0 = allocate_vector<T>(size);
  p = allocate_vector<T>(size);
  Ap = allocate_vector<T>(size);
  s = allocate_vector<T>(size);
  As = allocate_vector<T>(size);
  
  // Zero vectors. 
  zero_vector(r, size);
  zero_vector(r0, size);
  zero_vector(p, size);
  zero_vector(Ap, size); 
  zero_vector(s, size);
  zero_vector(As, size);

  // Initialize values.
  rsq = 0.0; bsqrt = 0.0; truersq = 0.0;
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));

  // 1. r = b - Ax. , r0 = arbitrary (use r).
  // Take advantage of initial guess in phi.
  (*matrix_vector)(Ap, phi, extra_info); invif.ops_count++;
  cxpayz(phi0, -1.0, Ap, r, size); // r is a temporary
  copy_vector(r0, r, size);
  
  // 2. p = r
  copy_vector(p, r, size); 
  
  // 2a. Initialize rho = <r, r0>.
  rho = dot(r0, r, size);
  
  // 2b. Initialize Ap.
  zero_vector(Ap, size);

  (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;
  
  // 3. iterate till convergence
  for(k = 0; k< max_iter; k++) {
    
    // 4. alpha = <r0, r>/<r0, Ap>
    alpha = rho/dot(r0, Ap, size);
    
    // 5. s = r - alpha Ap
    cxpayz(r, -alpha, Ap, s, size);
    
    // 6. Compute As, w = <s, As>/(As, As)
    (*matrix_vector)(As, s, extra_info); invif.ops_count++;
    omega = dot(As, s, size)/dot(As, As, size);
    
    // 7. Update phi = phi + alpha*p + omega*s
    caxpbypz(alpha, p, omega, s, phi, size);
    
    // 8. Update r = s - omega*As
    cxpayz(s, -omega, As, r, size);
    
    // 8a. If ||r|| is sufficiently small, quit.
    rsq = norm2sq(r, size);
    print_verbosity_resid(verb, "BiCGStab", k+1, invif.ops_count, sqrt(rsq)/bsqrt);
    
    if (sqrt(rsq) < eps*bsqrt || k==max_iter-1)
    {
      break;
    }
    
    // 9. rhoNew = <r0, r>.
    rhoNew = dot(r0, r, size);
    beta = rhoNew/rho*(alpha/omega);
    rho = rhoNew;
    
    // 10. Update p = r + beta*p - omega*beta*Ap
    caxpbypcz(T(1.0), r, -omega*beta, Ap, beta, p, size);
    
    
    zero_vector(Ap, size);
    (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;
    
    
  }
  
  if(k == max_iter-1) {
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
  (*matrix_vector)(Ap,phi,extra_info); invif.ops_count++;
  truersq = diffnorm2sq(Ap, phi0, size);
  
  // Free all the things!
  deallocate_vector(&r);
  deallocate_vector(&r0);
  deallocate_vector(&p);
  deallocate_vector(&Ap);
  deallocate_vector(&s);
  deallocate_vector(&As);
  
  print_verbosity_summary(verb, "BiCGStab", invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);

  //  printf("# CG: Converged iter = %d, rsq = %e, truersq = %e\n",k,rsq,truersq);
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = "BiCGStab";
  return invif; // Convergence 

}

// Performs BiCGStab with restarts when restart_freq is hit.
// This may be sloppy, but it works.
template <typename T>
inversion_info minv_vector_bicgstab_restart(T *phi, T *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb = nullptr)
{
  using Real = typename RealReducer<T>::type;
  int iter; // counts total number of iterations.
  int ops_count; 
  inversion_info invif;
  Real bsqrt = sqrt(norm2sq(phi0, size));
  
  stringstream ss;
  ss << "BiCGStab(" << restart_freq << ")";
  
  inversion_verbose_struct verb_rest;
  shuffle_verbosity_restart(&verb_rest, verb);
  
  iter = 0; ops_count = 0; 
  do
  {
    invif = minv_vector_bicgstab(phi, phi0, size, min(max_iter, restart_freq), res, matrix_vector, extra_info, &verb_rest);
    iter += invif.iter; ops_count += invif.ops_count; 
    
    print_verbosity_restart(verb, ss.str(), iter, ops_count, sqrt(invif.resSq)/bsqrt);
  }
  while (iter < max_iter && invif.success == false && sqrt(invif.resSq)/bsqrt > res);
  
  invif.iter = iter;
  invif.ops_count = ops_count; 
  
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