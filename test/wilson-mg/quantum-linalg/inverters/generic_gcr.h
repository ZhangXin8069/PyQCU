// Copyright (c) 2017 Evan S Weinberg
// GCR inverter.
// Solves lhs = A^(-1) rhs with GCR.
// Makes no assumptions about the structure of the matrix. 

#ifndef QLINALG_INVERTER_GCR
#define QLINALG_INVERTER_GCR

#include <string>
#include <sstream>
#include <vector>
#include <complex>

using std::complex;
using std::stringstream;
using std::vector;

#include "inverter_struct.h"
#include "../verbosity/verbosity.h"

// Taken from section 6.9 of Saad, 2nd Edition.
template <typename T>
inversion_info minv_vector_gcr(T *phi, T *phi0, int size, int max_iter, double eps, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb = nullptr)
{
  using Real = typename RealReducer<T>::type;

  // Initialize vectors.
  T *x, *r, *Ar, *p, *Ap;
  vector<T*> p_store, Ap_store; // GCR requires explicit reorthogonalization against old search vectors.
  Real rsq, bsqrt, truersq;
  T alpha, beta_ij;
  int k,i,ii;
  inversion_info invif;

  // Allocate memory.
  x = allocate_vector<T>(size);
  r = allocate_vector<T>(size);
  Ar = allocate_vector<T>(size);
  p = allocate_vector<T>(size);
  Ap = allocate_vector<T>(size);
  
  // Zero vectors. 
  zero_vector(p, size);  zero_vector(r, size);
  zero_vector(Ap, size); zero_vector(Ar, size);

  // Initialize values.
  rsq = 0.0; bsqrt = 0.0; truersq = 0.0;
  
  // copy_vector initial guess into solution.
  copy_vector(x, phi, size);
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // 1. r_0 = b - Ax_0. x is phi, the initial guess.
  (*matrix_vector)(p, x, extra_info); invif.ops_count++; // Put Ax_0 into p, temp.
  cxpayz(phi0, -1.0, p, r, size);
  
  // 2. p_0 = r_0.
  copy_vector(p, r, size);
  
  // 3. Compute A p_0.
  (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;

  // iterate until convergence
  for(k = 0; k< max_iter; k++) {
    
    // If we've hit here, push the latest p, Ap to storage.
    p_store.push_back(p);
    Ap_store.push_back(Ap);
    
    // 4. alpha = <r, Ap_k>/<Ap_k, Ap_k>
    alpha = dot(Ap, r, size)/norm2sq(Ap, size);
    
    // 5. x = x + alpha p_k
    caxpy(alpha, p, x, size);
    
    // 6. r = r - alpha Ap_k
    caxpy(-alpha, Ap, r, size);
    
    // Compute norm.
    rsq = norm2sq(r, size);
    
    print_verbosity_resid(verb, "GCR", k+1, invif.ops_count, sqrt(rsq)/bsqrt); 
    
    // Check convergence. 
    if (sqrt(rsq) < eps*bsqrt || k == max_iter-1) {
      //        printf("Final rsq = %g\n", rsqNew);
      break;
    }
    
    // 7. Compute Ar.
    zero_vector(Ar, size);
    (*matrix_vector)(Ar, r, extra_info); invif.ops_count++;
    
    // 8. b_ij = -<Ap_i, Ar_{j+1}>/<Ap_i, Ap_i> for i = 0, ..., j
    // 9. p_{j+1} = r_{j+1} + sum_i=0^j b_ij p_i
    // 10. Ap_{j+1} = Ar_{j+1} + sum_i=0^j b_ij Ap_i
    p = allocate_vector<T>(size);  copy_vector(p, r, size);
    Ap = allocate_vector<T>(size); copy_vector(Ap, Ar, size);
    for (ii = 0; ii <= k; ii++)
    {
      beta_ij = -dot(Ap_store[ii], Ar, size)/norm2sq(Ap_store[ii], size);
      caxpy(beta_ij, p_store[ii], p, size);
      caxpy(beta_ij, Ap_store[ii], Ap, size);
    }
  } 
    
  if(k == max_iter-1) {
    //printf("CG: Failed to converge iter = %d, rsq = %e\n", k,rsq);
    invif.success = false;
    //return 0;// Failed convergence 
  }
  else
  {
     invif.success = true;
     //printf("CG: Converged in %d iterations.\n", k);
  }
  k++;
  
  // Check true residual.
  zero_vector(p,size);
  (*matrix_vector)(p,x,extra_info); invif.ops_count++;
  truersq = diffnorm2sq(p, phi0, size);
  
  // copy_vector solution into phi.
  copy_vector(phi, x, size);
  
  // Free all the things!
  deallocate_vector(&x);
  deallocate_vector(&r);
  deallocate_vector(&Ar);
  int l = p_store.size();
  for (i = 0; i < l; i++)
  {
    deallocate_vector(&p_store[i]);
    deallocate_vector(&Ap_store[i]);
  }

  print_verbosity_summary(verb, "GCR", invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = "GCR";
  return invif; // Convergence 
} 

// Performs GCR(restart_freq) with restarts when restart_freq is hit.
// This may be sloppy, but it works.
template <typename T>
inversion_info minv_vector_gcr_restart(T *phi, T *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb = nullptr)
{
  using Real = typename RealReducer<T>::type;

  int iter; // counts total number of iterations.
  int ops_count; 
  inversion_info invif;
  Real bsqrt = sqrt(norm2sq(phi0, size));
  
  stringstream ss;
  ss << "GCR(" << restart_freq << ")";
  
  inversion_verbose_struct verb_rest;
  shuffle_verbosity_restart(&verb_rest, verb);
  
  iter = 0; ops_count = 0; 
  do
  {
    invif = minv_vector_gcr(phi, phi0, size, min(max_iter, restart_freq), res, matrix_vector, extra_info, &verb_rest);
    iter += invif.iter;
    ops_count += invif.ops_count; 
    
    print_verbosity_restart(verb, ss.str(), iter, ops_count, sqrt(invif.resSq)/bsqrt);
  }
  while (iter < max_iter && invif.success == false && sqrt(invif.resSq)/bsqrt > res);
  
  invif.iter = iter;
  invif.ops_count = ops_count; 
  
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
  
  print_verbosity_summary(verb, ss.str(), invif.success, iter, invif.ops_count, sqrt(invif.resSq)/bsqrt);
  
  return invif;
}



#endif