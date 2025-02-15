// Copyright (c) 2017 Evan S Weinberg
// BiCGStab inverter.
// Solves lhs = A^(-1) rhs with bicgstab-l
// Makes no assumptions about the matrix.

// Defined in the paper "BICGSTAB(L) for linear equations involving unsymmetric matrices with complex spectrum"
// G. Sleijpen, D. Fokkema, 1993. 
// Based on Kate Clark's implementation in CPS, src file src/util/dirac_op/d_op_wilson_types/bicgstab.C

#ifndef QLINALG_INVERTER_BICGSTAB_L
#define QLINALG_INVERTER_BICGSTAB_L

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

// Solves lhs = A^(-1) rhs using bicgstab-l.
template <typename T>
inversion_info minv_vector_bicgstab_l(T *phi, T *phi0, int size, int max_iter, double eps, int l, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb = nullptr)
{
  using Real = typename RealReducer<T>::type;

  // Initialize vectors.
  T *r0, **r, **u;
  T rho0, rho1, alpha, omega, beta;
  T *gamma, *gamma_prime, *gamma_prime_prime, **tau; 
  Real *sigma, bsqrt, truersq; 
  int k,i,j;
  inversion_info invif;
  
  // Prepare verbosity.
  stringstream ss;
  ss << "BiCGStab-" << l;

  // Allocate memory.
  r0 = allocate_vector<T>(size);
  r = new T*[l+1];
  u = new T*[l+1];
  for (i = 0; i < l+1; i++)
  {
    r[i] = allocate_vector<T>(size);
    u[i] = allocate_vector<T>(size); 
  }
  
  sigma = new Real[l+1];
  gamma = new T[l+1];
  gamma_prime = new T[l+1];
  gamma_prime_prime = new T[l+1];
  tau = new T*[l+1];
  for (i = 0; i < l+1; i++)
  {
    tau[i] = new T[l+1];
  }
  
  // Zero vectors. 
  zero_vector(r0, size);
  for (i = 0; i < l+1; i++)
  {
    zero_vector(r[i], size);
    zero_vector(u[i], size);
  }
  
  zero_vector(sigma, l+1);
  zero_vector(gamma, l+1);
  zero_vector(gamma_prime, l+1);
  zero_vector(gamma_prime_prime, l+1);
  for (i = 0; i < l+1; i++)
  {
    zero_vector(tau[i], l+1);
  }

  // Initialize values.
  /*rsq = 0.0; */bsqrt = 0.0; truersq = 0.0;
  rho0 = 1.;
  alpha = 0.;
  omega = 1.; 
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));

  // 1. r[0]0 = b - Ax. r0 = r[0]. sigma[0] = ||r[0]||^2
  // Take advantage of initial guess in phi. Use u[0] as a tmp.
  zero_vector(u[0], size); 
  (*matrix_vector)(u[0], phi, extra_info); invif.ops_count++;
  cxpayz(phi0, -1.0, u[0], r[0], size); // r0 = b - Ax0, phi0 -> b, u[0] -> Ax0.
  sigma[0] = norm2sq(r[0], size);
  copy_vector(r0, r[0], size);
  zero_vector(u[0], size);
  
  
  // 2. iterate till convergence
  // This was written with a bit of a headache, so my conventions in my comments are
  // all over the place... sorry about that...
  for(k = 0; k < max_iter; k+=l) {
    
    // rho0 = -omega*rho0;
    rho0 *= -omega; 
    
    // BiCG part.
    for (j = 0; j < l; j++)
    {
      // rho1 = <r0, r_j>, beta = alpha*rho1/rho0, rho0 = rho1
      rho1 = dot(r0, r[j], size);
      beta = alpha*rho1/rho0;
      rho0 = rho1;
      // for i = 0 .. j, u[i] = r[i] - beta*u[i];
      for (i = 0; i <= j; i++)
      {
        cxpay(r[i], -beta, u[i], size);
      }
      // u[j+1] = A u[j];
      zero_vector(u[j+1], size); 
      (*matrix_vector)(u[j+1], u[j], extra_info); invif.ops_count++;
      
      // alpha = rho0/<r0, u[j+1]>
      alpha = rho0/dot(r0, u[j+1], size);
      
      // for i = 0 .. j, r[i] = r[i] - alpha u[i+1]
      for (i = 0; i <= j; i++)
      {
        caxpy(-alpha, u[i+1], r[i], size);
      }
      
      // r[j+1] = A r[j], x = x + alpha*u[0]
      (*matrix_vector)(r[j+1], r[j], extra_info); invif.ops_count++;
      caxpy(alpha, u[0], phi, size);
    } // End BiCG part.
    
    // MR part. Really just modified Gram-Schmidt.
    // I could probably write this in terms of the normalize, orthonormalize functions I have, but eh.
    // The algorithm definition uses the byproducts of the Gram-Schmidt to update x, etc,
    // Don't want to disentangle it at the moment,
    for (j = 1; j <= l; j++)
    {
      for (i = 1; i < j; i++)
      {
        // tau_ij = <r_i,r_j>/sigma_i
        tau[i][j] = dot(r[i], r[j], size)/sigma[i];
        
        // r_j = r_j - tau_ij r[i];
        caxpy(-tau[i][j], r[i], r[j], size);
      }
        
      // sigma_j = r_j^2, gamma'_j = <r_0, r_j>/sigma_j
      sigma[j] = norm2sq(r[j], size);
      gamma_prime[j] = dot(r[j],r[0], size)/sigma[j];
    }
        
    // gamma[l] = gamma'_l, omega = gamma[l]
    gamma[l] = gamma_prime[l];
    omega = gamma[l];
    
    // gamma = T^(-1) gamma_prime. Check paper for defn of T.
    for (j = l-1; j > 0; j--)
    {
      // Internal def: gamma[j] = gamma'_j - \sum_{i = j+1 to l} tau_ji gamma_i
      gamma[j] = gamma_prime[j];
      for (i = j+1; i <= l; i++)
      {
        gamma[j] = gamma[j] - tau[j][i]*gamma[i];
      }
    }
    
    // gamma'' = T S gamma. Check paper for defn of S.
    for (j = 1; j < l; j++)
    {
      gamma_prime_prime[j] = gamma[j+1];
      for (i = j+1; i < l; i++)
      {
        gamma_prime_prime[j] = gamma_prime_prime[j] + tau[j][i]*gamma[i+1];
      }
    }
    
    // Update x, r, u.
    // x = x + gamma_1 r_0, r_0 = r_0 - gamma'_l r_l, u_0 = u_0 - gamma_l u_l
    caxpyBzpx(gamma[1], r[0], phi, -gamma_prime[l], r[l], size);
    caxpy(-gamma[l], u[l], u[0], size);
           
    // for j = 1 .. l-1: u[0] -= gamma_j u[j], phi += gamma''_j r[j], r[0] -= gamma'_j r[j].
    for (j = 1; j < l; j++)
    {
      caxpy(-gamma[j], u[j], u[0], size);
      caxpyBxpz(gamma_prime_prime[j], r[j], phi, -gamma_prime[j], r[0], size);
    }
    
    // sigma[0] = r_0^2. This is rsq in my other codes. 
    sigma[0] = norm2sq(r[0], size);
    print_verbosity_resid(verb, ss.str(), k+l, invif.ops_count, sqrt(sigma[0])/bsqrt);
    
    // Check for convergence.
    if (sqrt(sigma[0]) < eps*bsqrt)
    {
      //rsq = sigma[0];
      break;
    }
  }
  
  // Weird counting conventions give me headaches.
  if(k >= max_iter-l) {
    //printf("CG: Failed to converge iter = %d, rsq = %e\n", k,rsq);
    invif.success = false;
  }
  else
  {
     //printf("CG: Converged in %d iterations.\n", k);
     invif.success = true;
  }
	k+=l; 
  
  // Check the true residual. Use u[0] as a tmp.
  (*matrix_vector)(u[0],phi,extra_info); invif.ops_count++; 
  truersq = diffnorm2sq(u[0], phi0, size);
  
  // Free all the things!
  deallocate_vector(&r0);
  for (i = 0; i < l+1; i++)
  {
    deallocate_vector(&r[i]);
    deallocate_vector(&u[i]);
  }
  delete[] r;
  delete[] u;
  
  delete[] sigma; 
  delete[] gamma;
  delete[] gamma_prime;
  delete[] gamma_prime_prime;
  for (i = 0; i < l+1; i++)
  {
    delete[] tau[i];
  }
  delete[] tau;
  
  print_verbosity_summary(verb, ss.str(), invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);

  //  printf("# CG: Converged iter = %d, rsq = %e, truersq = %e\n",k,rsq,truersq);
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = ss.str();
  return invif; // Convergence 

} 

// Performs BiCGStab-l(restart_freq) with restarts when restart_freq is hit.
template <typename T>
inversion_info minv_vector_bicgstab_l_restart(T *phi, T *phi0, int size, int max_iter, double res, int restart_freq, int l, void (*matrix_vector)(T*,T*,void*), void* extra_info, inversion_verbose_struct* verb = nullptr)
{
  using Real = typename RealReducer<T>::type;
  int iter; // counts total number of iterations.
  int ops_count;
  inversion_info invif;
  Real bsqrt = sqrt(norm2sq(phi0, size));
  
  inversion_verbose_struct verb_rest;
  shuffle_verbosity_restart(&verb_rest, verb);
  
  stringstream ss;
  ss << "BiCGStab-" << l << "(" << restart_freq << ")";
  
  iter = 0; ops_count = 0; 
  do
  {
    invif = minv_vector_bicgstab_l(phi, phi0, size, min(max_iter, restart_freq), res, l, matrix_vector, extra_info, &verb_rest);
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