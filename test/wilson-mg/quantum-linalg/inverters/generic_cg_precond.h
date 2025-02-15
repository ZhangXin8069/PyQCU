// Copyright (c) 2017 Evan S Weinberg
// stationary preconditioned CG inverter.
// Assumes the operator and the preconditioner is hermitian positive definite.


#ifndef QLINALG_INVERTER_PCG
#define QLINALG_INVERTER_PCG

#include <string>
#include <sstream>
#include <vector>
#include <complex>

using std::complex;
using std::stringstream;
using std::vector;

#ifndef QLINALG_FCN_POINTER
#define QLINALG_FCN_POINTER
typedef void (*matrix_op_real)(double*,double*,void*);
typedef void (*matrix_op_cplx)(complex<double>*,complex<double>*,void*);
#endif

#include "inverter_struct.h"
#include "../verbosity/verbosity.h"


// Preconditioned CG
inversion_info minv_vector_cg_precond(double  *phi, double  *phi0, int size, int max_iter, double eps, void (*matrix_vector)(double*,double*,void*), void* extra_info, void (*precond_matrix_vector)(double*,double*,int,void*,inversion_verbose_struct*), void* precond_info, inversion_verbose_struct* verbosity = 0);

inversion_info minv_vector_cg_precond(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double eps, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, void (*precond_matrix_vector)(complex<double>*,complex<double>*,int,void*,inversion_verbose_struct*), void* precond_info, inversion_verbose_struct* verbosity = 0);

// Restarted Preconditioned CG
inversion_info minv_vector_cg_precond_restart(double  *phi, double  *phi0, int size, int max_iter, double eps, int restart_freq, void (*matrix_vector)(double*,double*,void*), void* extra_info, void (*precond_matrix_vector)(double*,double*,int,void*,inversion_verbose_struct*), void* precond_info, inversion_verbose_struct* verbosity = 0);

inversion_info minv_vector_cg_precond_restart(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double eps, int restart_freq, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, void (*precond_matrix_vector)(complex<double>*,complex<double>*,int,void*,inversion_verbose_struct*), void* precond_info, inversion_verbose_struct* verbosity = 0);


// Solves lhs = A^(-1) rhs
inversion_info minv_vector_cg_precond(double  *phi, double  *phi0, int size, int max_iter, double eps, void (*matrix_vector)(double*,double*,void*), void* extra_info, void (*precond_matrix_vector)(double*,double*,int,void*,inversion_verbose_struct*), void* precond_info, inversion_verbose_struct* verb)
{
  // Initialize vectors.
  double *r, *p, *Ap, *z;
  double alpha, beta, zdotr, zdotr_new, rsq, bsqrt, truersq;
  int k;
  inversion_info invif;
  
  // For preconditioning verbosity.
  inversion_verbose_struct verb_prec;
  shuffle_verbosity_precond(&verb_prec, verb);

  // Allocate memory.
  r = allocate_vector<double>(size);
  p = allocate_vector<double>(size);
  Ap = allocate_vector<double>(size);
  z = allocate_vector<double>(size);
  
  // Zero vectors. 
  zero_vector(r, size);  zero_vector(z, size);
  zero_vector(p, size); zero_vector(Ap, size);

  // Initialize values.
  rsq = zdotr = zdotr_new = bsqrt = truersq = 0.0;
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // 1. r_0 = b - Ax_0. 
  (*matrix_vector)(p, phi, extra_info); invif.ops_count++; // Put Ax_0 into p, temp.
  cxpayz(phi0, -1.0, p, r, size); // r = b - Ax_0, b -> phi0, p -> x0
  
  // 2. z_0 = M^(-1) r_0.
  (*precond_matrix_vector)(z, r, size, precond_info, &verb_prec); 
  
  // 3. p_0 = z_0, Compute A p_0
  copy_vector(p, z, size); 
  (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;

  // Compute zdotr
  zdotr = dot(z, r, size);

  // iterate until convergence
  for(k = 0; k< max_iter; k++) {
    
    // 4. alpha = <z, r>/<p , Ap>
    alpha = zdotr/dot(p, Ap, size);
    
    // 5. phi = phi + alpha p_k
    caxpy(alpha, p, phi, size);
    
    // 6. r = r - alpha Ap_k
    caxpy(-alpha, Ap, r, size);
    
    // Compute norm.
    rsq = norm2sq(r, size);
    
    print_verbosity_resid(verb, "PCG", k+1, invif.ops_count, sqrt(rsq)/bsqrt); 
    
    //printf("Rel residual: %.8e\n", sqrt(rsq)/bsqrt); fflush(stdout);
    
    // Check convergence. 
    if (sqrt(rsq) < eps*bsqrt || k==max_iter-1) {
      //        printf("Final rsq = %g\n", rsqNew);
      break;
    }
    
    // 7. z = M^(-1) r;
    zero_vector(z, size); 
    (*precond_matrix_vector)(z, r, size, precond_info, &verb_prec); 
    
    // 8. beta = r dot z (new) / r dot z
    zdotr_new = dot(r, z, size);
    beta = zdotr_new / zdotr;
    zdotr = zdotr_new;

    // 9. p = z + beta p
    cxpay(z, beta, p, size);

    // 10. Compute the new Ap.
    zero_vector(Ap, size);
    (*matrix_vector)(Ap, p, extra_info);

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
  (*matrix_vector)(p,phi,extra_info); invif.ops_count++;
  truersq = diffnorm2sq(p, phi0, size);
  
  // Free all the things!
  deallocate_vector(&r);
  deallocate_vector(&z);
  deallocate_vector(&p);
  deallocate_vector(&Ap);
  
  print_verbosity_summary(verb, "PCG", invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = "Preconditioned CG";
  return invif; // Convergence 
} 

// Performs VPGCR(restart_freq) with restarts when restart_freq is hit.
// This may be sloppy, but it works.
inversion_info minv_vector_cg_precond_restart(double  *phi, double  *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(double*,double*,void*), void* extra_info, void (*precond_matrix_vector)(double*,double*,int,void*,inversion_verbose_struct*), void* precond_info, inversion_verbose_struct* verb)
{
  int iter; // counts total number of iterations.
  int ops_count; 
  inversion_info invif;
  double bsqrt = sqrt(norm2sq(phi0, size));
  
  stringstream ss;
  ss << "Preconditioned Restarted CG(" << restart_freq << ")";

  inversion_verbose_struct verb_rest;
  shuffle_verbosity_restart(&verb_rest, verb);
  
  iter = 0; ops_count = 0; 
  do
  {
    invif = minv_vector_cg_precond(phi, phi0, size, min(max_iter, restart_freq), res, matrix_vector, extra_info, precond_matrix_vector, precond_info, &verb_rest);
    iter += invif.iter; ops_count += invif.ops_count; 
    
    print_verbosity_restart(verb, ss.str(), iter, ops_count, sqrt(invif.resSq)/bsqrt);
  }
  while (iter < max_iter && invif.success == false && sqrt(invif.resSq)/bsqrt > res);
  
  invif.iter = iter; invif.ops_count = ops_count; 

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

inversion_info minv_vector_cg_precond(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double eps, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, void (*precond_matrix_vector)(complex<double>*,complex<double>*,int,void*,inversion_verbose_struct*), void* precond_info, inversion_verbose_struct* verb)
{

  // Initialize vectors.
  complex<double> *r, *p, *Ap, *z;
  complex<double> alpha, beta, zdotr, zdotr_new;
  double rsq, bsqrt, truersq;
  int k;
  inversion_info invif;
  
  // For preconditioning verbosity.
  inversion_verbose_struct verb_prec;
  shuffle_verbosity_precond(&verb_prec, verb);

  // Allocate memory.
  r = allocate_vector<complex<double> >(size);
  p = allocate_vector<complex<double> >(size);
  Ap = allocate_vector<complex<double> >(size);
  z = allocate_vector<complex<double> >(size);
  
  // Zero vectors. 
  zero_vector(r, size);  zero_vector(z, size);
  zero_vector(p, size); zero_vector(Ap, size);

  // Initialize values.
  zdotr = zdotr_new = rsq = bsqrt = truersq = 0.0;
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // 1. r_0 = b - Ax_0. 
  (*matrix_vector)(p, phi, extra_info); invif.ops_count++; // Put Ax_0 into p, temp.
  cxpayz(phi0, -1.0, p, r, size); // r = b - Ax_0, b -> phi0, p -> x0
  
  // 2. z_0 = M^(-1) r_0.
  (*precond_matrix_vector)(z, r, size, precond_info, &verb_prec); 
  
  // 3. p_0 = z_0, Compute A p_0
  copy_vector(p, z, size); 
  (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;

  // Compute zdotr
  zdotr = dot(z, r, size);

  // iterate until convergence
  for(k = 0; k< max_iter; k++) {
    
    // 4. alpha = <z, r>/<p , Ap>
    alpha = zdotr/dot(p, Ap, size);
    
    // 5. phi = phi + alpha p_k
    caxpy(alpha, p, phi, size);
    
    // 6. r = r - alpha Ap_k
    caxpy(-alpha, Ap, r, size);
    
    // Compute norm.
    rsq = norm2sq(r, size);
    
    print_verbosity_resid(verb, "PCG", k+1, invif.ops_count, sqrt(rsq)/bsqrt); 
    
    //printf("Rel residual: %.8e\n", sqrt(rsq)/bsqrt); fflush(stdout);
    
    // Check convergence. 
    if (sqrt(rsq) < eps*bsqrt || k==max_iter-1) {
      //        printf("Final rsq = %g\n", rsqNew);
      break;
    }
    
    // 7. z = M^(-1) r;
    zero_vector(z, size); 
    (*precond_matrix_vector)(z, r, size, precond_info, &verb_prec); 
    
    // 8. beta = r dot z (new) / r dot z
    zdotr_new = dot(r, z, size);
    beta = zdotr_new / zdotr;
    zdotr = zdotr_new;

    // 9. p = z + beta p
    cxpay(z, beta, p, size);

    // 10. Compute the new Ap.
    zero_vector(Ap, size);
    (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;

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
  (*matrix_vector)(p,phi,extra_info); invif.ops_count++;
  truersq = diffnorm2sq(p, phi0, size);
  
  // Free all the things!
  deallocate_vector(&r);
  deallocate_vector(&z);
  deallocate_vector(&p);
  deallocate_vector(&Ap);
  
  print_verbosity_summary(verb, "PCG", invif.success, k, invif.ops_count, sqrt(truersq)/bsqrt);
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = "Preconditioned CG";
  return invif; // Convergence 
} 

// Performs CG(restart_freq) with restarts when restart_freq is hit.
// This may be sloppy, but it works.
inversion_info minv_vector_cg_precond_restart(complex<double>  *phi, complex<double>  *phi0, int size, int max_iter, double res, int restart_freq, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, void (*precond_matrix_vector)(complex<double>*,complex<double>*,int,void*,inversion_verbose_struct*), void* precond_info, inversion_verbose_struct* verb)
{
  int iter; // counts total number of iterations.
  int ops_count; 
  inversion_info invif;
  double bsqrt = sqrt(norm2sq(phi0, size));
  
  stringstream ss;
  ss << "Preconditioned CG(" << restart_freq << ")";

  inversion_verbose_struct verb_rest;
  shuffle_verbosity_restart(&verb_rest, verb);
  
  iter = 0; ops_count = 0; 
  do
  {
    invif = minv_vector_cg_precond(phi, phi0, size, min(max_iter, restart_freq), res, matrix_vector, extra_info, precond_matrix_vector, precond_info, &verb_rest);
    iter += invif.iter; ops_count += invif.ops_count; 
    
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