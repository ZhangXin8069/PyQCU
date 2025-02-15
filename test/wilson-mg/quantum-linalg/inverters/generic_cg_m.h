// Copyright (c) 2017 Evan S Weinberg
// Multshift CG inverter.
// Solves lhs[i] = (A+shift[i])^(-1) rhs with CG, solving
//   for all shifts at once. 
// Based on the definition in http://arxiv.org/pdf/hep-lat/9612014.pdf
// Assumes the matrix (A+shift[i]) is Hermitian (symmetric) positive
//   definite for all shift[i]'s. 

#ifndef QLINALG_INVERTER_CG_M
#define QLINALG_INVERTER_CG_M

#include <string>
#include <complex>

using std::complex;

#ifndef QLINALG_FCN_POINTER
#define QLINALG_FCN_POINTER
typedef void (*matrix_op_real)(double*,double*,void*);
typedef void (*matrix_op_cplx)(complex<double>*,complex<double>*,void*);
#endif

#include "inverter_struct.h"
#include "../verbosity/verbosity.h"

inversion_info minv_vector_cg_m(double **phi, double *phi0, int n_shift, int size, int resid_freq_check, int max_iter, double eps, double* shifts, void (*matrix_vector)(double*,double*,void*), void* extra_info, bool worst_first = false, inversion_verbose_struct* verbosity = 0);
inversion_info minv_vector_cg_m(complex<double> **phi, complex<double>  *phi0, int n_shift, int size, int resid_freq_check, int max_iter, double eps, double* shifts, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, bool worst_first = false, inversion_verbose_struct* verbosity = 0);

// multishift CG starts with 0 initial guess, so a restarted version wouldn't work. 

// Solves lhs = A^(-1) rhs using multishift CG as defined in http://arxiv.org/pdf/hep-lat/9612014.pdf
// Assumes there are n_shift values in "shifts".
// If they are sorted s.t. the smallest shift is the smallest (worst-conditioned
//   solve), set worst_first = true. 
// resid_freq_check is how often to check the residual of other solutions.
// This lets us stop iterating on converged systems. 
inversion_info minv_vector_cg_m(double **phi, double *phi0, int n_shift, int size, int resid_freq_check, int max_iter, double eps, double* shifts, void (*matrix_vector)(double*,double*,void*), void* extra_info, bool worst_first, inversion_verbose_struct* verb)
{
  
  // Initialize vectors.
  double *r, *p, *Ap;
  double **p_s;
  double alpha, beta, beta_prev, rsq, rsqNew, bsqrt, truersq, tmp; 
  double *alpha_s, *beta_s, *zeta_s, *zeta_s_prev;
  int k,i,n;
  int n_shift_rem = n_shift; // number of systems to still iterate on. 
  int* mapping; // holds the mapping between vectors and the original vector ordering.
                // this is because some vectors may converge before others. 
  double* tmp_ptr; // temporary pointer for swaps.
  double tmp_dbl; // temporary double for swaps. 
  int tmp_int; // temporary int for swaps. 
  
  // Prepare an inversion_info for multiple residuals.
  inversion_info invif(n_shift); 

  // Allocate memory.
  alpha_s = new double[n_shift];
  beta_s = new double[n_shift];
  zeta_s = new double[n_shift];
  zeta_s_prev = new double[n_shift];
  
  p_s = new double*[n_shift];
  for (n = 0; n < n_shift; n++)
  {
    p_s[n] = allocate_vector<double>(size);
  }
  
  r = allocate_vector<double>(size);
  p = allocate_vector<double>(size);
  Ap = allocate_vector<double>(size);
  
  // Initialize mapping.
  mapping = new int[n_shift];
  for (n = 0; n < n_shift; n++)
  {
    mapping[n] = n; // All vectors are currently in order.
  }

  // Initialize values.
  rsq = 0.0; rsqNew = 0.0; bsqrt = 0.0; truersq = 0.0; k=0;
  for (n = 0; n < n_shift; n++)
  {
    // beta_0, zeta_0, zeta_-1
    beta_s[n] = zeta_s[n] = zeta_s_prev[n] = 1.0;
    // alpha_0. 
    alpha_s[n] = 0.0;
  }
  beta = 1.0; alpha = 0.0;

  // Zero vectors;
  zero_vector(r, size); 
  zero_vector(p, size); zero_vector(Ap, size);
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // There can't be an initial guess... though it is sort of possible, in reference to:
  // http://arxiv.org/pdf/0810.1081v1.pdf
  
  // 1. x_sigma = 0, r = p_sigma = b.
  for (n = 0; n < n_shift; n++)
  {
    copy_vector(p_s[n], phi0, size);
    zero_vector(phi[n], size);
  }
  copy_vector(p, phi0, size);
  copy_vector(r, phi0, size);
  
  // Compute Ap.
  zero_vector(Ap, size);
  (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;
  
  // Compute rsq.
  rsq = norm2sq(r, size);

  // iterate till convergence
  for(k = 0; k< max_iter; k++) {
    
    // 2. beta_i = - rsq / pAp. Which is a weird switch from the normal notation, but whatever.
    beta_prev = beta; 
    beta = -rsq/dot(p, Ap, size);
    //cout << "beta = " << beta << "\n";
    
    for (n = 0; n < n_shift_rem; n++)
    {
      // 3. Calculate beta_i^sigma, zeta_i+1^sigma according to 2.42 to 2.44.
      // zeta_{i+1}^sigma = complicated...
      tmp = zeta_s[n]; // Save zeta_i to pop into the prev zeta.
      zeta_s[n] = (zeta_s[n]*zeta_s_prev[n]*beta_prev)/(beta*alpha*(zeta_s_prev[n]-zeta_s[n]) + zeta_s_prev[n]*beta_prev*(1.0-shifts[n]*beta));
      zeta_s_prev[n] = tmp; 
      
      //cout << "zeta_n = " << zeta_s[n] << ", zeta_{n-1} = " << zeta_s_prev[n];
      
      // beta_i^sigma = beta_i zeta_{n+1}^sigma / zeta_n^sigma
      beta_s[n] = beta*zeta_s[n]/zeta_s_prev[n];
      
      // 4. x_s = x_s - beta_s p_s
      caxpy(-beta_s[n], p_s[n], phi[n], size); 
      
      //cout << ", beta_n = " << beta_s[n] << "\n"; 
    }

    // 5. r = r + beta Ap
    caxpy(beta, Ap, r, size);
    
    // Exit if new residual is small enough
    rsqNew = norm2sq(r, size);
    
    print_verbosity_resid(verb, "CG-M", k+1, invif.ops_count, sqrt(rsqNew)/bsqrt); 
    
    // The residual of the shifted systems is zeta_s[n]*sqrt(rsqNew). Stop iterating on converged systems.
    if (k % resid_freq_check == 0)
    {
      for (n = 0; n < n_shift_rem; n++)
      {
        if (zeta_s[n]*sqrt(rsqNew) < eps*bsqrt) // if the residual of vector 'n' is sufficiently small...
        {
          // Permute it out.
          n_shift_rem--;
          
          //cout << "Vector " << mapping[n] << " has converged.\n" << flush;
          
          if (n_shift_rem != n) // Reorder in the case of out-of-order convergence. 
          {
            // Update mapping.
            tmp_int = mapping[n_shift_rem];
            mapping[n_shift_rem] = mapping[n];
            mapping[n] = tmp_int;

            // Permute phi, p_s, alpha_s, beta_s, zeta_s, zeta_s_prev, shifts. 
            tmp_ptr = phi[n_shift_rem];
            phi[n_shift_rem] = phi[n];
            phi[n] = tmp_ptr;
            
            tmp_ptr = p_s[n_shift_rem];
            p_s[n_shift_rem] = p_s[n];
            p_s[n] = tmp_ptr;
            
            tmp_dbl = alpha_s[n_shift_rem];
            alpha_s[n_shift_rem] = alpha_s[n];
            alpha_s[n] = tmp_dbl;
            
            tmp_dbl = beta_s[n_shift_rem];
            beta_s[n_shift_rem] = beta_s[n];
            beta_s[n] = tmp_dbl;
            
            tmp_dbl = zeta_s[n_shift_rem];
            zeta_s[n_shift_rem] = zeta_s[n];
            zeta_s[n] = tmp_dbl;
            
            tmp_dbl = zeta_s_prev[n_shift_rem];
            zeta_s_prev[n_shift_rem] = zeta_s_prev[n];
            zeta_s_prev[n] = tmp_dbl;
            
            tmp_dbl = shifts[n_shift_rem];
            shifts[n_shift_rem] = shifts[n];
            shifts[n] = tmp_dbl;
            
            // We swapped with the end, so we need to recheck the end.
            n--;
          }
        }
      }
    }

    if (/*sqrt(rsqNew) < eps*bsqrt || */(worst_first && abs(zeta_s[0])*sqrt(rsqNew) < eps*bsqrt) || n_shift_rem == 0 || k == max_iter-1) {
      //        printf("Final rsq = %g\n", rsqNew);
      break;
    }
    
    
  
    // 6. alpha = rsqNew / rsq.
    alpha = rsqNew / rsq;
    rsq = rsqNew; 
    
    //cout << "alpha = " << alpha << "\n";  
    
    for (n = 0; n < n_shift_rem; n++)
    {
      // 7. alpha_s = alpha * zeta_s * beta_s / (zeta_s_prev * beta)
      alpha_s[n] = alpha*zeta_s[n]*beta_s[n]/(zeta_s_prev[n] * beta);
      //cout << "alpha_n = " << alpha_s[n] << "\n";
      
      // 8. p_s = zeta_s_prev r + alpha_s p_s
      caxpby(zeta_s[n], r, alpha_s[n], p_s[n], size);
    }
    
    // Compute the new Ap.
    cxpay(r, alpha, p, size);
    (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;
  } 
    
  if(k == max_iter-1) {
    invif.success = false;
  }
  else
  {
     invif.success = true;
  }
  k++;
  
  // Undo the permutation damage.
  // Only need to permute phi, shifts. 
  for (n = 0; n < n_shift; n++)
  {
    // Find the true n'th vector.
    if (mapping[n] != n)
    {
      for (int m = n+1; m < n_shift; m++)
      {
        if (mapping[m] == n) // Match, swap.
        {
          tmp_ptr = phi[m];
          phi[m] = phi[n];
          phi[n] = tmp_ptr;
          
          tmp_dbl = shifts[m];
          shifts[m] = shifts[n];
          shifts[n] = tmp_dbl;
          
          mapping[m] = mapping[n];
          mapping[n] = n;
          
          n--;
          break;
        }
      }
    }
  }
  
  // Calculate explicit rsqs.
  double* relres = new double[n_shift];
  for (n = 0; n < n_shift; n++)
  {
    zero_vector(Ap, size);
    (*matrix_vector)(Ap, phi[n], extra_info); invif.ops_count++;
    caxpy(shifts[n], phi[n], Ap, size);
    invif.resSqmrhs[n] = diffnorm2sq(Ap, phi0, size);
    relres[n] = sqrt(invif.resSqmrhs[n])/bsqrt;
  }
  
  
  // Free all the things!
  deallocate_vector(&r);
  deallocate_vector(&p);
  deallocate_vector(&Ap);
  
  for (i = 0; i < n_shift; i++)
  {
    deallocate_vector(&p_s[i]);
  }
  delete[] p_s;
  
  delete[] alpha_s;
  delete[] beta_s;
  delete[] zeta_s;
  delete[] zeta_s_prev;
  
  delete[] mapping; 

  print_verbosity_summary_multi(verb, "CG-M", invif.success, k, invif.ops_count, relres, n_shift);
  delete[] relres; 
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = "CG-M";
  return invif; // Convergence 
} 

inversion_info minv_vector_cg_m(complex<double>  **phi, complex<double>  *phi0, int n_shift, int size, int resid_freq_check, int max_iter, double eps, double* shifts, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, bool worst_first, inversion_verbose_struct* verb)
{
  // Initialize vectors.
  complex<double> *r, *p, *Ap;
  complex<double> **p_s;
  complex<double> alpha, beta, beta_prev, tmp; 
  double rsq, rsqNew, bsqrt, truersq;
  complex<double> *alpha_s, *beta_s, *zeta_s, *zeta_s_prev;
  int k,i,n;
  int n_shift_rem = n_shift; // number of systems to still iterate on. 
  int* mapping; // holds the mapping between vectors and the original vector ordering.
                // this is because some vectors may converge before others. 
  complex<double>* tmp_ptr; // temporary pointer for swaps.
  double tmp_dbl_real; // temporary pointer for swaps.
  complex<double> tmp_dbl; // temporary double for swaps. 
  int tmp_int; // temporary int for swaps. 
  
  // Prepare an inversion_info for multiple residuals.
  inversion_info invif(n_shift); 

  // Allocate memory.
  alpha_s = new complex<double>[n_shift];
  beta_s = new complex<double>[n_shift];
  zeta_s = new complex<double>[n_shift];
  zeta_s_prev = new complex<double>[n_shift];
  
  p_s = new complex<double>*[n_shift];
  for (n = 0; n < n_shift; n++)
  {
    p_s[n] = allocate_vector<complex<double>>(size);
  }
  
  r = allocate_vector<complex<double>>(size);
  p = allocate_vector<complex<double>>(size);
  Ap = allocate_vector<complex<double>>(size);
  
  // Initialize mapping.
  mapping = new int[n_shift];
  for (n = 0; n < n_shift; n++)
  {
    mapping[n] = n; // All vectors are currently in order.
  }

  // Initialize values.
  rsq = 0.0; rsqNew = 0.0; bsqrt = 0.0; truersq = 0.0; k=0;
  for (n = 0; n < n_shift; n++)
  {
    // beta_0, zeta_0, zeta_-1
    beta_s[n] = zeta_s[n] = zeta_s_prev[n] = 1.0;
    // alpha_0. 
    alpha_s[n] = 0.0;
  }
  beta = 1.0; alpha = 0.0;

  // Zero vectors;
  zero_vector(r, size); 
  zero_vector(p, size); zero_vector(Ap, size);
  
  // Find norm of rhs.
  bsqrt = sqrt(norm2sq(phi0, size));
  
  // There can't be an initial guess... though it is sort of possible, in reference to:
  // http://arxiv.org/pdf/0810.1081v1.pdf
  
  // 1. x_sigma = 0, r = p_sigma = b.
  for (n = 0; n < n_shift; n++)
  {
    copy_vector(p_s[n], phi0, size);
    zero_vector(phi[n], size);
  }
  copy_vector(p, phi0, size);
  copy_vector(r, phi0, size);
  
  // Compute Ap.
  zero_vector(Ap, size);  
  (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;

  // Compute rsq.
  rsq = norm2sq(r, size);
  
  // iterate till convergence
  for(k = 0; k< max_iter; k++) {
    
    // 2. beta_i = - rsq / pAp. Which is a weird switch from the normal notation, but whatever.
    beta_prev = beta; 
    beta = -rsq/dot(p, Ap, size);
    //cout << "beta = " << beta << "\n";
    
    for (n = 0; n < n_shift_rem; n++)
    {
      // 3. Calculate beta_i^sigma, zeta_i+1^sigma according to 2.42 to 2.44.
      // zeta_{i+1}^sigma = complicated...
      tmp = zeta_s[n]; // Save zeta_i to pop into the prev zeta.
      zeta_s[n] = (zeta_s[n]*zeta_s_prev[n]*beta_prev)/(beta*alpha*(zeta_s_prev[n]-zeta_s[n]) + zeta_s_prev[n]*beta_prev*(1.0-shifts[n]*beta));
      zeta_s_prev[n] = tmp; 
      
      //cout << "zeta_n = " << zeta_s[n] << ", zeta_{n-1} = " << zeta_s_prev[n];
      
      // beta_i^sigma = beta_i zeta_{n+1}^sigma / zeta_n^sigma
      beta_s[n] = beta*zeta_s[n]/zeta_s_prev[n];
      
      // 4. x_s = x_s - beta_s p_s
      caxpy(-beta_s[n], p_s[n], phi[n], size); 
      
      //cout << ", beta_n = " << beta_s[n] << "\n"; 
    }

    // 5. r = r + beta Ap
    for (i = 0; i < size; i++)
    {
      r[i] = r[i] + beta*Ap[i];
    }
    
    // Exit if new residual is small enough
    rsqNew = norm2sq(r, size);
    
    print_verbosity_resid(verb, "CG-M", k+1, invif.ops_count, sqrt(rsqNew)/bsqrt); 
    
    // The residual of the shifted systems is zeta_s[n]*sqrt(rsqNew). Stop iterating on converged systems.
    if (k % resid_freq_check == 0)
    {
      for (n = 0; n < n_shift_rem; n++)
      {
        if (abs(zeta_s[n])*sqrt(rsqNew) < eps*bsqrt) // if the residual of vector 'n' is sufficiently small...
        {
          // Permute it out.
          n_shift_rem--;
          
          //cout << "Vector " << mapping[n] << " has converged.\n" << flush;
          
          if (n_shift_rem != n) // Reorder in the case of out-of-order convergence. 
          {
            // Update mapping.
            tmp_int = mapping[n_shift_rem];
            mapping[n_shift_rem] = mapping[n];
            mapping[n] = tmp_int;

            // Permute phi, p_s, alpha_s, beta_s, zeta_s, zeta_s_prev, shifts. 
            tmp_ptr = phi[n_shift_rem];
            phi[n_shift_rem] = phi[n];
            phi[n] = tmp_ptr;
            
            tmp_ptr = p_s[n_shift_rem];
            p_s[n_shift_rem] = p_s[n];
            p_s[n] = tmp_ptr;
            
            tmp_dbl = alpha_s[n_shift_rem];
            alpha_s[n_shift_rem] = alpha_s[n];
            alpha_s[n] = tmp_dbl;
            
            tmp_dbl = beta_s[n_shift_rem];
            beta_s[n_shift_rem] = beta_s[n];
            beta_s[n] = tmp_dbl;
            
            tmp_dbl = zeta_s[n_shift_rem];
            zeta_s[n_shift_rem] = zeta_s[n];
            zeta_s[n] = tmp_dbl;
            
            tmp_dbl = zeta_s_prev[n_shift_rem];
            zeta_s_prev[n_shift_rem] = zeta_s_prev[n];
            zeta_s_prev[n] = tmp_dbl;
            
            tmp_dbl_real = shifts[n_shift_rem];
            shifts[n_shift_rem] = shifts[n];
            shifts[n] = tmp_dbl_real;
            
            // We swapped with the end, so we need to recheck the end.
            n--;
          }
        }
      }
    }

    if (/*sqrt(rsqNew) < eps*bsqrt || */(worst_first && abs(zeta_s[0])*sqrt(rsqNew) < eps*bsqrt) || n_shift_rem == 0 || k == max_iter-1) {
      //        printf("Final rsq = %g\n", rsqNew);
      break;
    }
    
    
  
    // 6. alpha = rsqNew / rsq.
    alpha = rsqNew / rsq;
    rsq = rsqNew; 
    
    //cout << "alpha = " << alpha << "\n";  
    
    for (n = 0; n < n_shift_rem; n++)
    {
      // 7. alpha_s = alpha * zeta_s * beta_s / (zeta_s_prev * beta)
      alpha_s[n] = alpha*zeta_s[n]*beta_s[n]/(zeta_s_prev[n] * beta);
      //cout << "alpha_n = " << alpha_s[n] << "\n";
      
      // 8. p_s = zeta_s_prev r + alpha_s p_s
      caxpby(zeta_s[n], r, alpha_s[n], p_s[n], size);
    }
    
    // Compute the new Ap.
    cxpay(r, alpha, p, size);
    (*matrix_vector)(Ap, p, extra_info); invif.ops_count++;
  } 
    
  if(k == max_iter-1) {
    invif.success = false;
  }
  else
  {
     invif.success = true;
  }
  k++;
  
  // Undo the permutation damage.
  // Only need to permute phi, shifts. 
  for (n = 0; n < n_shift; n++)
  {
    // Find the true n'th vector.
    if (mapping[n] != n)
    {
      for (int m = n+1; m < n_shift; m++)
      {
        if (mapping[m] == n) // Match, swap.
        {
          tmp_ptr = phi[m];
          phi[m] = phi[n];
          phi[n] = tmp_ptr;
          
          tmp_dbl_real = shifts[m];
          shifts[m] = shifts[n];
          shifts[n] = tmp_dbl_real;
          
          mapping[m] = mapping[n];
          mapping[n] = n;
          
          n--;
          break;
        }
      }
    }
  }
  
  // Calculate explicit rsqs.
  double* relres = new double[n_shift];
  for (n = 0; n < n_shift; n++)
  {
    zero_vector(Ap, size);
    (*matrix_vector)(Ap, phi[n], extra_info); invif.ops_count++;
    caxpy(shifts[n], phi[n], Ap, size);
    invif.resSqmrhs[n] = diffnorm2sq(Ap, phi0, size);
    relres[n] = sqrt(invif.resSqmrhs[n])/bsqrt;
  }
  
  
  // Free all the things!
  deallocate_vector(&r);
  deallocate_vector(&p);
  deallocate_vector(&Ap);
  
  for (i = 0; i < n_shift; i++)
  {
    deallocate_vector(&p_s[i]);
  }
  delete[] p_s;
  
  delete[] alpha_s;
  delete[] beta_s;
  delete[] zeta_s;
  delete[] zeta_s_prev;
  
  delete[] mapping; 

  print_verbosity_summary_multi(verb, "CG-M", invif.success, k, invif.ops_count, relres, n_shift);
  delete[] relres; 
  
  invif.resSq = truersq;
  invif.iter = k;
  invif.name = "CG-M";
  return invif; // Convergence 
}


#endif