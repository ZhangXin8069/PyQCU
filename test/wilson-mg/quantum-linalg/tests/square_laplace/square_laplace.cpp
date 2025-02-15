// Copyright (c) 2017 Evan S Weinberg
// Test code for a real operator.

#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <complex>

#include "blas/generic_vector.h"

#include "inverters/generic_richardson.h"
#include "inverters/generic_minres.h"
#include "inverters/generic_cheby_iters.h"
#include "inverters/generic_cg.h"
#include "inverters/generic_cr.h"
#include "inverters/generic_bicgstab.h"
#include "inverters/generic_bicgstab_l.h"
#include "inverters/generic_gcr.h"
#include "inverters/generic_tfqmr.h"

#include "inverters/generic_cg_precond.h"
#include "inverters/generic_gcr_var_precond.h"

#include "inverters/generic_cg_m.h"

#include "square_laplace.h"


using namespace std; 

// Square laplacian function.
void square_laplacian(double* lhs, double* rhs, void* extra_data);

// Reference preconditioning function, which preconditions solving
// the square laplace equation with 8 iterations of CG (I know this is
// a silly preconditioner, but it's meant to demonstrate a point).
void square_laplacian_cgpreconditioner(double* lhs, double* rhs, int size, void* extra_data, inversion_verbose_struct* verb);

// Reference preconditioning function, which preconditions solving
// the square laplace equation with 8 iterations of Cheby iterations
void square_laplacian_chebypreconditioner(double* lhs, double* rhs, int size, void* extra_data, inversion_verbose_struct* verb);


// Prepare vectors for various tests.
void reset_vectors(double* rhs, double* lhs, double* check, int length)
{
  zero_vector(lhs, length*length);
  zero_vector(rhs, length*length);
  rhs[length/2+(length/2)*length] = 1.0; // set a point on rhs.
  zero_vector(check, length*length);
}


int main(int argc, char** argv)
{  
  //double *lattice; // Holds the gauge field.
  double *lhs, *rhs, *check; // For some Kinetic terms.
  double explicit_resid = 0.0;
  double bnorm = 0.0;
  inversion_info invif;

  // Basic information about the lattice.
  int length = 128;
  double m_sq = 0.001;
  
  // Structure which gets passed to the function.
  laplace_struct lapstr;
  lapstr.length = length;
  lapstr.m_sq = m_sq;
  
  // Parameters related to solve.
  double tol = 1e-8;
  int max_iter = 4000;
  double minres_relaxation_param = 0.8;
  double richardson_relaxation_param = 0.2;
  int restart_freq = 64; // for restarted solves. 
  int bicgstabl = 8; // L for, well, BiCGstab-L. 
  
  // Some start-up.
  int volume = length*length;
  
  //lattice = allocate_vector<double>(2*volume);  // Gauge field eventually.
  
  // Vectors. 
  lhs = allocate_vector<double>(volume);
  rhs = allocate_vector<double>(volume);
  check = allocate_vector<double>(volume);
  
  // Zero out vectors, set rhs point.
  // zero_vector(lattice, 2*volume);
  reset_vectors(rhs, lhs, check, length);
  
  // Get norm for rhs.
  bnorm = sqrt(norm2sq<double>(rhs, volume));

  printf("Solving A [lhs] = [rhs] for lhs, using a point source. Single mass test: m^2 = %f.\n\n", m_sq);

  /****************
  * NORMAL SOLVES *
  ****************/
  
  // lhs = A^(-1) rhs
  // Arguments:
  // 1: lhs
  // 2: rhs
  // 3: size of vector
  // 4: maximum iterations
  // 5: residual
  // 5a for gcr_restart: how often to restart.
  // 5b for richardson, minres: overrelaxation parameter (can leave this out, assumes 1)
  // 5c for richardson: how often to check the residual
  // 6: function pointer
  // 7: "extra data": 
  // 8: optional, verbosity information.

  /* Richardson w/ Relaxation param, checking residual every 10 iters. */
  reset_vectors(rhs, lhs, check, length);
  invif = minv_vector_richardson(lhs, rhs, volume, max_iter, tol, richardson_relaxation_param, 10, square_laplacian, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  

  /* MinRes w/ Relaxation param. */
  reset_vectors(rhs, lhs, check, length);
  invif = minv_vector_minres(lhs, rhs, volume, max_iter, tol, minres_relaxation_param, square_laplacian, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  
  /* Cheby iterations w/ known eigenvalues */
  reset_vectors(rhs, lhs, check, length);
  invif = minv_vector_cheby_iters(lhs, rhs, volume, max_iter, tol, m_sq, 8.0+m_sq, 10, square_laplacian, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  

  /* CG */
  reset_vectors(rhs, lhs, check, length);
  invif = minv_vector_cg(lhs, rhs, volume, max_iter, tol, square_laplacian, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);

  /* CR */
  reset_vectors(rhs, lhs, check, length);
  invif = minv_vector_cr(lhs, rhs, volume, max_iter, tol, square_laplacian, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  
  /* BiCGstab */
  reset_vectors(rhs, lhs, check, length);
  invif = minv_vector_bicgstab(lhs, rhs, volume, max_iter, tol, square_laplacian, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  
  /* BiCGstab-L */
  reset_vectors(rhs, lhs, check, length);
  invif = minv_vector_bicgstab_l(lhs, rhs, volume, max_iter, tol, bicgstabl,  square_laplacian, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  
  /* Restarted GCR */
  reset_vectors(rhs, lhs, check, length);
  invif = minv_vector_gcr_restart(lhs, rhs, volume, max_iter, tol, restart_freq, square_laplacian, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);

  /* TFQMR */
  reset_vectors(rhs, lhs, check, length);
  invif = minv_vector_tfqmr(lhs, rhs, volume, max_iter, tol, square_laplacian, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  
  /************************
  * PRECONDITIONED SOLVES *
  ************************/
  
  // lhs = A^(-1) rhs
  // Arguments:
  // 1: lhs
  // 2: rhs
  // 3: size of vector
  // 4: maximum iterations
  // 5: residual
  // 5a for gcr_restart: how often to restart.
  // 5b for bicgstab_l: what value of l to use. 
  // 6: function pointer
  // 7: "extra data":
  // 8: preconditioning function pointer
  // 9: preconditioning "extra_data"
  // 10: optional, verbosity information.

  /* Variably preconditioned CG */
  reset_vectors(rhs, lhs, check, length); 
  invif = minv_vector_cg_precond(lhs, rhs, volume, max_iter, tol, square_laplacian, &lapstr, square_laplacian_cgpreconditioner, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  
  /* Variably preconditioned CG */
  reset_vectors(rhs, lhs, check, length); 
  invif = minv_vector_cg_precond(lhs, rhs, volume, max_iter, tol, square_laplacian, &lapstr, square_laplacian_chebypreconditioner, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s (Cheby) took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s (Cheby) took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  
  
  /* Variably preconditioned GCR */
  reset_vectors(rhs, lhs, check, length); 
  invif = minv_vector_gcr_var_precond(lhs, rhs, volume, max_iter, tol, square_laplacian, &lapstr, square_laplacian_cgpreconditioner, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);

  /* Variably preconditioned GCR, Cheby */
  reset_vectors(rhs, lhs, check, length); 
  invif = minv_vector_gcr_var_precond(lhs, rhs, volume, max_iter, tol, square_laplacian, &lapstr, square_laplacian_chebypreconditioner, &lapstr);
  if (invif.success == true)
  {
    printf("Algorithm %s (Cheby) took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s (Cheby) took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
  }
  printf("Computing [check] = A [lhs] as a confirmation.\n");
  // Check and make sure we get the right answer.
  square_laplacian(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  
  /********************
  * MULTISHIFT SOLVES *
  ********************/
  
  /* Special check for multishift. */
  printf("Solving (A + shift[i]) lhs[i] = [rhs] for lhs, using a point source. Multishift mass test: m^2 = %f, %f, %f.\n\n", m_sq, m_sq+0.01, m_sq+0.05);
  
  reset_vectors(rhs, lhs, check, length);
  // Need to allocate multiple left hand sides and shifts.
  double** multi_lhs = new double*[3];
  for (int i = 0; i < 3; i++)
  {
    multi_lhs[i] = allocate_vector<double>(volume);
  }
  // Shifts are ~relative~ to m_sq.
  double* shifts = new double[3];
  shifts[0] = 0.0; shifts[1] = 0.01; shifts[2] = 0.05;
  // Frequency of checking solution of shifted soln. See description
  //   below.
  int multi_check_freq = 10; 
  
  // Arg 1: array of left hand sides.
  // Arg 2: right hand side.
  // Arg 3: number of left hand sides.
  // Arg 4: length of vectors.
  // Arg 5: frequency of checks of shifted solutions.
  //          There's a O(1) simple relation for checking if a shifted
  //          solution has converged, but it could get expensive
  //          to check it every iteration. This parameter denotes
  //          how frequently to check it. If a shifted solution has
  //          converged, it doesn't have to be updated every iteration,
  //          saving two caxpy-like operations per iter.
  // Arg 6: maximum number of iterations
  // Arg 7: target solution tolerance
  // Arg 8: pointer to array of shifts. 
  // Arg 9: function pointer for mat-vec operation. 
  // Arg 10: structure for mat-vec operation.
  // Arg 11: Optional: True if the worst-conditioned shift is the first shift,
  //           (default) false otherwise. This seems pointless for CG (because
  //           it is), but it's there for consistency with multishift CR and
  //           BiCGstab.
  // Arg 12: Optional: verbosity structure. 
  invif = minv_vector_cg_m(multi_lhs, rhs, 3, volume, multi_check_freq, 
                             max_iter, tol, shifts, square_laplacian,
                             &lapstr, true);
  if (invif.success == true)
  {
    printf("Algorithm %s took %d iterations to reach a tolerance of {%.8e, %.8e, %.8e} for m_sq {%.3f, %.3f, %.3f}.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSqmrhs[0])/bnorm, sqrt(invif.resSqmrhs[1])/bnorm, sqrt(invif.resSqmrhs[2])/bnorm, m_sq + shifts[0], m_sq + shifts[1], m_sq + shifts[2]);
  }
  else // failed, maybe.
  {
    printf("Potential error! Algorithm %s took %d iterations to reach a tolerance of {%.8e, %.8e, %.8e} for m_sq {%.3f, %.3f, %.3f}.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSqmrhs[0])/bnorm, sqrt(invif.resSqmrhs[1])/bnorm, sqrt(invif.resSqmrhs[2])/bnorm, m_sq + shifts[0], m_sq + shifts[1], m_sq + shifts[2]);
  }
  printf("Computing [check] = A [lhs][0] as a confirmation.\n");
  // Check and make sure we get the right answer.
  lapstr.m_sq = m_sq + shifts[0];
  square_laplacian(check, multi_lhs[0], &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n", explicit_resid);
  printf("Computing [check] = A [lhs][1] as a confirmation.\n");
  // Check and make sure we get the right answer.
  lapstr.m_sq = m_sq + shifts[1];
  square_laplacian(check, multi_lhs[1], &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n", explicit_resid);
  printf("Computing [check] = A [lhs][1] as a confirmation.\n");
  // Check and make sure we get the right answer.
  lapstr.m_sq = m_sq + shifts[2];
  square_laplacian(check, multi_lhs[2], &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n", explicit_resid);
  lapstr.m_sq = m_sq;
  

  // Free the lattice.
  //delete[] lattice;
  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&check);
  
  // deallocate quantities relevant for multi-shift solve.
  for (int i = 0; i < 3; i++)
  {
    deallocate_vector(&multi_lhs[i]);
  }
  delete[] multi_lhs; 
  delete[] shifts; 
}

// Square lattice.
// Kinetic term for a 2D laplacian w/ period bc. Applies lhs = A*rhs.
// The unit vectors are e_1 = xhat, e_2 = yhat.
// The "extra_data" allows us to generalize these functions later.
// It would become an internal structure in C++ code.
void square_laplacian(double* lhs, double* rhs, void* extra_data)
{
  // Declare variables.
  int i;
  int x,y;
  laplace_struct* lapstr = (laplace_struct*)extra_data;

  int length = lapstr->length;
  int volume = lapstr->length*lapstr->length;
  double m_sq = lapstr->m_sq;
  
  // For a 2D square lattice, the stencil is:
  //     |  0 -1  0 |
  //     | -1 +4 -1 |
  //     |  0 -1  0 |
  //
  // e2 = yhat
  // ^
  // | 
  // |-> e1 = xhat

  // Apply the stencil.
  for (i = 0; i < volume; i++)
  {
    lhs[i] = 0.0;
    x = i%length; // integer mod.
    y = i/length; // integer divide.

    // + e1.
    lhs[i] = lhs[i]-rhs[y*length+((x+1)%length)];

    // - e1.
    lhs[i] = lhs[i]-rhs[y*length+((x+length-1)%length)];

    // + e2.
    lhs[i] = lhs[i]-rhs[((y+1)%length)*length+x];

    // - e2.
    lhs[i] = lhs[i]-rhs[((y+length-1)%length)*length+x];

    // 0
    // Added mass term here.
    lhs[i] = lhs[i]+(4+m_sq)*rhs[i];
  }

}

// Reference preconditioning function, which preconditions solving
// the square laplace equation with 8 iterations of CG (I know this is
// a silly preconditioner, but it's meant to demonstrate a point).
void square_laplacian_cgpreconditioner(double* lhs, double* rhs, int size, void* extra_data, inversion_verbose_struct* verb)
{
  // Run 8 iterations of CG.
  // 8-> max of 8 iterations
  // 1e-15 -> make sure the 8 iterations is what dominates.
  // Otherwise pass data through.
  minv_vector_cg(lhs, rhs, size, 8, 1e-15, square_laplacian, extra_data, verb);
  
  // If I was smart about this, I'd write a special structure which holds
  // a restart count, tolerance, matrix pointer, and the original 'extra_data'.
  // This would avoid hard coding everything.
}

// Reference preconditioning function, which preconditions solving
// the square laplace equation with 8 iterations of Cheby
void square_laplacian_chebypreconditioner(double* lhs, double* rhs, int size, void* extra_data, inversion_verbose_struct* verb)
{
  laplace_struct* lapstr = (laplace_struct*)extra_data;

  // Run 8 iterations of Cheby
  // 8-> max of 8 iterations
  minv_vector_cheby_iters(lhs, rhs, size, 8, 1e-15, lapstr->m_sq, 8.0+lapstr->m_sq, 8, square_laplacian, extra_data, verb);

}
