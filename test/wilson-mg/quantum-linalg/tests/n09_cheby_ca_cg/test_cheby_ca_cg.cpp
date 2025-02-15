// Copyright (c) 2017 Evan S Weinberg
// Test code for a real operator.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <sstream>
#include <complex>
#include <random>

// Borrow dense matrix eigenvalue routines.
#include <Eigen/Dense>
using namespace Eigen;

#include "blas/generic_vector.h"
#include "inverters/generic_cg.h"
#include "inverters/generic_ca_cg.h"
#include "inverters/generic_cheby_ca_cg.h"

using namespace std; 

// Define a function that just has varying values along the diagonal.
// Makes it easy to control the condition number.
struct scale_info {
  int size;
  double* scale_vec;
  double lambda_min; // minimum eigenvalue
  double lambda_max; // maximum eigenvalues
  bool log_distribute; // randomly distribute evenly or logarithmically?

  scale_info(int size, double lambda_min, double lambda_max, bool log_distribute, std::mt19937& generator)
   : size(size), lambda_min(lambda_min), lambda_max(lambda_max), log_distribute(log_distribute) {

    scale_vec = allocate_vector<double>(size);

    // First: gurantee lambda_min and lambda_max exist.
    scale_vec[0] = lambda_min;
    scale_vec[1] = lambda_max;

    if (log_distribute) {
      double log_lambda_min = log(lambda_min);
      double log_lambda_max = log(lambda_max);
      random_uniform(scale_vec+2, size-2, generator, log_lambda_min, log_lambda_max);
      exp_vector(scale_vec+2, size-2);
    } else {
      random_uniform(scale_vec+2, size-2, generator, lambda_min, lambda_max);
    }
  }

  scale_info(int size, std::mt19937& generator)
   : scale_info(size, 1., 10., true, generator) { ; }

  ~scale_info() {
    deallocate_vector(&scale_vec);
  }


};

// Define a function that just has varying values along the diagonal.
void scale_function(complex<double>* rhs, complex<double>* lhs, void* extra_data) {
  scale_info& info = *((scale_info*)(extra_data));
  cxtyz(info.scale_vec, lhs, rhs, info.size);
}


// Prepare vectors for various tests.
void reset_vectors(complex<double>* rhs, complex<double>* lhs, complex<double>* check, complex<double>* rhs_backup, int size)
{
  zero_vector(lhs, size);
  copy_vector(rhs, rhs_backup, size);
  zero_vector(check, size);
}


int main(int argc, char** argv)
{  
    //double *lattice; // Holds the gauge field.
  complex<double> *lhs, *rhs, *rhs_backup, *check; // For some Kinetic terms.
  double explicit_resid = 0.0;
  double bnorm = 0.0;
  inversion_info invif;
  std::mt19937 generator (1337u); // RNG, 1337u is the seed. 

  // Basic information about the linop
  int size = 8192;
  double lambda_min = 1e-6;
  double lambda_max = 10;
  
  // Parameters related to solve.
  double tol = 1e-8;
  int max_iter = 1000;
  
  // Vectors. 
  lhs = allocate_vector<complex<double>>(size);
  rhs = allocate_vector<complex<double>>(size);
  rhs_backup = allocate_vector<complex<double>>(size);
  check = allocate_vector<complex<double>>(size);

  // Set up a default rhs.
  random_uniform(rhs_backup, size, generator, 0.5, 1.5);
  
  // Zero out vectors, set rhs to standard backed up version
  reset_vectors(rhs, lhs, check, rhs_backup, size);

  // Set up operator.
  scale_info scinf(size, lambda_min, lambda_max, false, generator);
  
  // Get norm for rhs.
  bnorm = sqrt(norm2sq(rhs, size));

  printf("Solving A [lhs] = [rhs] for lhs, using a random source. Operator has eigenvalues distributed between %e and %e \n\n", lambda_min, lambda_max);

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

  inversion_verbose_struct verb(VERB_DETAIL, "Details: ");

  /* CG */
  reset_vectors(rhs, lhs, check, rhs_backup, size);
  invif = minv_vector_cg(lhs, rhs, size, max_iter, tol, scale_function, &scinf, &verb);
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
  scale_function(check, lhs, &scinf);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, size))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  

  /* CA-CG */
  for (int ca_s = 2; ca_s <= 30; ca_s++)
  {
    reset_vectors(rhs, lhs, check, rhs_backup, size);
    invif = minv_vector_ca_cg(lhs, rhs, size, max_iter, tol, ca_s, scale_function, &scinf, &verb);
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
    scale_function(check, lhs, &scinf);
    explicit_resid = sqrt(diffnorm2sq(rhs, check, size))/bnorm; // sqrt(|rhs - check|^2)/bnorm
    printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);


    reset_vectors(rhs, lhs, check, rhs_backup, size);
    // invif = minv_vector_cheby_ca_cg(lhs, rhs, size, max_iter, tol, 1.01*lambda_max, ca_s, scale_function, &scinf, &verb); // assume lambda_min = 0
    invif = minv_vector_cheby_ca_cg(lhs, rhs, size, max_iter, tol, lambda_min, lambda_max, ca_s, scale_function, &scinf, &verb);
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
    scale_function(check, lhs, &scinf);
    explicit_resid = sqrt(diffnorm2sq(rhs, check, size))/bnorm; // sqrt(|rhs - check|^2)/bnorm
    printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  }
  

  //////////////
  // CLEAN UP //
  //////////////

  // Free the lattice.
  //delete[] lattice;
  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&rhs_backup);
  deallocate_vector(&check);
  return 0;
}


