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

#include "square_laplace.h"

using namespace std; 


// Prepare vectors for various tests.
void reset_vectors(complex<double>* rhs, complex<double>* lhs, complex<double>* check, int length)
{
  zero_vector(lhs, length*length);
  zero_vector(rhs, length*length);
  rhs[length/2+(length/2)*length] = 1.0; // set a point on rhs.
  zero_vector(check, length*length);
}


int main(int argc, char** argv)
{  
    //double *lattice; // Holds the gauge field.
  complex<double> *lhs, *rhs, *check; // For some Kinetic terms.
  complex<double> *gauge_links; 
  double explicit_resid = 0.0;
  double bnorm = 0.0;
  inversion_info invif;
  std::mt19937 generator (1337u); // RNG, 1337u is the seed. 

  // Basic information about the lattice.
  int length = 32;
  double m_sq = 0.01;
  double inv_variance = 6.0; // inverse of variance for gaussian non-compact U(1) links.
  
  // Create a random compact U(1) link.
  gauge_links = allocate_vector<complex<double>>(2*length*length);
  
  // Remark: fills real and imaginary with gaussian numbers.
  gaussian(gauge_links, 2*length*length, generator, 1.0/inv_variance);
  polar(gauge_links, 2*length*length); // ignores imag part to make compact link.
  
  
  // Structure which gets passed to the function.
  laplace_gauged_struct lapstr;
  lapstr.length = length;
  lapstr.m_sq = m_sq;
  lapstr.gauge_links = gauge_links; 
  
  // Parameters related to solve.
  double tol = 1e-8;
  int max_iter = 200;
  //double minres_relaxation_param = 0.8;
  //double richardson_relaxation_param = 0.2;
  //int restart_freq = 64; // for restarted solves. 
  //int bicgstabl = 8; // L for, well, BiCGstab-L. 
  
  // Some start-up.
  int volume = length*length;
  
  //lattice = allocate_vector<double>(2*volume);  // Gauge field eventually.
  
  // Vectors. 
  lhs = allocate_vector<complex<double>>(volume);
  rhs = allocate_vector<complex<double>>(volume);
  check = allocate_vector<complex<double>>(volume);
  
  // Zero out vectors, set rhs point.
  // zero_vector(lattice, 2*volume);
  reset_vectors(rhs, lhs, check, length);
  
  // Get norm for rhs.
  bnorm = sqrt(norm2sq(rhs, volume));

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

  inversion_verbose_struct verb(VERB_DETAIL, "Details: ");

  /* CG */
  reset_vectors(rhs, lhs, check, length);
  invif = minv_vector_cg(lhs, rhs, volume, max_iter, tol, square_laplacian_gauged, &lapstr, &verb);
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
  square_laplacian_gauged(check, lhs, &lapstr);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
  printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  

  /* CA-CG */
  for (int ca_s = 2; ca_s <= 10; ca_s++)
  {
    reset_vectors(rhs, lhs, check, length);
    invif = minv_vector_ca_cg(lhs, rhs, volume, max_iter, tol, ca_s, square_laplacian_gauged, &lapstr, &verb);
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
    square_laplacian_gauged(check, lhs, &lapstr);
    explicit_resid = sqrt(diffnorm2sq(rhs, check, volume))/bnorm; // sqrt(|rhs - check|^2)/bnorm
    printf("[check] should equal [rhs]. The residual is %15.20e.\n\n", explicit_resid);
  }
  

  //////////////
  // CLEAN UP //
  //////////////

  // Free the lattice.
  //delete[] lattice;
  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&check);
  deallocate_vector(&gauge_links); 
  return 0;
}


