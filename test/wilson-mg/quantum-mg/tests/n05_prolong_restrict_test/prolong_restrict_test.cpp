// Copyright (c) 2017 Evan S Weinberg
// Test prolongation and restriction
// using both a symmetric and asymmetric 
// process.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <random>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"

// QMG
#include "lattice/lattice.h"
#include "transfer/transfer.h"


int main(int argc, char** argv)
{
  // Set output to not be that long.
  cout << setiosflags(ios::fixed) << setprecision(6);

  //Iterators and such.
  int i;

  // Random number generator.
  std::mt19937 generator (1337u);

  // Basic information for fine level.
  const int x_len = 4;
  const int y_len = 4;
  const int dof = 1;

  // Blocking size.
  const int x_block = 4;
  const int y_block = 4;

  // Number of null vectors (equiv. degrees of freedom on coarse level)
  const int coarse_dof = 6;

  // Create a lattice object for the fine lattice.
  Lattice2D* lat = new Lattice2D(x_len, y_len, dof);

  // Create a lattice object for the coarse lattice.
  Lattice2D* coarse_lat = new Lattice2D(x_len/x_block, y_len/y_block, coarse_dof);

  // Coarse, fine size.
  const int fine_size_cv = lat->get_size_cv();
  const int coarse_size_cv = coarse_lat->get_size_cv();

  // Allocate space for vectors.
  complex<double>** prolong_vecs = new complex<double>*[coarse_dof];
  complex<double>** restrict_vecs = new complex<double>*[coarse_dof];
  for (i = 0; i < coarse_dof; i++)
  {
    prolong_vecs[i] = allocate_vector<complex<double>>(fine_size_cv);
    restrict_vecs[i] = allocate_vector<complex<double>>(fine_size_cv);
  }

  // Allocate space for two fine vectors.
  complex<double>* fine_vec_1 = allocate_vector<complex<double>>(fine_size_cv);
  complex<double>* fine_vec_2 = allocate_vector<complex<double>>(fine_size_cv);

  // Allocate space for two coarse vectors.
  complex<double>* coarse_vec_1 = allocate_vector<complex<double>>(coarse_size_cv);
  complex<double>* coarse_vec_2 = allocate_vector<complex<double>>(coarse_size_cv);

  ////////////////////////////////////////////
  // TEST 1: SYMMETRIC PROLONG AND RESTRICT //
  ////////////////////////////////////////////

  // Generate random vectors.
  for (i = 0; i < coarse_dof; i++)
  {
    gaussian(prolong_vecs[i], fine_size_cv, generator);
  }

  // Create a symmetric transfer object.
  TransferMG* symmetric_transfer_obj = new TransferMG(lat, coarse_lat, prolong_vecs, true);

  // Subtest 1: Restrict then prolong should preserve each null vector.
  std::cout << "Symmetric Test: Preserve fine null vectors, ||(1 - P P^dagger) v[i]||_f.\n";
  for (i = 0; i < coarse_dof; i++)
  {
    zero_vector(coarse_vec_1, coarse_size_cv);
    zero_vector(fine_vec_1, fine_size_cv);
    symmetric_transfer_obj->restrict_f2c(prolong_vecs[i], coarse_vec_1);
    symmetric_transfer_obj->prolong_c2f(coarse_vec_1, fine_vec_1);
    std::cout << "Vector " << i << " diff " << sqrt(diffnorm2sq(prolong_vecs[i], fine_vec_1, fine_size_cv)/norm2sq(prolong_vecs[i], fine_size_cv)) << "\n";
  }

  // Subtest 2: Prolong then restrict should preserve coarse space.
  std::cout << "\nSymmetric Test: Preserve coarse space, ||(1 - P^dagger P) v_coarse||_c \n";
  gaussian(coarse_vec_1, coarse_size_cv, generator);
  zero_vector(fine_vec_1, fine_size_cv);
  zero_vector(coarse_vec_2, coarse_size_cv);
  symmetric_transfer_obj->prolong_c2f(coarse_vec_1, fine_vec_1);
  symmetric_transfer_obj->restrict_f2c(fine_vec_1, coarse_vec_2);
  std::cout << "Preserve coarse space diff " << sqrt(diffnorm2sq(coarse_vec_1, coarse_vec_2, coarse_size_cv)/norm2sq(coarse_vec_1, coarse_size_cv)) << "\n";

  delete symmetric_transfer_obj;

  /////////////////////////////////////////////
  // TEST 2: ASYMMETRIC PROLONG AND RESTRICT //
  /////////////////////////////////////////////

  // Generate random vectors.
  for (i = 0; i < coarse_dof; i++)
  {
    gaussian(prolong_vecs[i], fine_size_cv, generator);
    gaussian(restrict_vecs[i], fine_size_cv, generator);
  }

  // Create a symmetric transfer object.
  TransferMG* asymmetric_transfer_obj = new TransferMG(lat, coarse_lat, prolong_vecs, restrict_vecs, true);

  // Subtest 1: Restrict then prolong should preserve each null vector.
  std::cout << "Asymmetric Test: Preserve fine null vectors, ||(1 - P R) v[i]||_f.\n";
  for (i = 0; i < coarse_dof; i++)
  {
    zero_vector(coarse_vec_1, coarse_size_cv);
    zero_vector(fine_vec_1, fine_size_cv);
    asymmetric_transfer_obj->restrict_f2c(prolong_vecs[i], coarse_vec_1);
    asymmetric_transfer_obj->prolong_c2f(coarse_vec_1, fine_vec_1);
    std::cout << "Vector " << i << " diff " << sqrt(diffnorm2sq(prolong_vecs[i], fine_vec_1, fine_size_cv)/norm2sq(prolong_vecs[i], fine_size_cv)) << "\n";
  }

  // Subtest 2: Prolong then restrict should preserve coarse space.
  std::cout << "\nAsymmetric Test: Preserve coarse space, ||(1 - R P) v_coarse||_c \n";
  gaussian(coarse_vec_1, coarse_size_cv, generator);
  zero_vector(fine_vec_1, fine_size_cv);
  zero_vector(coarse_vec_2, coarse_size_cv);
  asymmetric_transfer_obj->prolong_c2f(coarse_vec_1, fine_vec_1);
  asymmetric_transfer_obj->restrict_f2c(fine_vec_1, coarse_vec_2);
  std::cout << "Preserve coarse space diff " << sqrt(diffnorm2sq(coarse_vec_1, coarse_vec_2, coarse_size_cv)/norm2sq(coarse_vec_1, coarse_size_cv)) << "\n";

  delete asymmetric_transfer_obj;

  ////////////////////////////
  // AND THAT'S ALL, FOLKS! //
  ////////////////////////////

  // Clean up.
  deallocate_vector(&coarse_vec_1);
  deallocate_vector(&coarse_vec_2);
  deallocate_vector(&fine_vec_1);
  deallocate_vector(&fine_vec_2);

  for (i = 0; i < coarse_dof; i++)
  {
    deallocate_vector(&prolong_vecs[i]);
    deallocate_vector(&restrict_vecs[i]);
  }
  delete[] prolong_vecs;
  delete[] restrict_vecs;

  return 0;
}
