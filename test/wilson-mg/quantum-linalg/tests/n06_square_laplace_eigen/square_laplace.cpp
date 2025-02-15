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

#include "blas/generic_vector.h"

#include "square_laplace.h"

using namespace std; 
using namespace Eigen;

typedef Matrix<std::complex<double>, Dynamic, Dynamic, ColMajor> cMatrix;
typedef Matrix<double, Dynamic, Dynamic, ColMajor> dMatrix;

int main(int argc, char** argv)
{  
  double *rhs_real;
  double *rhs_real_indef;
  complex<double> *rhs_cplx;
  complex<double> *rhs_cplx_indef;
  complex<double> *gauge_links;

  // Set output precision to be long.
  cout << setprecision(10);

  // RNG related things.
  std::mt19937 generator (1337u); // RNG, 1337u is the seed. 
  double inv_variance = 6.0; // inverse of variance for gaussian non-compact U(1) links.

  // Basic information about the lattice.
  int length = 8;
  double m_sq = 0.001;
  
  // Some start-up.
  int volume = length*length;
  int wilson_volume = 2*volume;
  
  // Create a random compact U(1) link.
  gauge_links = allocate_vector<complex<double>>(2*length*length);
  gaussian_real(gauge_links, 2*length*length, generator, 1.0/inv_variance);
  polar(gauge_links, 2*length*length);
  
  // Vectors. 
  rhs_real = allocate_vector<double>(volume);
  rhs_real_indef = allocate_vector<double>(volume);
  rhs_cplx = allocate_vector<complex<double>>(volume);
  rhs_cplx_indef = allocate_vector<complex<double>>(wilson_volume);

  // Zero out the vector.
  zero_vector(rhs_real, length*length);
  zero_vector(rhs_real_indef, length*length);
  zero_vector(rhs_cplx, length*length);
  zero_vector(rhs_cplx_indef, 2*length*length);

  //////////////////////////
  // REAL, SYMMETRIC CASE //
  //////////////////////////

  std::cout << "Real, Symmetric case.\n\n";

  // Structure which gets passed to the function.
  laplace_struct lapstr;
  lapstr.length = length;
  lapstr.m_sq = m_sq;

  // Allocate a sufficiently gigantic matrix.
  dMatrix mat_real = dMatrix::Zero(volume, volume);

  // Form matrix elements. This is where it's important that
  // dMatrix and cMatrix are column major.
  // I should probably make this safer by using a "Map".
  for (int i = 0; i < volume; i++)
  {
    // Set a point on the rhs for a matrix element.
    zero_vector(rhs_real, volume);
    rhs_real[i] = 1.0;

    // Where we put the result of the matrix element.
    double* mptr = &(mat_real(i*volume));

    square_laplacian(mptr, rhs_real, &lapstr);
  }

  // This should be the matrix. Let's print it to make sure.

  if (volume <= 16)
  {
    std::cout << mat_real << "\n";
  }

  // Get the eigenvalues.
  SelfAdjointEigenSolver<dMatrix> eigsolve_real(volume);
  eigsolve_real.compute(mat_real);

  std::cout << "The eigenvalues are:\n" << eigsolve_real.eigenvalues() << "\n";

  /////////////////////////////
  // COMPLEX, HERMITIAN CASE //
  /////////////////////////////

  std::cout << "\n\nComplex, Symmetric case.\n\n";

  // Structure which gets passed to the function.
  laplace_gauged_struct lapstr_gauged;
  lapstr_gauged.length = length;
  lapstr_gauged.m_sq = m_sq;
  lapstr_gauged.gauge_links = gauge_links; 

  // Allocate a sufficiently gigantic matrix.
  cMatrix mat_cplx = cMatrix::Zero(volume, volume);

  // Form matrix elements. This is where it's important that
  // dMatrix and cMatrix are column major.
  // I should probably make this safer by using a "Map".
  for (int i = 0; i < volume; i++)
  {
    // Set a point on the rhs for a matrix element.
    zero_vector(rhs_cplx, volume);
    rhs_cplx[i] = 1.0;

    // Where we put the result of the matrix element.
    complex<double>* mptr = &(mat_cplx(i*volume));

    square_laplacian_gauged(mptr, rhs_cplx, &lapstr_gauged);
  }

  // This should be the matrix. Let's print it to make sure.

  if (volume <= 9)
  {
    std::cout << mat_cplx << "\n";
  }

  // Get the eigenvalues.
  SelfAdjointEigenSolver<cMatrix> eigsolve_cplx(volume);
  eigsolve_cplx.compute(mat_cplx);

  std::cout << "The eigenvalues are:\n" << eigsolve_cplx.eigenvalues() << "\n";

  ///////////////////////////
  // REAL, INDEFINITE CASE //
  ///////////////////////////

  std::cout << "\n\nReal, Indefinite case.\n\n";

  // Allocate a sufficiently gigantic matrix.
  dMatrix mat_real_indef = dMatrix::Zero(volume, volume);

  // Form matrix elements. This is where it's important that
  // dMatrix and cMatrix are column major.
  // I should probably make this safer by using a "Map".
  for (int i = 0; i < volume; i++)
  {
    // Set a point on the rhs for a matrix element.
    zero_vector(rhs_real_indef, volume);
    rhs_real_indef[i] = 1.0;

    // Where we put the result of the matrix element.
    double* mptr = &(mat_real_indef(i*volume));

    square_staggered(mptr, rhs_real_indef, &lapstr);
  }

  // This should be the matrix. Let's print it to make sure.

  if (volume <= 16)
  {
    std::cout << mat_real_indef << "\n";
  }

  // Get the eigenvalues.
  EigenSolver<dMatrix> eigsolve_real_indef(wilson_volume);
  eigsolve_real_indef.compute(mat_real_indef);

  std::cout << "The eigenvalues are:\n" << eigsolve_real_indef.eigenvalues() << "\n";

  //////////////////////////////
  // COMPLEX, INDEFINITE CASE //
  //////////////////////////////

  std::cout << "\n\nComplex, Indefinite case.\n\n";

  // Allocate a sufficiently gigantic matrix.
  cMatrix mat_cplx_indef = cMatrix::Zero(wilson_volume, wilson_volume);

  // Form matrix elements. This is where it's important that
  // dMatrix and cMatrix are column major.
  // I should probably make this safer by using a "Map".
  for (int i = 0; i < wilson_volume; i++)
  {
    // Set a point on the rhs for a matrix element.
    zero_vector(rhs_cplx_indef, wilson_volume);
    rhs_cplx_indef[i] = 1.0;

    // Where we put the result of the matrix element.
    complex<double>* mptr = &(mat_cplx_indef(i*wilson_volume));

    square_wilson_gauged(mptr, rhs_cplx_indef, &lapstr_gauged);
  }

  // This should be the matrix. Let's print it to make sure.

  if (volume <= 9)
  {
    std::cout << mat_cplx_indef << "\n";
  }

  // Get the eigenvalues.
  ComplexEigenSolver<cMatrix> eigsolve_cplx_indef(wilson_volume);
  eigsolve_cplx_indef.compute(mat_cplx_indef);

  std::cout << "The eigenvalues are:\n" << eigsolve_cplx_indef.eigenvalues() << "\n";

  //////////////
  // CLEAN UP //
  //////////////

  // Free the lattice.
  //delete[] lattice;
  deallocate_vector(&rhs_real);
  deallocate_vector(&rhs_real_indef);
  deallocate_vector(&rhs_cplx);
  deallocate_vector(&rhs_cplx_indef);
  deallocate_vector(&gauge_links);
  return 0;
}


