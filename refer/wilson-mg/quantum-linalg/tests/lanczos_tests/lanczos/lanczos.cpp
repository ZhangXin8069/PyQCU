// Copyright (c) 2017 Evan S Weinberg
// Test code for a Hermitian
// Lanczos without restarts, deflation, etc.
// Based on arXiv:1512.08135.

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

#include "../square_laplace.h"

// Operator class
#include "../operator.h"

// Lanczos
#include "../lanczos.h"

using namespace std; 
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, ColMajor> dMatrix;
typedef Matrix<std::complex<double>, Dynamic, Dynamic, ColMajor> cMatrix;

int main(int argc, char** argv)
{  
  complex<double> *rhs_cplx;

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
  
  // Create a random compact U(1) link.
  complex<double>* gauge_links = allocate_vector<complex<double>>(2*length*length);
  gaussian_real(gauge_links, 2*length*length, generator, 1.0/inv_variance);
  polar(gauge_links, 2*length*length);
  
  // Vectors.
  rhs_cplx = allocate_vector<complex<double>>(volume);

  // Zero out the vector.
  zero_vector(rhs_cplx, length*length);

  // Structure which gets passed to the function.
  laplace_gauged_struct lapstr_gauged;
  lapstr_gauged.length = length;
  lapstr_gauged.m_sq = m_sq;
  lapstr_gauged.gauge_links = gauge_links; 

  // Uncomment this to get the free field.
  //constant_vector(gauge_links, 1.0, 2*volume);
  //std::cout << "Free case.\n\n";

  std::cout << "Interacting case.\n\n";


  // Create an object. Wrap the square laplace function for convenience.
  FunctionWrapper<complex<double>> lap_fcn(square_laplacian_gauged, &lapstr_gauged, volume);

  // m-step
  const int m = 20;

  // Create a Lanczos object.
  SimpleComplexLanczos<double> lanczos(&lap_fcn, m, generator);

  // Compute eigenvalues
  lanczos.compute();

  // Get Ritz values
  double* ritzvalues = new double[m];
  lanczos.ritzvalues(ritzvalues);

  // Print the Ritz values
  std::cout << "The Ritz values from a search space of size " << m << " are:\n";
  for (int i = 0; i < m; i++) {
    std::cout << ritzvalues[i] << "\n";
  }

  complex<double>** ritzvectors = new complex<double>*[m];
  for (int i = 0; i < m; i++) {
    ritzvectors[i] = allocate_vector<complex<double>>(volume);
  }

  // Get the Ritz vectors
  lanczos.ritzvectors(ritzvectors);


  // Comparison: Let's get the eigenvalues of the full operator!
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

    lap_fcn(mptr, rhs_cplx);
  }

  // Get the eigenvalues.
  SelfAdjointEigenSolver<cMatrix> eigsolve_cplx(volume);
  eigsolve_cplx.compute(mat_cplx);

  std::cout << "The eigenvalues are:\n" << eigsolve_cplx.eigenvalues() << "\n";

  ////////////////////////////////
  // COMPARE LOWEST EIGENVECTOR //
  ////////////////////////////////

  std::cout << "\n\nCompare results, smallest eigenvalue:\n\n";
  std::cout << "Lanczos Exact Ratio\n";
  for (int i = 0; i < volume; i++) {
    std::cout << ritzvectors[0][i] << " " << eigsolve_cplx.eigenvectors()(i,0)
              << " " << ritzvectors[0][i]/eigsolve_cplx.eigenvectors()(i,0) << "\n";
  }

  /////////////////////////////////
  // COMPARE LARGEST EIGENVECTOR //
  /////////////////////////////////

  std::cout << "\n\nCompare results, largest eigenvalue:\n\n";
  std::cout << "Lanczos Exact Ratio\n";
  for (int i = 0; i < volume; i++) {
    std::cout << ritzvectors[m-1][i] << " " << eigsolve_cplx.eigenvectors()(i,volume-1)
              << " " << ritzvectors[m-1][i]/eigsolve_cplx.eigenvectors()(i,volume-1) << "\n";
  }

  //////////////
  // CLEAN UP //
  //////////////

  delete[] ritzvalues;

  for (int i = 0; i < m; i++) {
    deallocate_vector(&ritzvectors[i]);
  }
  delete[] ritzvectors;

  deallocate_vector(&rhs_cplx);
  deallocate_vector(&gauge_links);
  return 0;
}


