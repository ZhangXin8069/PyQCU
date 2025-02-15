// Copyright (c) 2018 Evan S Weinberg
// Thick restarted and deflated Lanczos.
// Finds some number of exterior largest or
// smallest values, NOT in magnitude.
// (Doesn't try solve the interior problem,
// though there is a way to trick it with the 
// operator (A-sigma)^2 ...)
// Based on arXiv:1512:08135

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <sstream>
#include <complex>
#include <random>
#include <vector>
#include <functional>

// Borrow dense matrix eigenvalue routines.
#include <Eigen/Dense>

#include "blas/generic_vector.h"

#include "../square_laplace.h"

// Operator class
#include "../operator.h"
#include "../poly_operator.h"

// Lanczos
#include "../lanczos.h"
#include "../thick_deflate_lanczos.h"

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

  ///////////////////////////////////
  // Properties of Linear operator //
  ///////////////////////////////////
  double inv_variance = 6.0; // inverse of variance for gaussian non-compact U(1) links.

  // Basic information about the lattice.
  int length = 24;
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


  //////////////////////////////////////
  // Create a linear operator object. //
  //////////////////////////////////////

  FunctionWrapper<complex<double>> lap_fcn(square_laplacian_gauged, &lapstr_gauged, volume);


  //////////////////////////////////////
  // Properties for Lanczos algorithm //
  //////////////////////////////////////

  // Fill a struct
  TRCLStruct lanc_props;
  lanc_props.n_ev = 10; // get 10 eigenvalues
  lanc_props.m = 30; // subspace size of 20
  lanc_props.tol = 1e-10; // lock at a tolerance of 1e-8
  lanc_props.max_restarts = 500; // maximum of 10 restarts
  lanc_props.preserved_space = 10; // space preserved after restart,
                                  // set to -1 to default to m/4+1
  lanc_props.deflate = false; // don't deflate locked eigenvalues
  lanc_props.generator = &generator; // passed by reference
  lanc_props.verbose = false;

  // Get the smallest eigenvalues
  ThickRestartComplexLanczos<double,std::less<double>> lanczos(&lap_fcn, lanc_props);

  //////////////
  // Compute! //
  //////////////

  lanczos.compute();

  // Get number of converged eigenvalues
  int n_converged = lanczos.num_converged();

  // Get the eigenvalues
  double* eigenvalues = new double[n_converged];
  lanczos.ritzvalues(eigenvalues);

  // Print the Ritz values
  std::cout << "The " << n_converged << " converged eigenvalues are:\n";
  for (int i = 0; i < n_converged; i++) {
    std::cout << eigenvalues[i] << "\n";
  }

  // Get converged eigenvectors
  complex<double>** eigenvectors = new complex<double>*[n_converged];
  for (int i = 0; i < n_converged; i++) {
    eigenvectors[i] = allocate_vector<complex<double>>(volume);
  }

  // Get the Ritz vectors
  lanczos.ritzvectors(eigenvectors);

  ///////////////////////////////////////////
  // Do the polynomial accelerated version //
  ///////////////////////////////////////////

  std::cout << "\nPolynomial Accleration\n";

  // Get approximate bounds from a small Lanczos
  const int m_mini = 6;
  SimpleComplexLanczos<double> simp_lanczos(&lap_fcn, m_mini, generator);
  simp_lanczos.compute();
  double approx_eigs[m_mini];
  simp_lanczos.ritzvalues((double*)approx_eigs);
  // print the approximate eigenvalues
  std::cout << "The " << m_mini << " Ritz values are:\n";
  for (int i = 0; i < m_mini; i++) {
    std::cout << approx_eigs[i] << "\n";
  }
  std::cout << "\n";
  double approx_min = approx_eigs[0]; // get the approximate min
  double approx_max = approx_eigs[m_mini-1]*1.2; // and overshoot the max
  std::cout << "The linear op window is " << approx_min << " to " << approx_max << "\n\n";

  // Make a linear interpolation of the laplace op.
  LinearMapToUnit<complex<double>,double> linear_lap(&lap_fcn, approx_min, approx_max);

  // And make the 20th order poly accelerated form
  OneOverOnePlusX<complex<double> > poly_accel(&linear_lap, 50);


  // Fill a struct
  TRCLStruct poly_lanc_props;
  poly_lanc_props.n_ev = 10; // get 10 eigenvalues
  poly_lanc_props.m = 30; // subspace size of 20
  poly_lanc_props.tol = 1e-10; // lock at a tolerance of 1e-8
  poly_lanc_props.max_restarts = 20; // maximum of 10 restarts
  poly_lanc_props.preserved_space = 10; // space preserved after restart,
                                  // set to -1 to default to m/4+1
  poly_lanc_props.deflate = true; // deflate locked eigenvalues
  poly_lanc_props.generator = &generator; // passed by reference
  poly_lanc_props.verbose = false;

  // Get the largest eigenvalues, since we're poly acceling
  ThickRestartComplexLanczos<double,std::greater<double>> poly_lanczos(&poly_accel, poly_lanc_props);
  poly_lanczos.compute();
  int n_poly_converged = poly_lanczos.num_converged();
  std::cout << n_poly_converged << "\n";

  // Get the poly eigenvalues
  double* poly_eigenvalues = new double[n_poly_converged];
  poly_lanczos.ritzvalues(poly_eigenvalues);

  // Print the Ritz values
  std::cout << "The " << n_poly_converged << " converged eigenvalues are:\n";
  for (int i = 0; i < n_poly_converged; i++) {
    std::cout << poly_eigenvalues[i] << "\n";
  }

  // Get the poly eigenvectors
  complex<double>** poly_eigenvectors = new complex<double>*[n_poly_converged];
  for (int i = 0; i < n_poly_converged; i++) {
    poly_eigenvectors[i] = allocate_vector<complex<double>>(volume);
  }

  // Get the Ritz vectors
  poly_lanczos.ritzvectors(poly_eigenvectors);

  // Get eigenvalues of original system
  complex<double>* intermediate = allocate_vector<complex<double>>(volume);
  std::cout << "\nThe " << n_poly_converged << " reconstructed eigenvalues are:\n";
  for (int i = 0; i < n_poly_converged; i++) {
    zero_vector(intermediate, volume);
    lap_fcn(intermediate, poly_eigenvectors[i]);
    double rec_eval = re_dot(poly_eigenvectors[i], intermediate, volume);
    std::cout << rec_eval << "\n";
  }
  

  //////////////////////////
  // Get all eigenvalues! //
  //////////////////////////


  if (volume <= 256) {

    // Let's get the eigenvalues of the full operator!

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

    // Get the eigenvalues.
    SelfAdjointEigenSolver<cMatrix> eigsolve_cplx(volume);
    eigsolve_cplx.compute(mat_cplx);

    std::cout << "The eigenvalues are:\n" << eigsolve_cplx.eigenvalues() << "\n";

    ////////////////////////////////
    // COMPARE LOWEST EIGENVECTOR //
    ////////////////////////////////

    std::cout << "Compare results:\n\n";
    std::cout << "Lanczos Exact Ratio\n";
    for (int i = 0; i < volume; i++) {
      std::cout << eigenvectors[0][i] << " " << eigsolve_cplx.eigenvectors().col(0)(i)
                << " " << eigenvectors[0][i]/eigsolve_cplx.eigenvectors().col(0)(i) << "\n";
    }
  }

  //////////////
  // CLEAN UP //
  //////////////

  delete[] eigenvalues;
  delete[] poly_eigenvalues;
  for (int i = 0; i < n_converged; i++) {
    deallocate_vector(&eigenvectors[i]);
  }
  delete[] eigenvectors;
  for (int i = 0; i < n_poly_converged; i++) {
    deallocate_vector(&poly_eigenvectors[i]);
  }
  delete[] poly_eigenvectors;
  deallocate_vector(&intermediate);

  deallocate_vector(&rhs_cplx);
  deallocate_vector(&gauge_links);
  return 0;
}


