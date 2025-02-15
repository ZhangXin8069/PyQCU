// Copyright (c) 2018 Evan S Weinberg
// Test bounds on the residual of sequential solves

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

#include "square_wilson.h"
#include "inverters/generic_bicgstab.h"

using namespace std; 
using namespace Eigen;

typedef Matrix<std::complex<double>, Dynamic, Dynamic, ColMajor> cMatrix;

int main(int argc, char** argv)
{  

  complex<double> *gauge_links;

  // Set output precision to be long.
  cout << setprecision(10);

  // Inversion info
  inversion_info invif;

  // RNG related things.
  std::mt19937 generator (1337u); // RNG, 1337u is the seed. 
  double inv_variance = 6.0; // inverse of variance for gaussian non-compact U(1) links.

  // Basic information about the lattice.
  int length = 24;
  double m_sq = 0.001;

  // max iter
  int max_iter = 100000;

  // Set tolerances below.I'll nonetheless type the proof up tomorrow, but in any case, I'm feeling good

  // Some start-up.
  int volume = length*length;
  int wilson_volume = 2*volume;
  
  // Create a random compact U(1) link.
  gauge_links = allocate_vector<complex<double>>(2*length*length);
  gaussian_real(gauge_links, 2*length*length, generator, 1.0/inv_variance);
  polar(gauge_links, 2*length*length);

  // Structure which gets passed to the function.
  laplace_gauged_struct lapstr_gauged;
  lapstr_gauged.length = length;
  lapstr_gauged.m_sq = m_sq;
  lapstr_gauged.gauge_links = gauge_links; 

  ///////////////////////////////////////////////////
  // GET THE CONDITION NUMBER OF THE WILSON MATRIX //
  ///////////////////////////////////////////////////

  // Need to get the sqrt of the largest and smallest eigenvalue of the normal op

  double condition_number = 0;

  {
    complex<double> *rhs_cplx;
    complex<double> *inter_cplx;

    // Vectors. 
    rhs_cplx = allocate_vector<complex<double>>(wilson_volume);
    inter_cplx = allocate_vector<complex<double>>(wilson_volume);

    // Zero out the vector.
    zero_vector(rhs_cplx, 2*length*length);
    zero_vector(inter_cplx, 2*length*length);

    // Allocate a sufficiently gigantic matrix.
    cMatrix mat_cplx = cMatrix::Zero(wilson_volume, wilson_volume);

    // Form matrix elements. This is where it's important that
    // dMatrix and cMatrix are column major.
    // I should probably make this safer by using a "Map".
    for (int i = 0; i < wilson_volume; i++)
    {
      // Set a point on the rhs for a matrix element.
      zero_vector(rhs_cplx, wilson_volume);
      rhs_cplx[i] = 1.0;
      zero_vector(inter_cplx, wilson_volume);
      square_wilson_gauged(inter_cplx, rhs_cplx, &lapstr_gauged);

      // Where we put the result of the matrix element.
      complex<double>* mptr = &(mat_cplx(i*wilson_volume));

      square_wilson_dagger_gauged(mptr, inter_cplx, &lapstr_gauged);
    }

    SelfAdjointEigenSolver<cMatrix> eigsolve_cplx(volume);
    eigsolve_cplx.compute(mat_cplx);

    double largest_eval = real(eigsolve_cplx.eigenvalues()(wilson_volume-1));
    double smallest_eval = real(eigsolve_cplx.eigenvalues()(0));
    std::cout << " Largest singular value: " << sqrt(largest_eval) << "\n";
    std::cout << "Smallest singular value: " << sqrt(smallest_eval) << "\n";
    
    condition_number = sqrt(largest_eval/smallest_eval);
    std::cout << "       Condition number: " << condition_number << "\n";

    deallocate_vector(&rhs_cplx);
    deallocate_vector(&inter_cplx);
  }

  ///////////////////////////////////
  // PERFORM SEQUENTIAL INVERSIONS //
  ///////////////////////////////////

  // truly desired tolerance.
  double eps = 1e-10; 

  // tolerances
  //double eps_1 = 1e-3;
  //double eps_2 = 1e-14;

  // bound
  double eps_1 = eps/2;
  double eps_2 = eps/(2*condition_number);

  {
    // Get a rhs b.
    complex<double>* b;
    b = allocate_vector<complex<double>>(wilson_volume);
    gaussian_real(b, wilson_volume, generator, 1.0/inv_variance);

    double bnorm = sqrt(norm2sq(b, wilson_volume));

    // We want to solve M^dag M x = b.
    // Define Y == M^dag, X == M
    // Solve Y X x = b sequentially.
    // First, define y == X x and solve Y y = b.
    std::cout << "\nSolve Y y = b\n";
    complex<double>* y;
    y = allocate_vector<complex<double>>(wilson_volume);
    zero_vector(y, wilson_volume);

    invif = minv_vector_bicgstab(y, b, wilson_volume, max_iter, eps_1, square_wilson_dagger_gauged, &lapstr_gauged);
    if (invif.success == true)
    {
      printf("  Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/bnorm);
    }

    // Next, solve M x = y
    double ynorm = sqrt(norm2sq(y, wilson_volume));
    std::cout << "\nSolve X x = y\n";
    complex<double>* x;
    x = allocate_vector<complex<double>>(wilson_volume);
    zero_vector(x, wilson_volume);

    invif = minv_vector_bicgstab(x, y, wilson_volume, max_iter, eps_2, square_wilson_gauged, &lapstr_gauged);
    if (invif.success == true)
    {
      printf("  Algorithm %s took %d iterations to reach a tolerance of %.8e.\n", invif.name.c_str(), invif.iter, sqrt(invif.resSq)/ynorm);
    }

    // Do the verify.
    complex<double>* tmp;
    tmp = allocate_vector<complex<double>>(wilson_volume);
    zero_vector(tmp, wilson_volume);

    complex<double>* check;
    check = allocate_vector<complex<double>>(wilson_volume);
    zero_vector(check, wilson_volume);

    // Apply M^dag M
    square_wilson_gauged(tmp, x, &lapstr_gauged);
    square_wilson_dagger_gauged(check, tmp, &lapstr_gauged);
    double rel_res = sqrt(diffnorm2sq(check, b, wilson_volume))/bnorm;
    printf("\nThe normal op tolerance is %.8e.\n", rel_res);

    // Check the equality.
    printf("\nThis should be below %.8e.\n", eps_1 + condition_number*eps_2);

    deallocate_vector(&b);
    deallocate_vector(&y);
    deallocate_vector(&x);
    deallocate_vector(&tmp);
    deallocate_vector(&check);

  }

  //////////////
  // CLEAN UP //
  //////////////

  // Free the lattice.
  //delete[] lattice;

  
  deallocate_vector(&gauge_links);
  return 0;
}


