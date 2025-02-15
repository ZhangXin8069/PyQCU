// Copyright (c) 2017 Evan S Weinberg
// Test of a square laplace implementation.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"
#include "verbosity/verbosity.h"
#include "inverters/inverter_struct.h"
#include "inverters/generic_cg.h"

// QMG
#include "lattice/lattice.h"
#include "cshift/cshift_2d.h"

#include "free_laplace.h"

int main(int argc, char** argv)
{
  // Set output to not be that long.
  cout << setiosflags(ios::fixed) << setprecision(6);

  // Iterators and such.

  // Basic information.
  const int x_len = 32;
  const int y_len = 24;
  const int dof = 1;

  // Laplace specific information.
  const double m_sq = 0.1*0.1;

  // Create a lattice object.
  Lattice2D* lat = new Lattice2D(x_len, y_len, dof);

  // Create a free laplace stencil.
  FreeLaplace2D* lap_stencil = new FreeLaplace2D(lat, m_sq);

  // Get a vector size.
  const int cv_size = lat->get_size_cv();

  // Prepare some storage.
  complex<double>* rhs = allocate_vector<complex<double>>(cv_size);
  complex<double>* lhs = allocate_vector<complex<double>>(cv_size);
  complex<double>* check = allocate_vector<complex<double>>(cv_size);

  // Zero the vectors.
  zero_vector(rhs, cv_size);
  zero_vector(lhs, cv_size);
  zero_vector(check, cv_size);

  // Apply stencil on even rhs.
  cout << "[QMG-TEST]: Test square laplace on even point.\n";

  // Drop a point on the rhs on an even site.
  rhs[lat->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;

  // Apply stencil!
  lap_stencil->apply_M(lhs, rhs);

  // Check results.
  cout << "[QMG-TEST]: Self: " << lhs[lat->cv_coord_to_index(x_len/2, y_len/2, 0)] << "\n";
  cout << "[QMG-TEST]: +x: " << lhs[lat->cv_coord_to_index(x_len/2+1, y_len/2, 0)] << "\n";
  cout << "[QMG-TEST]: +y: " << lhs[lat->cv_coord_to_index(x_len/2, y_len/2+1, 0)] << "\n";
  cout << "[QMG-TEST]: -x: " << lhs[lat->cv_coord_to_index(x_len/2-1, y_len/2, 0)] << "\n";
  cout << "[QMG-TEST]: -y: " << lhs[lat->cv_coord_to_index(x_len/2, y_len/2-1, 0)] << "\n";

  // Zero the vectors.
  zero_vector(rhs, cv_size);
  zero_vector(lhs, cv_size);

  // Apply stencil on odd rhs.
  cout << "[QMG-TEST]: Test square laplace on odd point.\n";

  // Drop a point on the lhs on an odd site.
  rhs[lat->cv_coord_to_index(x_len/2, y_len/2+1, 0)] = 1.0;

  // Apply stencil!
  lap_stencil->apply_M(lhs, rhs);

  // Check results.
  cout << "[QMG-TEST]: Self: " << lhs[lat->cv_coord_to_index(x_len/2, y_len/2+1, 0)] << "\n";
  cout << "[QMG-TEST]: +x: " << lhs[lat->cv_coord_to_index(x_len/2+1, y_len/2+1, 0)] << "\n";
  cout << "[QMG-TEST]: +y: " << lhs[lat->cv_coord_to_index(x_len/2, y_len/2+1+1, 0)] << "\n";
  cout << "[QMG-TEST]: -x: " << lhs[lat->cv_coord_to_index(x_len/2-1, y_len/2+1, 0)] << "\n";
  cout << "[QMG-TEST]: -y: " << lhs[lat->cv_coord_to_index(x_len/2, y_len/2+1-1, 0)] << "\n";

  // Apply stencil again.
  zero_vector(rhs, cv_size);
  lap_stencil->apply_M(rhs, lhs);

  // Check a few results.
  cout << "[QMG-TEST]: Self: " << rhs[lat->cv_coord_to_index(x_len/2, y_len/2+1, 0)] << "\n";
  cout << "[QMG-TEST]: +x: " << rhs[lat->cv_coord_to_index(x_len/2+1, y_len/2+1, 0)] << "\n";
  cout << "[QMG-TEST]: +2x: " << rhs[lat->cv_coord_to_index(x_len/2+2, y_len/2+1, 0)] << "\n";
  
  // Zero the vectors.
  zero_vector(rhs, cv_size);
  zero_vector(lhs, cv_size);

  // Try an inversion.
  cout << "[QMG-TEST]: Test a matrix inversion on an even point.\n";

  // Drop a point on the rhs on an even site, get norm.
  rhs[lat->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  double rhs_norm = sqrt(norm2sq(rhs, cv_size));

  // Define inverter parameters.
  int max_iter = 4000;
  double tol = 1e-7;

  // Prepare verbosity struct.
  // By default prints with high verbosity, switch to VERB_SUMMARY or VERB_NONE
  // if you want to see less.
  inversion_verbose_struct* verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-CG-INFO]: "));

  // Reset double output format.
  cout << setiosflags(ios::scientific) << setprecision(6);

  // Perform a CG inversion.
  inversion_info invif = minv_vector_cg(lhs, rhs, cv_size, max_iter, tol, apply_stencil_2D_M, (void*)lap_stencil, verb);

  // Check results.
  if (invif.success == true)
  {
    cout << "[QMG-TEST]: Algorithm " << invif.name << " took " << invif.iter
      << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  else // failed, maybe.
  {
    cout << "[QMG-TEST]: Potential Error! Algorithm " << invif.name
      << " took " << invif.iter << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  cout << "[QMG-TEST]: Computing [check] = A [lhs] as a confirmation.\n";
  // Check and make sure we get the right answer.
  lap_stencil->apply_M(check, lhs);
  double explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative error is " << explicit_resid << "\n";

  // Clean up.
  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&check);

  delete verb; 
  delete lap_stencil;
  delete lat;

  return 0;
}