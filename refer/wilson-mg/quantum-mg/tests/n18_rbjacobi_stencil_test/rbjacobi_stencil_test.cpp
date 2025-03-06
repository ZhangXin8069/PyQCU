// Copyright (c) 2017 Evan S Weinberg
// Test the right block jacobi stencil construction in two ways:
// * Visual inspection for Laplace (trivial rescale)
// * Visual inspection for Wilson (trivial rescale)
// * Testing right block jacobi solve by adding
//     gaussian noise to the clover (to make it non-trivial)
// * Testing a right block jacobi schur solve

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"
#include "verbosity/verbosity.h"
#include "inverters/inverter_struct.h"
#include "inverters/generic_gcr.h"

// QMG
#include "lattice/lattice.h"
#include "cshift/cshift_2d.h"
#include "u1/u1_utils.h"

#include "operators/wilson.h"
#include "operators/gaugedlaplace.h"

int main(int argc, char** argv)
{
  // Set output to not be that long.
  cout << setiosflags(ios::scientific) << setprecision(15);

  // Random number generator.
  std::mt19937 generator (1337u);

  // Basic information.
  const int x_len = 32;
  const int y_len = 32;
  const double mass = 0.1;

  // Prepare the gauge field.
  Lattice2D* lat_gauge = new Lattice2D(x_len, y_len, 1);
  complex<double>* gauge_field = allocate_vector<complex<double>>(lat_gauge->get_size_gauge());
  read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l32t32b100_heatbath.dat");
  //unit_gauge_u1(gauge_field, lat_gauge);
  delete lat_gauge;

  ////////////////////////////////////////
  // Test one: gauged laplace operator. //
  ////////////////////////////////////////

  Lattice2D* lat_laplace = new Lattice2D(x_len, y_len, GaugedLaplace2D::get_dof());

  // Create a stencil.
  GaugedLaplace2D* laplace_stencil = new GaugedLaplace2D(lat_laplace, mass*mass, gauge_field);

  std::cout << "Test 1: The laplace stencil should rescale trivially after right block jacobi preconditioning.\n";

  // Print a site.
  std::cout << "[LAPLACE-SITE]: \n";
  laplace_stencil->print_stencil_site(0, 1);

  // Build the rbjacobi stencil.
  laplace_stencil->build_rbjacobi_stencil();

  // Print a rbjacobi site.
  std::cout << "\n[LAPLACE-RBJACOBI-SITE]: \n";
  laplace_stencil->print_stencil_rbjacobi_site(0, 1);

  delete laplace_stencil;
  delete lat_laplace;
  
  ////////////////////////////////
  // Test two: Wilson operator. //
  ////////////////////////////////

  Lattice2D* lat_wilson = new Lattice2D(x_len, y_len, Wilson2D::get_dof());

  // Create a stencil.
  Wilson2D* wilson_stencil = new Wilson2D(lat_wilson, mass, gauge_field);

  std::cout << "\n\nTest 2: The wilson stencil should rescale trivially after right block jacobi preconditioning.\n";

  // Print a site.
  std::cout << "\n[WILSON-SITE]: \n";
  wilson_stencil->print_stencil_site(0, 1);

  // Build the rbjacobi stencil.
  wilson_stencil->build_rbjacobi_stencil();

  // Print a rbjacobi site.
  std::cout << "\n[WILSON-RBJACOBI-SITE]: \n";
  wilson_stencil->print_stencil_rbjacobi_site(0, 1);

  // We want to rebuild the next time.
  delete wilson_stencil;

  ///////////////////////////
  // Test three: GCR solve //
  ///////////////////////////

  // Define inverter parameters (reused for test four)
  int max_iter = 4000;
  double tol = 1e-7;
  int restart_freq = 32; 

  // Prepare verbosity struct (reused for test four)
  inversion_verbose_struct* verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-ORIGINAL]: "));

  // Get vector size, matrix size (reused for test four)
  const int cv_size = lat_wilson->get_size_cv();
  const int cm_size = lat_wilson->get_size_cm();

  // Prepare some storage (reused for test four)
  complex<double>* rhs = allocate_vector<complex<double>>(cv_size);
  complex<double>* extra = allocate_vector<complex<double>>(cv_size);
  complex<double>* lhs = allocate_vector<complex<double>>(cv_size);
  complex<double>* extra_2 = allocate_vector<complex<double>>(cv_size);
  complex<double>* check = allocate_vector<complex<double>>(cv_size);

  // Zero the vectors.
  zero_vector(rhs, cv_size);
  zero_vector(extra, cv_size);
  zero_vector(lhs, cv_size);
  zero_vector(extra_2, cv_size);
  zero_vector(check, cv_size);

  // Create a new Wilson op.
  wilson_stencil = new Wilson2D(lat_wilson, mass, gauge_field);

  // Hack to make the right block jacobi less trivial:
  // Put gaussian noise into clover.
  complex<double>* noise = allocate_vector<complex<double>>(cm_size);
  gaussian(noise, cm_size, generator, 0.1);
  cxpy(noise, wilson_stencil->clover, cm_size);
  deallocate_vector(&noise);

  // Build the rbjacobi stencil.
  wilson_stencil->build_rbjacobi_stencil();

  // Try an inversion with original op.
  cout << "\n\n[QMG-TEST]: Test 3, GCR with noisy Wilson2D.\n";

  // Drop a point on the rhs on an even site.
  rhs[lat_wilson->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  double rhs_norm = sqrt(norm2sq(rhs, cv_size));
  
  // Reset double output format.
  cout << setiosflags(ios::scientific) << setprecision(6);

  // Perform a BiCGstab-L inversion.
  inversion_info invif = minv_vector_gcr_restart(lhs, rhs, cv_size, max_iter, tol, restart_freq, apply_stencil_2D_M, (void*)wilson_stencil, verb);

  // Check and make sure we get the right answer.
  wilson_stencil->apply_M(check, lhs);
  double explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative residual for original noisy Wilson2D is " << explicit_resid << "\n";

  //////////////////////////////////////////////////////
  // Test four: GCR Solve, right block preconditioned //
  //////////////////////////////////////////////////////

  // Prepare vectors. 
  zero_vector(rhs, cv_size);
  zero_vector(extra, cv_size);
  zero_vector(lhs, cv_size);
  zero_vector(check, cv_size);

  // Update verbosity structure. 
  delete verb;
  verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-RBJACOBI]: "));

  // Try an inversion.
  cout << "\n\n[QMG-TEST]: Test 4, GCR with right block preconditioned noisy Wilson2D.\n";

  // Drop a point on the rhs on an even site.
  rhs[lat_wilson->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  rhs[lat_wilson->cv_coord_to_index(x_len/2, y_len/2+1, 0)] = 3.0;
  rhs_norm = sqrt(norm2sq(rhs, cv_size));

  // Perform a GCR inversion
  invif = minv_vector_gcr_restart(extra, rhs, cv_size, max_iter, tol, restart_freq, apply_stencil_2D_M_rbjacobi, (void*)wilson_stencil, verb);

  // Reconstruct the true solution.
  wilson_stencil->reconstruct_M(lhs, extra, rhs, QMG_MATVEC_RIGHT_JACOBI);

  // Check and make sure we get the right answer.
  wilson_stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative residual for right block jacobi is " << explicit_resid << "\n";

  /////////////////////////////////
  // Test five: GCR Solve, schur //
  /////////////////////////////////

  // Prepare vectors. 
  zero_vector(rhs, cv_size);
  zero_vector(extra, cv_size);
  zero_vector(extra_2, cv_size);
  zero_vector(lhs, cv_size);
  zero_vector(check, cv_size);

  // Update verbosity structure. 
  delete verb;
  verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-SCHUR]: "));

  // Try an inversion.
  cout << "\n\n[QMG-TEST]: Test 5, GCR with right block schur preconditioned noisy Wilson2D.\n";

  // Drop a point on the rhs on an even site.
  rhs[lat_wilson->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  rhs[lat_wilson->cv_coord_to_index(x_len/2, y_len/2+1, 0)] = 3.0;
  rhs_norm = sqrt(norm2sq(rhs, cv_size));

  // Prepare the lhs.
  wilson_stencil->prepare_M(extra, rhs, QMG_MATVEC_RIGHT_SCHUR);

  cout << "MEH " << norm2sq(extra, cv_size) << "\n";

  // Perform a GCR inversion
  invif = minv_vector_gcr_restart(extra_2, extra, cv_size/2, max_iter, tol, restart_freq, apply_stencil_2D_M_rbjacobi_schur, (void*)wilson_stencil, verb);

  // Reconstruct the true solution.
  wilson_stencil->reconstruct_M(lhs, extra_2, rhs, QMG_MATVEC_RIGHT_SCHUR);

  // Check and make sure we get the right answer.
  wilson_stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative residual for right block jacobi is " << explicit_resid << "\n";


  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&check);
  deallocate_vector(&extra);
  deallocate_vector(&extra_2);
  deallocate_vector(&gauge_field);

  delete verb;
  delete wilson_stencil;
  delete lat_wilson;

  return 0;
}
