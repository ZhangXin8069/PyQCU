// Copyright (c) 2017 Evan S Weinberg
// Test the dagger stencil construction in three ways:
// * Visual inspection for Gauge Laplace (no change)
// * Visual inspection for Wilson
// ** The off-diagonal part of the hopping terms should
//    flip sign, otherwise unchanged.
// * Testing D^\dagger D solve with CG for Wilson.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"
#include "verbosity/verbosity.h"
#include "inverters/inverter_struct.h"
#include "inverters/generic_bicgstab_l.h"
#include "inverters/generic_cg.h"

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

  std::cout << "Test 1: The laplace stencil should agree after daggering.\n";

  // Print a site.
  std::cout << "[LAPLACE-SITE]: \n";
  laplace_stencil->print_stencil_site(0, 1);

  // Build the dagger stencil.
  laplace_stencil->build_dagger_stencil();

  // Print a dagger site.
  std::cout << "\n[LAPLACE-DAGGER-SITE]: \n";
  laplace_stencil->print_stencil_dagger_site(0, 1);

  delete laplace_stencil;
  delete lat_laplace;
  
  ////////////////////////////////
  // Test two: Wilson operator. //
  ////////////////////////////////

  Lattice2D* lat_wilson = new Lattice2D(x_len, y_len, Wilson2D::get_dof());

  // Create a stencil.
  Wilson2D* wilson_stencil = new Wilson2D(lat_wilson, mass, gauge_field);

  std::cout << "\n\nTest 2: The clover term should be unchanged, and the hopping term should flip signs only on the off diagonal.\n";

  // Print a site.
  std::cout << "\n[WILSON-SITE]: \n";
  wilson_stencil->print_stencil_site(0, 1);

  // Build the dagger stencil.
  wilson_stencil->build_dagger_stencil();

  // Print a dagger site.
  std::cout << "\n[WILSON-DAGGER-SITE]: \n";
  wilson_stencil->print_stencil_dagger_site(0, 1);

  //////////////////////
  // Test three: CGNR //
  //////////////////////

  // Define inverter parameters (reused for test four)
  int max_iter = 4000;
  double tol = 1e-7;

  // Prepare verbosity struct (reused for test four)
  inversion_verbose_struct* verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-CGNR]: "));

  // Get vector size (reused for test four)
  const int cv_size = lat_wilson->get_size_cv();

  // Prepare some storage (reused for test four)
  complex<double>* rhs = allocate_vector<complex<double>>(cv_size);
  complex<double>* extra = allocate_vector<complex<double>>(cv_size);
  complex<double>* lhs = allocate_vector<complex<double>>(cv_size);
  complex<double>* check = allocate_vector<complex<double>>(cv_size);

  // Zero the vectors.
  zero_vector(rhs, cv_size);
  zero_vector(extra, cv_size);
  zero_vector(lhs, cv_size);
  zero_vector(check, cv_size);

  // Try an inversion.
  cout << "\n\n[QMG-TEST]: Test 3, CGNR with Wilson2D.\n";

  // Drop a point on the rhs on an even site.
  rhs[lat_wilson->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  double rhs_norm = sqrt(norm2sq(rhs, cv_size));

  // Apply D^\dagger.
  wilson_stencil->prepare_M(extra, rhs, QMG_MATVEC_MDAGGER_M);
  
  // Reset double output format.
  cout << setiosflags(ios::scientific) << setprecision(6);

  // Perform a CG inversion
  inversion_info invif = minv_vector_cg(lhs, extra, cv_size, max_iter, tol, apply_stencil_2D_M_dagger_M, (void*)wilson_stencil, verb);

  // Check and make sure we get the right answer.
  wilson_stencil->apply_M(check, lhs);
  double explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative error for CGNR is " << explicit_resid << "\n";

  /////////////////////
  // Test four: CGNE //
  /////////////////////

  // Zero the vectors.
  zero_vector(rhs, cv_size);
  zero_vector(extra, cv_size);
  zero_vector(lhs, cv_size);
  zero_vector(check, cv_size);

  // Update verbosity structure. 
  delete verb;
  verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-CGNE]: "));

  // Try an inversion.
  cout << "\n\n[QMG-TEST]: Test 4, Test CGNE with Wilson2D.\n";

  // Drop a point on the rhs on an even site.
  rhs[lat_wilson->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  rhs_norm = sqrt(norm2sq(rhs, cv_size));

  // Perform a CG inversion
  invif = minv_vector_cg(extra, rhs, cv_size, max_iter, tol, apply_stencil_2D_M_M_dagger, (void*)wilson_stencil, verb);

  // Reconstruct the true solution.
  wilson_stencil->reconstruct_M(lhs, extra, rhs, QMG_MATVEC_M_MDAGGER);

  // Check and make sure we get the right answer.
  wilson_stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative error for CGNE is " << explicit_resid << "\n";

  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&check);
  deallocate_vector(&extra);
  deallocate_vector(&gauge_field);

  delete verb;
  delete wilson_stencil;
  delete lat_wilson;

  return 0;
}
