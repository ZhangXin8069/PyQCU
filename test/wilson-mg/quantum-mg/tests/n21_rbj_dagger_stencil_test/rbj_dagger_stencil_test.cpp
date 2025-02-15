// Copyright (c) 2017 Evan S Weinberg
// Test the right block dagger jacobi stencil construction in two ways:
// * Visual inspection for Wilson
// ** The off-diagonal part of the hopping terms should
//    flip sign and rescale via mass, otherwise unchanged.
// * Testing right block jacobi solve by adding
//     gaussian noise to the clover (to make it non-trivial)
// * Testing rbj D^\dagger D solve with CG for Wilson.
// * Testing rbj D D^\dagger solve with CG for Wilson.

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
  
  ////////////////////////////////
  // Test one: Wilson operator. //
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

  // Build the rbj dagger stencil.
  wilson_stencil->build_rbj_dagger_stencil();

  // Print a rbjacobi site.
  std::cout << "\n[WILSON-RBJ-DAGGER-SITE]: \n";
  wilson_stencil->print_stencil_rbj_dagger_site(0, 2);

  // We want to rebuild the next time.
  delete wilson_stencil;  

  /////////////////////////
  // Test two: GCR solve //
  /////////////////////////

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

  // Build the rbj dagger stencil.
  wilson_stencil->build_rbj_dagger_stencil();

  // Try an inversion with original op.
  cout << "\n\n[QMG-TEST]: Test 2, GCR with noisy Wilson2D.\n";

  // Drop a point on the rhs on an even site.
  rhs[lat_wilson->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  double rhs_norm = sqrt(norm2sq(rhs, cv_size));
  
  // Reset double output format.
  cout << setiosflags(ios::scientific) << setprecision(6);

  // Perform a BiCGstab-L inversion.
  inversion_info invif = minv_vector_gcr_restart(lhs, rhs, cv_size, max_iter, tol, restart_freq, apply_stencil_2D_M_rbj_dagger, (void*)wilson_stencil, verb);

  // Check and make sure we get the right answer.
  wilson_stencil->apply_M_rbj_dagger(check, lhs);
  double explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative residual for original noisy Wilson2D is " << explicit_resid << "\n";

  ////////////////////////////////////////////////////////
  // Test three: CGNR Solve, right block preconditioned //
  ////////////////////////////////////////////////////////

  // Zero the vectors.
  zero_vector(rhs, cv_size);
  zero_vector(extra, cv_size);
  zero_vector(lhs, cv_size);
  zero_vector(check, cv_size);
  zero_vector(extra_2, cv_size);

  // Try an inversion.
  cout << "\n\n[QMG-TEST]: Test 3, CGNR with rbjacobi Wilson2D.\n";

  // Drop a point on the rhs on an even site.
  rhs[lat_wilson->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  rhs_norm = sqrt(norm2sq(rhs, cv_size));

  // Apply (D B^{-1})^\dagger.
  wilson_stencil->prepare_M(extra, rhs, QMG_MATVEC_RBJ_MDAGGER_M);
  
  // Reset double output format.
  cout << setiosflags(ios::scientific) << setprecision(6);

  // Perform a CG inversion
  invif = minv_vector_cg(extra_2, extra, cv_size, max_iter, tol, apply_stencil_2D_M_rbjacobi_MDM, (void*)wilson_stencil, verb);

  // Reconstruct the true solution.
  wilson_stencil->reconstruct_M(lhs, extra_2, rhs, QMG_MATVEC_RBJ_MDAGGER_M);

  // Check and make sure we get the right answer.
  wilson_stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative error for RBJ CGNR is " << explicit_resid << "\n";


  ////////////////////////////////////////////////////////
  // Test three: CGNE Solve, right block preconditioned //
  ////////////////////////////////////////////////////////

  // Zero the vectors.
  zero_vector(rhs, cv_size);
  zero_vector(extra, cv_size);
  zero_vector(lhs, cv_size);
  zero_vector(check, cv_size);
  zero_vector(extra_2, cv_size);

  // Try an inversion.
  cout << "\n\n[QMG-TEST]: Test 3, CGNE with rbjacobi Wilson2D.\n";

  // Drop a point on the rhs on an even site.
  rhs[lat_wilson->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  rhs_norm = sqrt(norm2sq(rhs, cv_size));

  // Apply (D B^{-1})^\dagger.
  wilson_stencil->prepare_M(extra, rhs, QMG_MATVEC_RBJ_M_MDAGGER);
  
  // Reset double output format.
  cout << setiosflags(ios::scientific) << setprecision(6);

  // Perform a CG inversion
  invif = minv_vector_cg(extra_2, extra, cv_size, max_iter, tol, apply_stencil_2D_M_rbjacobi_MMD, (void*)wilson_stencil, verb);

  // Reconstruct the true solution.
  wilson_stencil->reconstruct_M(lhs, extra_2, rhs, QMG_MATVEC_RBJ_M_MDAGGER);

  // Check and make sure we get the right answer.
  wilson_stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative error for RBJ CGNE is " << explicit_resid << "\n";


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
