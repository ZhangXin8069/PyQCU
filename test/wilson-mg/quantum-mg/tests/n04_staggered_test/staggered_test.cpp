// Copyright (c) 2017 Evan S Weinberg
// Test of a staggered implementation.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <random>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"
#include "verbosity/verbosity.h"
#include "inverters/inverter_struct.h"
#include "inverters/generic_cg.h"
#include "inverters/generic_gcr.h"

// QMG
#include "lattice/lattice.h"
#include "cshift/cshift_2d.h"
#include "u1/u1_utils.h"

#include "operators/staggered.h"

int main(int argc, char** argv)
{
  if (argc != 4 && argc != 5)
  {
    std::cout << "Error: ./staggered_circle expects three arguments, L, mass, beta.\n";
    return -1;
  }

  // Set output to not be that long.
  cout << setiosflags(ios::fixed) << setprecision(20);

  // Iterators and such.
  int i;

  // Get a random seed from the command line
  unsigned int seed = 1337u;
  if (argc == 5)
    seed = atoi(argv[4]);

  // Random number generator.
  std::mt19937 generator (seed);

  // Basic information for fine level.
  const int x_len = stoi(argv[1]);
  const int y_len = stoi(argv[1]);
  const double beta = stod(argv[3]); 
  const int dof = Staggered2D::get_dof();

  // Mass of the staggered operator
  double mass = stod(argv[2]);

  // Are we testing the free case?
  bool do_free = false;

  // Create a lattice object.
  Lattice2D* lat = new Lattice2D(x_len, y_len, dof);

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

  // Prepare the gauge field.
  Lattice2D* lat_gauge = new Lattice2D(x_len, y_len, 1); // hack...
  complex<double>* gauge_field = allocate_vector<complex<double>>(lat_gauge->get_size_gauge());
  if (do_free)
  {
    unit_gauge_u1(gauge_field, lat_gauge);
  }
  else
  {
    bool need_heatbath = false;
    if (beta == 6.0)
    {
      switch (x_len)
      {
        case 32:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l32t32b60_heatbath.dat");
          break;
        case 64:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l64t64b60_heatbath.dat");
          break;
        case 128:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l128t128b60_heatbath.dat");
          break;
        case 192:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l192t192b60_heatbath.dat");
          break;
        case 256:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l256t256b60_heatbath.dat");
          break;
        default:
          need_heatbath = true;
          break;
      }
    }
    else if (beta == 10.0)
    {
      switch (x_len)
      {
        case 32:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l32t32b100_heatbath.dat");
          break;
        case 64:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l64t64b100_heatbath.dat");
          break;
        case 128:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l128t128b100_heatbath.dat");
          break;
        case 192:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l192t192b100_heatbath.dat");
          break;
        default:
          need_heatbath = true;
          break;
      }
    }
    else
      need_heatbath = true;

    // Hardcode
    need_heatbath = true;

    if (need_heatbath)
    {
      std::cout << "[QMG-NOTE]: L = " << x_len << " beta = " << beta << " requires heatbath generation.\n";

      int n_therm = 4000; // how many heatbath steps to perform.
      int n_meas = 100; // how often to measure the plaquette, topo.
      double* phases = allocate_vector<double>(lat_gauge->get_size_gauge());
      random_uniform(phases, lat_gauge->get_size_gauge(), generator, -3.1415926535, 3.1415926535);
      //zero_vector(phases, lat_gauge->get_size_gauge());
      double plaq = 0.0; // track along the way
      double topo = 0.0;
      for (i = 0; i < n_therm; i += n_meas)
      {
        // Perform non-compact update.
        heatbath_noncompact_update(phases, lat_gauge, beta, n_therm/n_meas, generator);

        // Get compact links.
        polar_vector(phases, gauge_field, lat_gauge->get_size_gauge());

        plaq = std::real(get_plaquette_u1(gauge_field, lat_gauge));
        topo = std::real(get_topo_u1(gauge_field, lat_gauge));
        std::cout << "[QMG-HEATBATH]: Update " << i << " Plaq " << plaq << " Topo " << topo << "\n";
      }

      // Acquire final gauge field.
      polar_vector(phases, gauge_field, lat_gauge->get_size_gauge());

      // Clean up.
      deallocate_vector(&phases);
    }
  }
  delete lat_gauge;

  // Create a gauged laplace stencil.
  Staggered2D* stag_stencil = new Staggered2D(lat, mass, gauge_field);

  // Try an inversion.
  cout << "[QMG-TEST]: Test a matrix inversion on an random source.\n";

  // Drop a point on the rhs on an even site, get norm.
  gaussian(rhs, lat->get_size_cv(), generator);
  double rhs_norm = sqrt(norm2sq(rhs, cv_size));

  // Define inverter parameters.
  int max_iter = 4000;
  double tol = 1e-10;

  // Prepare verbosity struct.
  // By default prints with high verbosity, switch to VERB_SUMMARY or VERB_NONE
  // if you want to see less.
  inversion_verbose_struct* verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-GCR-INFO]: "));

#if 0
  // Perform a GCR inversion.
  inversion_info invif = minv_vector_gcr(lhs, rhs, cv_size, max_iter, tol, apply_stencil_2D_M, (void*)stag_stencil, verb);

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
  stag_stencil->apply_M(check, lhs);
  double explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative error is " << explicit_resid << "\n";
#endif

  // Try an e-o preconditioned solve.
  cout << "[QMG-TEST]: Test an even-odd preconditioned solve.\n";

  // Prepare. Use 'check' has a temporary b_new.
  stag_stencil->prepare_b(check, rhs);

  // Perform a CG inversion.
  verb->verb_prefix = std::string("[QMG-TEST-CG-INFO]: ");
  inversion_info invif = minv_vector_cg(lhs, check, cv_size/2, max_iter, tol, apply_eo_staggered_2D_M, (void*)stag_stencil, verb);

  // Check results.
  if (invif.success == true)
  {
    cout << "[QMG-TEST-ITER]: Algorithm " << invif.name << " took " << invif.iter
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

  // Check and make sure we get the right answer. Reconstruct first.
  stag_stencil->reconstruct_x(lhs, rhs);
  zero_vector(check, cv_size);
  stag_stencil->apply_M(check, lhs);
  double explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative error is " << explicit_resid << "\n";

  // Clean up.
  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&check);
  deallocate_vector(&gauge_field);

  delete verb; 
  delete stag_stencil;
  delete lat;

  return 0;
}
