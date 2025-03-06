// Copyright (c) 2017 Evan S Weinberg
// Test of a square laplace implementation.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <random>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"
#include "interfaces/arpack/generic_arpack.h"

// QMG
#include "lattice/lattice.h"
#include "cshift/cshift_2d.h"
#include "u1/u1_utils.h"

#include "operators/wilson.h"

int main(int argc, char** argv)
{
  // Set output to not be that long.
  cout << setiosflags(ios::fixed) << setprecision(6);

  // Iterators and such.

  // Random number generator
  std::mt19937 generator (1337u);

  // Basic information.
  const int x_len = 16;
  const int y_len = 16;
  const double beta = 6.0;
  const int dof = 2;

  // Staggered specific information.
  // For 64^2, beta = 6.0, eigenvalues go negative around -0.075.
  const double mass = -0.07;

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

    if (need_heatbath)
    {
      std::cout << "[QMG-NOTE]: L = " << x_len << " beta = " << beta << " requires heatbath generation.\n";

      int n_therm = 4000; // how many heatbath steps to perform.
      int n_meas = 100; // how often to measure the plaquette, topo.
      double* phases = allocate_vector<double>(lat_gauge->get_size_gauge());
      zero_vector(phases, lat_gauge->get_size_gauge());
      double plaq = 0.0; // track along the way
      double topo = 0.0;
      for (int i = 0; i < n_therm; i += n_meas)
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
  Wilson2D* wilson_stencil = new Wilson2D(lat, mass, gauge_field);

  // Define some default params.
  int max_iter = 4000;
  double tol = 1e-15;

  //////////////////////////////
  // Get the entire spectrum. //
  //////////////////////////////

  arpack_dcn* arpack;

  complex<double>* eigs;
  complex<double>** evecs;

  if (cv_size <= 2048)
  {
    std::cout << "Computing entire spectrum.\n";
    arpack = new arpack_dcn(cv_size, max_iter, tol, apply_stencil_2D_M, (void*)wilson_stencil);

    eigs = new complex<double>[cv_size];
    evecs = new complex<double>*[cv_size];
    for (int i = 0; i < cv_size; i++)
    {
      evecs[i] = allocate_vector<complex<double> >(cv_size);
    }

    arpack->get_entire_eigensystem(eigs, arpack_dcn::ARPACK_SMALLEST_REAL);

    for (int i = 0; i < cv_size; i++)
      std::cout << "Eigenvalue " << i << " has value " << eigs[i] << "\n";

    for (int i = 0; i < cv_size; i++)
    {
      deallocate_vector(&evecs[i]);
    }
    delete[] evecs;
    delete[] eigs;
    delete arpack; 
  }

  ///////////////////////////////////
  // Get a subset of the spectrum. //
  ///////////////////////////////////

  // Prepare an arpack structure for grabbing some small
  // amount of eigenvalues.
  int n_eigens = 20;
  arpack = new arpack_dcn(cv_size, max_iter, tol,
                apply_stencil_2D_M, (void*)wilson_stencil,
                n_eigens, 3*n_eigens);

  std::cout << "Computing " << n_eigens << " lowest eigenvalues.\n";

  eigs = new complex<double>[n_eigens];

  if(!arpack->prepare_eigensystem(arpack_dcn::ARPACK_SMALLEST_MAGNITUDE, 20))
  {
    cout << "[ERROR]: Znaupd code: " << arpack->get_solve_info().znaupd_code << "\n";
  }
  arpack->get_eigensystem(eigs, arpack_dcn::ARPACK_SMALLEST_MAGNITUDE);

  for (int j = 0; j < n_eigens; j++)
    std::cout << eigs[j] << "\n";
  std::cout << "\n";

  delete eigs;
  delete arpack;

  // Clean up.
  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&check);
  deallocate_vector(&gauge_field);

  delete wilson_stencil;
  delete lat;

  return 0;
}
