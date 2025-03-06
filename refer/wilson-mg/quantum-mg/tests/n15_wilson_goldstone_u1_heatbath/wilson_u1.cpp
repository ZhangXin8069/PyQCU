// Copyright (c) Evan Weinberg 2017
// Generate non-compact quenched U(1) gauge fields
// via heatbath, measure the would-be goldstone
// correlator at zero momentum to extract the mass.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <random>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"
#include "verbosity/verbosity.h"
#include "inverters/inverter_struct.h"
#include "inverters/generic_bicgstab_l.h"

// QMG
#include "lattice/lattice.h"
#include "u1/u1_utils.h"
#include "reductions/reductions.h"
#include "operators/wilson.h"

int main(int argc, char** argv)
{
  cout << setiosflags(ios::fixed) << setprecision(6);

  // Iterator
  int i,j;

  // Random number generator
  std::mt19937 generator (1337u);

  // Some basic fields.
  const int x_len = 64;
  const int y_len = 64;
  const int dof = Wilson2D::get_dof();
  double beta = 6.0; 

  // Information about the Wilson operator.
  double mass = -0.07;

  // Define inverter parameters, inversion struct.
  inversion_verbose_struct* verb = new inversion_verbose_struct(VERB_SUMMARY, std::string("[QMG-WILSON-INFO]: "));
  int max_iter = 4000;
  double tol = 1e-10;
  int bicgstab_l = 6;
  inversion_info invif;

  // How many updates to do between measurements
  int n_update = 100;
  int n_therm = 1000;
  int n_max = 100000;

  // Create a lattice object for the fermion fields.
  Lattice2D* lat = new Lattice2D(x_len, y_len, dof);
  const int cv_size = lat->get_size_cv();

  // Create a lattice object for the u(1) gauge fields.
  Lattice2D* lat_gauge = new Lattice2D(x_len, y_len, 1);

  // Allocate a gauge fields
  complex<double>* gauge_field = allocate_vector<complex<double>>(lat_gauge->get_size_gauge());

  // Also need to keep around phases. If you truncate the phases to [-pi, pi),
  // you don't properly sample the gaussian distribution.
  double* phases = allocate_vector<double>(lat_gauge->get_size_gauge());

  // Create a unit gauge field -> zero phases.
  zero_vector(phases, lat_gauge->get_size_gauge());

  // Create the compact links.
  polar_vector(phases, gauge_field, lat_gauge->get_size_gauge());

  // Create a first Wilson operator. 
  Wilson2D* wilson = new Wilson2D(lat, mass, gauge_field);

  // Create vectors to store sources and propagators.
  complex<double>* src = allocate_vector<complex<double> >(cv_size);
  complex<double>* prop = allocate_vector<complex<double> >(cv_size);

  // Zero the src, drop a point.
  zero_vector(src, cv_size);
  src[lat->cv_coord_to_index(0,0,0)] = 1.0;

  // Create a non-zero initial guess (I get nans when I use a point?)
  gaussian(prop, cv_size, generator);


  // Create a place to accumulate the plaquette.
  double plaq = 0.0;
  double plaq_sq = 0.0;
  int count = 0;

  // Create a place to accumulate the would-be pion correlator. 
  double pion[y_len];
  double pion_sq[y_len];
  double pion_up[y_len];
  double pion_down[y_len];
  for (j = 0; j < y_len; j++)
    pion[j] = 0.0;

  // Do an initial measurement of the plaquette and topology.
  i = 0;
  cout << "[QMG-GAUGE]: "<< i << " " << get_plaquette_u1(gauge_field, lat_gauge) << " " << get_topo_u1(gauge_field, lat_gauge) << "\n";

  // Perform the heatbath update.
  for (i = n_update; i < n_max; i+=n_update)
  {
    // Perform non-compact update.
    heatbath_noncompact_update(phases, lat_gauge, beta, n_update, generator);

    // Get compact links.
    polar_vector(phases, gauge_field, lat_gauge->get_size_gauge());
    double plaq_tmp = std::real(get_plaquette_u1(gauge_field, lat_gauge));
    cout << i << " " << plaq_tmp << " " << get_topo_u1(gauge_field, lat_gauge) << "\n";
    if (i > n_therm)
    {
      plaq += plaq_tmp;
      plaq_sq += plaq_tmp*plaq_tmp;

      // Update the Wilson operator.
      wilson->update_links(gauge_field);
      cout << setiosflags(ios::scientific) << setprecision(6);

      // We need to perform two inversions: one for each parity component.

      ////////////////
      // Parity up: //
      ////////////////
      src[lat->cv_coord_to_index(0,0,0)] = 1.0;
      src[lat->cv_coord_to_index(0,0,1)] = 0.0;
      invif = minv_vector_bicgstab_l(prop, src, cv_size, max_iter, tol, bicgstab_l, apply_stencil_2D_M, (void*)wilson, verb);

      // Compute the norm2sq, update into accumulator.
      norm2sq_cv_timeslice(pion_up, prop, lat);

      // Fold the pion.
      for (j = 1; j < y_len/2; j++)
      {
        double tmp = 0.5*(pion_up[j] + pion_up[y_len-j]);
        pion_up[j] = pion_up[y_len-j] = tmp;
      }

      //////////////////
      // Parity down: //
      //////////////////
      src[lat->cv_coord_to_index(0,0,0)] = 0.0;
      src[lat->cv_coord_to_index(0,0,1)] = 1.0;
      constant_vector(prop, 1.0, cv_size); // For some reason we can't recycle the
                                           // old solution as an initial guess.
      invif = minv_vector_bicgstab_l(prop, src, cv_size, max_iter, tol, bicgstab_l, apply_stencil_2D_M, (void*)wilson, verb);

      // Compute the norm2sq, update into accumulator.
      norm2sq_cv_timeslice(pion_down, prop, lat);

      // Fold the pion.
      for (j = 1; j < y_len/2; j++)
      {
        double tmp = 0.5*(pion_down[j] + pion_down[y_len-j]);
        pion_down[j] = pion_down[y_len-j] = tmp;
      }

      // And we're done.

      // Accumulate pions.
      for (j = 0; j < y_len; j++)
      {
        pion[j] += pion_up[j] + pion_down[j];
        pion_sq[j] += (pion_up[j] + pion_down[j])*(pion_up[j] + pion_down[j]);
      }

      // Reset the output precision.
      cout << setiosflags(ios::fixed) << setprecision(6);

      // An update the counter. 
      count++;

    }
  }

  cout << "[QMG-GAUGE-FINAL]: The plaquette is " << plaq/count << " +/- " << sqrt((plaq_sq/count - plaq*plaq/(count*count))/(count)) << "\n";

  cout << "[QMG-BEGIN-PION]\n";
  // Print the pion correlator.
  for (j = 0; j < y_len; j++)
    cout << j << " " << pion[j]/count << " +/- " << sqrt((pion_sq[j]/count - pion[j]*pion[j]/(count*count))/(count)) << "\n";
  cout << "[QMG-END-PION]\n";

  cout << "[QMG-BEGIN-PION-EFFMASS]\n";
  // Print the pion effmass.
  for (j = 1; j < y_len-1; j++)
  {
    cout << j << " " << std::acosh((pion[j+1]+pion[j-1])/(2.0*pion[j])) << "\n";
  }
  cout << "[QMG-END-PION-EFFMASS]\n";


  // Clean up.
  deallocate_vector(&src);
  deallocate_vector(&prop);
  deallocate_vector(&phases);
  deallocate_vector(&gauge_field);

  delete wilson;
  delete verb; 
  delete lat_gauge; 
  delete lat;
}

