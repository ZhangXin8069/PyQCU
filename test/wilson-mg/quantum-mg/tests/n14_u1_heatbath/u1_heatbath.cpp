// Copyright (c) Evan Weinberg 2017
// Generate non-compact quenched U(1) gauge fields
// via heatbath. Based heavily on code by Richard Brower.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <random>

using namespace std;

#include "blas/generic_vector.h"
#include "lattice/lattice.h"
#include "u1/u1_utils.h"

int main(int argc, char** argv)
{
  cout << setiosflags(ios::fixed) << setprecision(6);

  // Iterator
  int i;

  // Random number generator
  std::mt19937 generator (1337u);

  // Some basic fields.
  const int x_len = 32;
  const int y_len = 32;
  double beta = 6.0; 

  // How many updates to do between measurements
  int n_update = 100;
  int n_therm = 4000;
  int n_max = 100000;

  // Create a lattice object. 
  Lattice2D* lat = new Lattice2D(x_len, y_len, 1);

  // Allocate a gauge fields
  complex<double>* field1 = allocate_vector<complex<double>>(lat->get_size_gauge());

  // Also need to keep around phases. If you truncate the phases to [-pi, pi),
  // you don't properly sample the gaussian distribution.
  double* phases = allocate_vector<double>(lat->get_size_gauge());

  // Create a unit gauge field -> zero phases.
  zero_vector(phases, lat->get_size_gauge());

  // Create a place to accumulate the plaquette.
  double plaq = 0.0;
  double plaq_sq = 0.0;
  int count = 0;

  // Do an initial measurement of the plaquette and topology.
  i = 0;
  cout << i << " " << get_plaquette_u1(field1, lat) << " " << get_topo_u1(field1, lat) << "\n";

  // Perform the heatbath update.
  for (i = n_update; i < n_max; i+=n_update)
  {
    // Perform non-compact update.
    heatbath_noncompact_update(phases, lat, beta, n_update, generator);

    // Get compact links.
    polar_vector(phases, field1, lat->get_size_gauge());

    //write_gauge_u1(field1, lat, "./output_cfg");

    double plaq_tmp = std::real(get_plaquette_u1(field1, lat));
    cout << i << " " << plaq_tmp << " " << get_topo_u1(field1, lat) << "\n";
    if (i > n_therm)
    {
      plaq += plaq_tmp;
      plaq_sq += plaq_tmp*plaq_tmp; 
      count++;
    }
  }

  cout << "The mean plaquette is " << plaq/count << " +/- " << sqrt((plaq_sq/count - plaq*plaq/(count*count))/(count)) << "\n";

  // Clean up.
  deallocate_vector(&phases);
  deallocate_vector(&field1);

  delete lat;
}

