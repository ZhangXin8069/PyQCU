// Test the various U1 routines in u1_utils.cpp.

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

  // Some basic fields.
  const int x_len = 16;
  const int y_len = 16;
  double beta = 3.0;
  double alpha = 0.1; 
  int n_iter = 3; 
  std::mt19937 generator (1337u);

  // Create a lattice object. 
  Lattice2D* lat = new Lattice2D(x_len, y_len, 1);

  // Allocate two gauge fields.
  complex<double>* field1 = allocate_vector<complex<double>>(lat->get_size_gauge());
  complex<double>* field2 = allocate_vector<complex<double>>(lat->get_size_gauge());

  // One gauge transform.
  complex<double>* trans1 = allocate_vector<complex<double>>(lat->get_size_cm());

  // Create a unit field!
  unit_gauge_u1(field1, lat);

  cout << "A unit gauge field has average plaquette " << get_plaquette_u1(field1, lat) << " and topology " << get_topo_u1(field1, lat) << "\n";

  // Create a series of random fields. 
  for (beta = 100; beta > 1e-4; beta*=0.1)
  {
    gauss_gauge_u1(field1, lat, generator, beta);
    cout << "A gauge field with beta " << beta << " has average plaquette " << get_plaquette_u1(field1, lat) << " and topology " << get_topo_u1(field1, lat) << "\n";

    // Smear the field.
    apply_ape_smear_u1(field2, field1, lat, alpha, n_iter);
    cout << " and, after " << n_iter << " iteration(s) of ape smearing with alpha=" << alpha << ", has average plaquette " << get_plaquette_u1(field2, lat) << " and topology " << get_topo_u1(field2, lat) << "\n";
  }

  // Create a hot field.
  rand_gauge_u1(field1, lat, generator);
  cout << "A random gauge field has average plaquette " << get_plaquette_u1(field1, lat) << " and topology " << get_topo_u1(field1, lat) << "\n";

  // Save the gauge field to file.
  cout << "Saving the random gauge field to file...\n";
  write_gauge_u1(field1, lat, "./cfg/cfg16_hot.dat");

  // Load the gauge field.
  cout << "Load the random gauge field from file...\n";
  read_gauge_u1(field2, lat, "./cfg/cfg16_hot.dat");

  // Check the plaquette.
  cout << "The loaded gauge field has average plaquette " << get_plaquette_u1(field2, lat) << " and topology " << get_topo_u1(field2, lat) << "\n";

  // Get a random gauge transform.
  cout << "Getting a random gauge transform.\n";
  rand_trans_u1(trans1, lat, generator);

  // Apply the random gauge transform, check plaquette.
  cout << "Applying the random gauge transform.\n";
  apply_gauge_trans_u1(field2, trans1, lat);

  cout << "After a random gauge transform, the average plaquette is " << get_plaquette_u1(field2, lat) << " and topology " << get_topo_u1(field2, lat) << "\n";

  // Creating a rather smooth field.
  cout << "Creating a rather smooth field for instanton tests.\n";
  gauss_gauge_u1(field2, lat, generator, 6.0);

  cout << "The smooth field has average plaquette " << get_plaquette_u1(field2, lat) << " and topology " << get_topo_u1(field2, lat) << "\n";

  // Create an instanton with charge 1, check plaquette and topology.
  create_instanton_u1(field2, lat, 1, x_len/2, y_len/2);

  cout << "After adding an instanton with charge 1, the average plaquette is " << get_plaquette_u1(field2, lat) << " and topology " << get_topo_u1(field2, lat) << "\n";

  // Clean up.
  deallocate_vector(&field1);
  deallocate_vector(&field2);
  deallocate_vector(&trans1);

  delete lat;
}

