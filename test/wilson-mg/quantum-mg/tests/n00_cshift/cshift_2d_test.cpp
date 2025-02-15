// Copyright (c) 2017 Evan S Weinberg
// Test of the cshift routines. 

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>

using namespace std;

#include "blas/generic_vector.h"
#include "lattice/lattice.h"
#include "cshift/cshift_2d.h"

int main(int argc, char** argv)
{
  // Set output to not be that long.
  cout << setiosflags(ios::fixed) << setprecision(0);

  // Iterators and such.
  int x, y;

  // Basic information.
  const int x_len = 6;
  const int y_len = 4;
  const int dof = 2;

  // Create a lattice object.
  Lattice2D* lat = new Lattice2D(x_len, y_len, dof);

  // Prepare some storage.
  complex<double>* lcomplex = allocate_vector<complex<double>>(lat->get_volume());
  complex<double>* lcvector = allocate_vector<complex<double>>(lat->get_size_cv());

  // Initialize the vectors.
  for (y = 0; y < lat->get_dim_mu(1); y++)
  {
    for (x = 0; x < lat->get_dim_mu(0); x++)
    {
      lcomplex[lat->coord_to_index(x, y)] = complex<double>(y*lat->get_dim_mu(0) + x + 1);
      
      lcvector[lat->cv_coord_to_index(x, y, 0)] = complex<double>(y*lat->get_dim_mu(0) + x + 1);
      lcvector[lat->cv_coord_to_index(x, y, 1)] = complex<double>(0.0, y*lat->get_dim_mu(0) + x + 1);
    }
  }

  // Test Cshifts.
  complex<double>* lcomplex_shift = allocate_vector<complex<double>>(lat->get_volume());
  complex<double>* lcvector_shift = allocate_vector<complex<double>>(lat->get_size_cv());

  cout << "LatticeComplex\n";
  // Print out vectors.
  for (y = lat->get_dim_mu(1)-1; y >= 0; y--)
  {
    for (x = 0; x < lat->get_dim_mu(0); x++)
    {
      cout << lcomplex[lat->coord_to_index(x, y)] << " ";
    }
    cout << "\n";
  }
  cout << "\n";

  cout << "From +x Cshift\n";
  zero_vector(lcomplex_shift, lat->get_volume());
  cshift(lcomplex_shift, lcomplex, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, 1, lat);
  for (y = lat->get_dim_mu(1)-1; y >= 0; y--)
  {
    for (x = 0; x < lat->get_dim_mu(0); x++)
    {
      cout << lcomplex_shift[lat->coord_to_index(x, y)] << " ";
    }
    cout << "\n";
  }
  cout << "\n";

  cout << "From +y Cshift\n";
  zero_vector(lcomplex_shift, lat->get_volume());
  cshift(lcomplex_shift, lcomplex, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, 1, lat);
  for (y = lat->get_dim_mu(1)-1; y >= 0; y--)
  {
    for (x = 0; x < lat->get_dim_mu(0); x++)
    {
      cout << lcomplex_shift[lat->coord_to_index(x, y)] << " ";
    }
    cout << "\n";
  }
  cout << "\n";

  cout << "From -x Cshift\n";
  zero_vector(lcomplex_shift, lat->get_volume());
  cshift(lcomplex_shift, lcomplex, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, 1, lat);
  for (y = lat->get_dim_mu(1)-1; y >= 0; y--)
  {
    for (x = 0; x < lat->get_dim_mu(0); x++)
    {
      cout << lcomplex_shift[lat->coord_to_index(x, y)] << " ";
    }
    cout << "\n";
  }
  cout << "\n";

  cout << "From -y Cshift\n";
  zero_vector(lcomplex_shift, lat->get_volume());
  cshift(lcomplex_shift, lcomplex, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, 1, lat);
  for (y = lat->get_dim_mu(1)-1; y >= 0; y--)
  {
    for (x = 0; x < lat->get_dim_mu(0); x++)
    {
      cout << lcomplex_shift[lat->coord_to_index(x, y)] << " ";
    }
    cout << "\n";
  }
  cout << "\n";

  cout << "LatticeColorVector\n";
  // Print out vectors.
  for (y = lat->get_dim_mu(1)-1; y >= 0; y--)
  {
    for (x = 0; x < lat->get_dim_mu(0); x++)
    {
      cout << "[" << lcvector[lat->cv_coord_to_index(x, y, 0)] << "," << lcvector[lat->cv_coord_to_index(x, y, 1)] << "] ";
    }
    cout << "\n";
  }
  cout << "\n";

  cout << "From +x Cshift\n";
  zero_vector(lcvector_shift, lat->get_volume());
  cshift(lcvector_shift, lcvector, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, lat->get_nc(), lat);
  for (y = lat->get_dim_mu(1)-1; y >= 0; y--)
  {
    for (x = 0; x < lat->get_dim_mu(0); x++)
    {
      cout << "[" << lcvector_shift[lat->cv_coord_to_index(x, y, 0)] << "," << lcvector_shift[lat->cv_coord_to_index(x, y, 1)] << "] ";
    }
    cout << "\n";
  }
  cout << "\n";

  cout << "From +y Cshift\n";
  zero_vector(lcvector_shift, lat->get_volume());
  cshift(lcvector_shift, lcvector, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, lat->get_nc(), lat);
  for (y = lat->get_dim_mu(1)-1; y >= 0; y--)
  {
    for (x = 0; x < lat->get_dim_mu(0); x++)
    {
      cout << "[" << lcvector_shift[lat->cv_coord_to_index(x, y, 0)] << "," << lcvector_shift[lat->cv_coord_to_index(x, y, 1)] << "] ";
    }
    cout << "\n";
  }
  cout << "\n";

  cout << "From -x Cshift\n";
  zero_vector(lcvector_shift, lat->get_volume());
  cshift(lcvector_shift, lcvector, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, lat->get_nc(), lat);
  for (y = lat->get_dim_mu(1)-1; y >= 0; y--)
  {
    for (x = 0; x < lat->get_dim_mu(0); x++)
    {
      cout << "[" << lcvector_shift[lat->cv_coord_to_index(x, y, 0)] << "," << lcvector_shift[lat->cv_coord_to_index(x, y, 1)] << "] ";
    }
    cout << "\n";
  }
  cout << "\n";

  cout << "From -y Cshift\n";
  zero_vector(lcvector_shift, lat->get_volume());
  cshift(lcvector_shift, lcvector, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, lat->get_nc(), lat);
  for (y = lat->get_dim_mu(1)-1; y >= 0; y--)
  {
    for (x = 0; x < lat->get_dim_mu(0); x++)
    {
      cout << "[" << lcvector_shift[lat->cv_coord_to_index(x, y, 0)] << "," << lcvector_shift[lat->cv_coord_to_index(x, y, 1)] << "] ";
    }
    cout << "\n";
  }
  cout << "\n";


  // Clean up.
  deallocate_vector(&lcomplex);
  deallocate_vector(&lcvector);
  deallocate_vector(&lcomplex_shift);
  deallocate_vector(&lcvector_shift);

  delete lat;

  return 0;
}