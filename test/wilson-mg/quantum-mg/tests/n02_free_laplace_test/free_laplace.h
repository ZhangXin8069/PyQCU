// Copyright (c) 2017 Evan S Weinberg
// Test of a square laplace implementation.

#ifndef QMG_FREE_LAPLACE
#define QMG_FREE_LAPLACE

#include <iostream>
#include <complex>
using std::complex;

#include "blas/generic_vector.h"
#include "lattice/lattice.h"
#include "stencil/stencil_2d.h"

// Set up a free laplace.
// This largely just requires overloading the 
// constructor.
struct FreeLaplace2D : public Stencil2D
{
protected:
  // Get rid of copy, assignment.
  FreeLaplace2D(FreeLaplace2D const &);
  FreeLaplace2D& operator=(FreeLaplace2D const &);

public:

  // Base constructor.
  FreeLaplace2D(Lattice2D* in_lat, complex<double> mass_sq)
    : Stencil2D(in_lat, QMG_PIECE_CLOVER_HOPPING, mass_sq, 0.0, 0.0)
  {
    if (lat->get_nc() != 1)
    {
      std::cout << "[QMG-ERROR]: FreeLaplace2D only supports Nc = 1.\n";
      return;
    }

    // The 2D free Laplace stencil is very easy.
    // It's 4 on the clover, -1 on the hopping.
    constant_vector(clover, 4, lat->get_size_cm());
    constant_vector(hopping, -1, lat->get_size_hopping());
  }
};

#endif
