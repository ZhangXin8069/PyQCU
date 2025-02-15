// Copyright (c) 2017 Evan S Weinberg
// Test header for a real and complex operator.

#ifndef SQUARE_LAPLACE_HEADER
#define SQUARE_LAPLACE_HEADER

struct laplace_struct
{
  int length;
  double m_sq;
};

struct laplace_gauged_struct
{
  int length;
  double m_sq;
  std::complex<double>* gauge_links; // size 2*length*length
};

#endif // SQUARE_LAPLACE_HEADER