// Copyright (c) 2017 Evan S Weinberg
// Header file for various reductions, such as on timeslices.

#include <complex>
#include <random>

using std::complex; 
using std::polar;

#ifndef QMG_REDUCTIONS
#define QMG_REDUCTIONS

// QMG
#include "lattice/lattice.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Perform a norm2sq on each timeslice of a color vector.
// This admits a more efficient implementation for Nd > 2.
// (Nd == 2 has issues because "y" is intertwined with "x".)
template<typename T>
inline void norm2sq_cv_timeslice(typename RealReducer<T>::type* sum, T* cv, Lattice2D* lat)
{
  int i;
  int x, y, c;
  const int nt = lat->get_dim_mu(lat->get_nd()-1);
  const int size_cv = lat->get_size_cv();
  typename RealReducer<T>::type norm2;

  for (i = 0; i < nt; i++)
    sum[i] = 0.0;

  for (i = 0; i < size_cv; i++)
  {
    norm2 = std::real(std::conj(cv[i])*cv[i]);
    lat->cv_index_to_coord(i, x, y, c);
    sum[y] = sum[y] + norm2;
  }
}

// Perform a re_dot on each timeslice of a color vector.
// This admits a more efficient implementation for Nd > 2.
// (Nd == 2 has issues because "y" is intertwined with "x".)
template<typename T>
inline void redot_cv_timeslice(typename RealReducer<T>::type* sum, T* cv1, T* cv2, Lattice2D* lat)
{
  int i;
  int x, y, c;
  const int nt = lat->get_dim_mu(lat->get_nd()-1);
  const int size_cv = lat->get_size_cv();
  typename RealReducer<T>::type dot_val;

  for (i = 0; i < nt; i++)
    sum[i] = 0.0;

  for (i = 0; i < size_cv; i++)
  {
    dot_val = ComplexBase<T>::real(ComplexBase<T>::conj(cv1[i])*cv2[i]);
    lat->cv_index_to_coord(i, x, y, c);
    sum[y] = sum[y] + dot_val;
  }
}

// Perform a dot on each timeslice of a color vector.
// This admits a more efficient implementation for Nd > 2.
// (Nd == 2 has issues because "y" is intertwined with "x".)
template<typename T>
inline void dot_cv_timeslice(typename Reducer<T>::type* sum, T* cv1, T* cv2, Lattice2D* lat)
{
  int i;
  int x, y, c;
  const int nt = lat->get_dim_mu(lat->get_nd()-1);
  const int size_cv = lat->get_size_cv();
  typename Reducer<T>::type dot_val;

  for (i = 0; i < nt; i++)
    sum[i] = 0.0;

  for (i = 0; i < size_cv; i++)
  {
    dot_val = ComplexBase<T>::conj(cv1[i])*cv2[i];
    lat->cv_index_to_coord(i, x, y, c);
    sum[y] = sum[y] + dot_val;
  }
}

// Create a real gaussian source on a timeslice, for a given dof.
template<typename T> inline void gaussian_wall_source(T* cv, int timeslice, int color, Lattice2D* lat, std::mt19937 &generator, T deviation = 1.0, T mean = 0.0)
{
  int i;
  int x, y, c;
  const int nt = lat->get_dim_mu(lat->get_nd()-1);
  if (timeslice >= nt)
  {
    std::cout << "[QMG-ERROR]: Cannot create gaussian wall source for t < Nt.\n";
    return;
  }

  const int nc = lat->get_nc();
  if (color >= nc)
  {
    std::cout << "[QMG-ERROR]: Cannot create gaussian wall source for color < Nc.\n";
    return;
  }

  const int size_cv = lat->get_size_cv();

  // Generate a normal distribution.
  std::normal_distribution<> dist(0.0, deviation);

  for (i = 0; i < size_cv; i++)
  {
    lat->cv_index_to_coord(i, x, y, c);
    if (c == color && y == timeslice)
    {
      cv[i] = static_cast<T>(mean + dist(generator));
    }
    else
    {
      cv[i] = static_cast<T>(0.0);
    }
  }
}

template<typename T> inline void gaussian_wall_source(std::complex<T>* cv, int timeslice, int color, Lattice2D* lat, std::mt19937 &generator, T deviation = 1.0, T mean = 0.0)
{
  int i;
  int x, y, c;
  const int nt = lat->get_dim_mu(lat->get_nd()-1);
  if (timeslice >= nt)
  {
    std::cout << "[QMG-ERROR]: Cannot create gaussian wall source for t < Nt.\n";
    return;
  }

  const int nc = lat->get_nc();
  if (color >= nc)
  {
    std::cout << "[QMG-ERROR]: Cannot create gaussian wall source for color < Nc.\n";
    return;
  }

  const int size_cv = lat->get_size_cv();

  // Generate a normal distribution.
  std::normal_distribution<> dist(0.0, deviation);

  for (i = 0; i < size_cv; i++)
  {
    lat->cv_index_to_coord(i, x, y, c);
    if (c == color && y == timeslice)
    {
      cv[i] = std::complex<T>(static_cast<T>(mean + dist(generator)), static_cast<T>(0.0));
    }
    else
    {
      cv[i] = static_cast<T>(0.0);
    }
  }
}

#endif // QMG_REDUCTIONS

