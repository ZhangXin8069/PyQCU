// Copyright (c) 2017 Evan S Weinberg
// Header file for templated local matrix operations.

#include <complex>

using std::complex;

#ifndef QLINALG_MATRIX_LOCAL
#define QLINALG_MATRIX_LOCAL

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifdef QLINALG_TEMPLATING
// Do a local mat-vec operation in row-major.
// I should just pull in Eigen for this,
// but why overcomplicate things for now.
// y += A*x, A matrix. y is length nrow, x is length ncol.
template<int nrow, int ncol, typename T> inline void cMATxpy_local(T* __restrict__ mat, T* __restrict__ x, T* __restrict__ y)
{
  for (int i = 0; i < nrow; i++)
  {
    T tmp = static_cast<T>(0.0);
    for (int j = 0; j < ncol; j++)
      tmp += mat[i*ncol+j]*x[j];
    y[i] += tmp;
  }
}

// y = A*x, A matrix. y is length nrow, x is length ncol.
template<int nrow, int ncol, typename T> inline void cMATxy_local(T* __restrict__ mat, T* __restrict__ x, T* __restrict__ y)
{
  for (int i = 0; i < nrow; i++)
  {
    T tmp = static_cast<T>(0.0);
    for (int j = 0; j < ncol; j++)
      tmp += mat[i*ncol+j]*x[j];
    y[i] = tmp;
  }
}

#endif
// Do a local mat-vec operation in row-major.
// I should just pull in Eigen for this,
// but why overcomplicate things for now.
// y += A*x, A matrix. y is length nrow, x is length ncol.
template<typename T> inline void cMATxpy_local(T* __restrict__ mat, T* __restrict__ x, T* __restrict__ y, const int nrow, const int ncol)
{
  for (int i = 0; i < nrow; i++)
  {
    T tmp = static_cast<T>(0.0);
    for (int j = 0; j < ncol; j++)
      tmp += mat[i*ncol+j]*x[j];
    y[i] += tmp;
  }
}

// y = A*x, A matrix. y is length nrow, x is length ncol.
template<typename T> inline void cMATxy_local(T* __restrict__ mat, T* __restrict__ x, T* __restrict__ y, const int nrow, const int ncol)
{
  for (int i = 0; i < nrow; i++)
  {
    T tmp = static_cast<T>(0.0);
    for (int j = 0; j < ncol; j++)
      tmp += mat[i*ncol+j]*x[j];
    y[i] = tmp;
  }
}

// Perform a local square transpose.
template<typename T> inline void cMATtranspose_square_local(T* mat, int ndim)
{
  for (int i = 0; i < ndim; i++)
    for (int j = i+1; j < ndim; j++)
      std::swap(mat[i*ndim+j], mat[j*ndim+i]);
}

// Perform a copy and transpose.
template<typename T> inline void cMATcopy_transpose_square_local(T* mat, T* mat_dest, int ndim)
{
  for (int i = 0; i < ndim; i++)
    for (int j = 0; j < ndim; j++)
      mat_dest[i*ndim+j] = mat[j*ndim+i];
}

// Perform a local hermitian transpose.
template<typename T> inline void cMATconjtrans_square_local(complex<T>* mat, int ndim)
{
  complex<T> tmp; 
  for (int i = 0; i < ndim; i++)
  {
    tmp = mat[i*ndim+i];
    mat[i*ndim+i] = std::conj(tmp);
    for (int j = i+1; j < ndim; j++)
    {
      tmp = std::conj(mat[i*ndim+j]);
      mat[i*ndim+j] = std::conj(mat[j*ndim+i]);
      mat[j*ndim+i] = tmp;
    }
  }
}

// Perform a copy and hermitian conjugate
template<typename T> inline void cMATcopy_conjtrans_square_local(T* mat, T* mat_dest, int ndim)
{
  for (int i = 0; i < ndim; i++)
    for (int j = 0; j < ndim; j++)
      mat_dest[i*ndim+j] = std::conj(mat[j*ndim+i]);
}

// Do a local mat-mat operation in row-major.
// Z = X*Y, square matrix. 
template<typename T> inline void cMATxtMATyMATz_square_local(T* x, T* y, T* z, int ndim)
{
  // Use a kij loop order to maximize cache efficiency.
  int i,j,k;
  T r;

  for (i = 0; i < ndim*ndim; i++)
    z[i] = 0.0;

  for (k = 0; k < ndim; k++)
  {
    for (i = 0; i < ndim; i++)
    {
      r = x[i*ndim+k];
      for (j = 0; j < ndim; j++)
      {
        z[i*ndim+j] += r*y[k*ndim+j];
      }
    }
  }
}


#endif // QLINALG_MATRIX_LOCAL