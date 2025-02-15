// Copyright (c) 2017 Evan S Weinberg
// Header file for templated vector operations.

#include <complex>
#include <random>

using std::complex; 
using std::polar;

#include "generic_local_matrix.h"

namespace ESW_QR
{
  #include "qr.h"
}

#ifndef QLINALG_MATRIX
#define QLINALG_MATRIX

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifdef QLINALG_TEMPLATING

// Apply the global mat-vec operation in row-major.
template<typename T> inline void cMATxpy(T* __restrict__ mat, T* __restrict__ x, T* __restrict__ y, const int nelem, const int nrow, const int ncol)
{
  const int mat_vol = nrow*ncol;
  if (nrow == 1 && ncol == 1)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local<1,1>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 2 && ncol == 2)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local<2,2>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 4 && ncol == 4)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local<4,4>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 8 && ncol == 8)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local<8,8>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 10 && ncol == 10)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local<10,10>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 12 && ncol == 12)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local<12,12>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 16 && ncol == 16)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local<16,16>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 20 && ncol == 20)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local<20,20>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 24 && ncol == 24)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local<24,24>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 32 && ncol == 32)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local<32,32>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local(mat+i*mat_vol, x+i*ncol, y+i*nrow, nrow, ncol);
    }
  }
}

template<typename T> inline void cMATxy(T* __restrict__ mat, T* __restrict__ x, T* __restrict__ y, const int nelem, const int nrow, const int ncol)
{
  const int mat_vol = nrow*ncol;
  if (nrow == 1 && ncol == 1)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<1,1>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 2 && ncol == 2)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<2,2>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 4 && ncol == 4)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<4,4>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 8 && ncol == 8)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<8,8>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 10 && ncol == 10)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local<10,10>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 12 && ncol == 12)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<12,12>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 16 && ncol == 16)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<16,16>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 20 && ncol == 20)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<20,20>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 24 && ncol == 24)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<24,24>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 32 && ncol == 32)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<32,32>(mat+i*mat_vol, x+i*ncol, y+i*nrow);
    }
  }
  else
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local(mat+i*mat_vol, x+i*ncol, y+i*nrow, nrow, ncol);
    }
  }
}

template<typename T> inline void cMAT_single_xy(T* __restrict__ mat, T* __restrict__ x, T* __restrict__ y, const int nelem, const int nrow, const int ncol)
{
  if (nrow == 1 && ncol == 1)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<1,1>(mat, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 2 && ncol == 2)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<2,2>(mat, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 4 && ncol == 4)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<4,4>(mat, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 8 && ncol == 8)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<8,8>(mat, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 10 && ncol == 10)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxpy_local<10,10>(mat, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 12 && ncol == 12)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<12,12>(mat, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 16 && ncol == 16)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<16,16>(mat, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 20 && ncol == 20)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<20,20>(mat, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 24 && ncol == 24)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<24,24>(mat, x+i*ncol, y+i*nrow);
    }
  }
  else if (nrow == 32 && ncol == 32)
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local<32,32>(mat, x+i*ncol, y+i*nrow);
    }
  }
  else
  {
    for (int i = 0; i < nelem; i++)
    {
      cMATxy_local(mat, x+i*ncol, y+i*nrow, nrow, ncol);
    }
  }
}

#else

// Apply the global mat-vec operation in row-major.
template<typename T> inline void cMATxpy(T* __restrict__ mat, T* __restrict__ x, T* __restrict__ y, const int nelem, const int nrow, const int ncol)
{
  const int mat_vol = nrow*ncol;
  for (int i = 0; i < nelem; i++)
  {
    cMATxpy_local(mat+i*mat_vol, x+i*ncol, y+i*nrow, nrow, ncol);
  }
}

template<typename T> inline void cMATxy(T* __restrict__ mat, T* __restrict__ x, T* __restrict__ y, const int nelem, const int nrow, const int ncol)
{
  const int mat_vol = nrow*ncol;
  for (int i = 0; i < nelem; i++)
  {
    cMATxy_local(mat+i*mat_vol, x+i*ncol, y+i*nrow, nrow, ncol);
  }
}

template<typename T> inline void cMAT_single_xy(T* __restrict__ mat, T* __restrict__ x, T* __restrict__ y, const int nelem, const int nrow, const int ncol)
{
  for (int i = 0; i < nelem; i++)
  {
    cMATxy_local(mat, x+i*ncol, y+i*nrow, nrow, ncol);
  }
}

#endif

// Transpose the global mat.
template<typename T> inline void cMATtranspose_square(T* mat, int nelem, int ndim)
{
  int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    cMATtranspose_square_local(mat+i*mat_vol, ndim);
  }
}

// Copy + Transpose the global mat.
template<typename T> inline void cMATcopy_transpose_square(T* mat, T* mat_dest, int nelem, int ndim)
{
  int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    cMATcopy_transpose_square_local(mat+i*mat_vol, mat_dest+i*mat_vol, ndim);
  }
}

// Conjugate transpose the global mat.
template<typename T> inline void cMATconjtrans_square(complex<T>* mat, int nelem, int ndim)
{
  int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    cMATconjtrans_square_local(mat+i*mat_vol, ndim);
  }
}

// Conjugate transpose the global mat.
template<typename T> inline void cMATcopy_conjtrans_square(complex<T>* mat, complex<T>* mat_dest, int nelem, int ndim)
{
  int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    cMATcopy_conjtrans_square_local(mat+i*mat_vol, mat_dest+i*mat_vol, ndim);
  }
}

// Do a mat-mat operation in row-major.
// Z = X*Y, square matrix. 
template<typename T> inline void cMATxtMATyMATz_square(T* xmat, T* ymat, T* zmat, int nelem, int ndim)
{
  const int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    cMATxtMATyMATz_square_local(xmat+i*mat_vol, ymat+i*mat_vol, zmat+i*mat_vol, ndim);
  }
}

// Do a mat-mat operation in row-major.
// Z = X*Y, square matrix, X same everywhere
template<typename T> inline void cMATx_single_tMATyMATz_square(T* xmat, T* ymat, T* zmat, int nelem, int ndim)
{
  const int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    cMATxtMATyMATz_square_local(xmat, ymat+i*mat_vol, zmat+i*mat_vol, ndim);
  }
}

// Do a mat-mat operation in row-major.
// Z = X*Y, square matrix, Y same everywhere
template<typename T> inline void cMATxtMATy_single_MATz_square(T* xmat, T* ymat, T* zmat, int nelem, int ndim)
{
  const int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    cMATxtMATyMATz_square_local(xmat+i*mat_vol, ymat, zmat+i*mat_vol, ndim);
  }
}

// Perform a QR decomposition in row-major.
// Complex only.
// xmat -> qmat * rmat
template<typename T> inline void cMATx_do_qr_square(complex<T>* xmat, complex<T>* qmat, complex<T>* rmat, int nelem, int ndim)
{
  const int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    ESW_QR::qr_decomposition(qmat+i*mat_vol, rmat+i*mat_vol, xmat+i*mat_vol, ndim);
  }
}

// Form the inverse of a matrix from a QR decomposition.
// Complex only.
template<typename T> inline void cMATqr_do_xinv_square(complex<T>* qmat, complex<T>* rmat, complex<T>* xinvmat, int nelem, int ndim)
{
  const int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    ESW_QR::matrix_invert_qr(xinvmat+i*mat_vol, qmat+i*mat_vol, rmat+i*mat_vol, ndim);
  }
}

// Compute the determinant from a QR decomposition.
// Complex only.
// Complex only.
template<typename T> inline void cMATqr_do_det_square(complex<T>* qmat, complex<T>* rmat, complex<T>* det, int nelem, int ndim)
{
  const int mat_vol = ndim*ndim;
  for (int i = 0; i < nelem; i++)
  {
    det[i] = ESW_QR::matrix_det_qr(qmat+i*mat_vol, rmat+i*mat_vol, ndim);
  }
}

#endif // QLINALG_MATRIX

// Compute constant mat times vec, repeating the same mat.
