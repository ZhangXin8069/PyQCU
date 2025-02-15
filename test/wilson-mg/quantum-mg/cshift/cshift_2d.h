// Copyright (c) 2017 Evan S Weinberg
// Various routines related to cshifts. 

#ifndef QMG_CSHIFT_2D 
#define QMG_CSHIFT_2D

#include "blas/generic_vector.h"
#include "../lattice/lattice.h"
#include <iostream>
using std::cout; 

// Enum for all possible stencil directions.
enum qmg_cshift_dir
{
  //QMG_CSHIFT_ALL = 0, // default, full stencil.
  QMG_CSHIFT_FROM_0 = 1,   // clover
  QMG_CSHIFT_FROM_XP1 = 2, // +x
  QMG_CSHIFT_FROM_YP1 = 3, // +y
  QMG_CSHIFT_FROM_XM1 = 4, // -x
  QMG_CSHIFT_FROM_YM1 = 5, // -y
  QMG_CSHIFT_FROM_XP2 = 6, // +2x
  QMG_CSHIFT_FROM_XP1YP1 = 10, // +x+y
  QMG_CSHIFT_FROM_YP2 = 7, // +2y
  QMG_CSHIFT_FROM_XM1YP1 = 11, // -x+y
  QMG_CSHIFT_FROM_XM2 = 8, // -2x
  QMG_CSHIFT_FROM_XM1YM1 = 12, // -x-y
  QMG_CSHIFT_FROM_YM2 = 9, // -2y
  QMG_CSHIFT_FROM_XP1YM1 = 13, // +x-y
};

enum qmg_eo
{
  QMG_EO_FROM_EVEN = 1,
  QMG_EO_FROM_ODD = 2,
  QMG_EO_FROM_EVENODD = 3,
};


// cshift functions need to be overhauled for MPI.
// Need halo regions.
// Probably should have thought through writing this correctly
// in the first place. 

// cshift function from even.
template<typename T> void cshift_from_even(T* lhs, T* rhs, qmg_cshift_dir cdir, const int dof_per_site, Lattice2D* lat)
{
  const int dof = dof_per_site;
  
  const int half_size = lat->get_volume()/2;
  const int half_size_dof = half_size*dof_per_site;

  const int half_rowsize = lat->get_dim_mu(0)/2;
  const int half_rowsize_dof = half_rowsize*dof_per_site; 
  switch (cdir)
  {
    case QMG_CSHIFT_FROM_0:
      // I mean, this is silly.
      copy_vector(lhs, rhs, half_size);
      break;
    case QMG_CSHIFT_FROM_XP1:
      // Loop over double rows in y.
      for (int i = 0; i < half_size_dof; i += 2*half_rowsize_dof)
      {
        // Odd y rows (no boundary)
        for (int j = 0; j < half_rowsize_dof; j++)
          lhs[half_size_dof + i + half_rowsize_dof + j] = rhs[i + half_rowsize_dof + j];

        // Even y rows (a boundary)
        for (int j = 0; j < half_rowsize_dof-dof; j++)
          lhs[half_size_dof + i + j] = rhs[i + dof + j];

        // Even y row boundary. Becomes MPI.
        for (int j = 0; j < dof; j++)
          lhs[half_size_dof + i + half_rowsize_dof - dof + j] = rhs[i + j];
      }
      break;
    case QMG_CSHIFT_FROM_XM1:
      // Loop over double rows in y.
      for (int i = 0; i < half_size_dof; i += 2*half_rowsize_dof)
      {
        // Even y rows (no boundary)
        for (int j = 0; j < half_rowsize_dof; j++)
          lhs[half_size_dof + i + j] = rhs[i + j];

        // Odd y rows (a boundary)
        for (int j = dof; j < half_rowsize_dof; j++)
          lhs[half_size_dof + i + half_rowsize_dof + j] = rhs[i + half_rowsize_dof - dof + j];

        // Odd y row boundary. Becomes MPI.
        for (int j = 0; j < dof; j++)
          lhs[half_size_dof + i + half_rowsize_dof + j] = rhs[i + 2*half_rowsize_dof - dof + j];
      }
      break;
    case QMG_CSHIFT_FROM_YP1:
      // Loop over all but periodic y-row.
      for (int i = 0; i < half_size_dof - half_rowsize_dof; i++)
      {
        lhs[half_size_dof + i] = rhs[half_rowsize_dof + i];
      }

      // Loop over periodic y-row. Becomes MPI.
      for (int i = 0; i < half_rowsize_dof; i++)
      {
        lhs[2*half_size_dof - half_rowsize_dof + i] = rhs[i];
      }
      break;
    case QMG_CSHIFT_FROM_YM1:
      // Loop over all but periodic y-row.
      for (int i = half_rowsize_dof; i < half_size_dof; i++)
      {
        lhs[half_size_dof + i] = rhs[- half_rowsize_dof + i];
      }

      // Loop over periodic y row. Becomes MPI.
      for (int i = 0; i < half_rowsize_dof; i++)
      {
        lhs[half_size_dof + i] = rhs[half_size_dof - half_rowsize_dof + i];
      }
      break; 
    case QMG_CSHIFT_FROM_XP2:
    case QMG_CSHIFT_FROM_XP1YP1:
    case QMG_CSHIFT_FROM_YP2:
    case QMG_CSHIFT_FROM_XM1YP1:
    case QMG_CSHIFT_FROM_XM2:
    case QMG_CSHIFT_FROM_XM1YM1:
    case QMG_CSHIFT_FROM_YM2:
    case QMG_CSHIFT_FROM_XP1YM1:
      cout << "[ERROR-QMG]: cshift_from_even does not support distance two stencils yet.\n";
      break; 
  }
}

// cshift function from odd.
template<typename T> void cshift_from_odd(T* lhs, T* rhs, qmg_cshift_dir cdir, int dof_per_site, Lattice2D* lat)
{
  int dof = dof_per_site;

  int half_size = lat->get_volume()/2;
  int half_size_dof = half_size*dof_per_site;

  int half_rowsize = lat->get_dim_mu(0)/2;
  int half_rowsize_dof = half_rowsize*dof_per_site; 
  switch (cdir)
  {
    case QMG_CSHIFT_FROM_0:
      // I mean, this is silly.
      copy_vector(lhs+half_size, rhs+half_size, half_size);
      break;
    case QMG_CSHIFT_FROM_XP1:
      // Loop over double rows in y.
      for (int i = 0; i < half_size_dof; i += 2*half_rowsize_dof)
      {
        // Even y rows (no boundary)
        for (int j = 0; j < half_rowsize_dof; j++)
          lhs[i + j] = rhs[half_size_dof+i+j];

        // Odd y rows (a boundary)
        for (int j = 0; j < half_rowsize_dof-dof; j++)
          lhs[i + half_rowsize_dof + j] = rhs[half_size_dof + i + half_rowsize_dof + dof + j];

        // Odd y row boundary. Becomes MPI.
        for (int j = 0; j < dof; j++)
          lhs[i + 2*half_rowsize_dof - dof + j] = rhs[half_size_dof + i + half_rowsize_dof + j];
      }
      break;
    case QMG_CSHIFT_FROM_XM1:
      // Loop over double rows in y.
      for (int i = 0; i < half_size_dof; i += 2*half_rowsize_dof)
      {
        // Odd y rows (no boundary)
        for (int j = 0; j < half_rowsize_dof; j++)
          lhs[i + half_rowsize_dof + j] = rhs[half_size_dof + i + half_rowsize_dof + j];

        // Even y rows (a boundary)
        for (int j = dof; j < half_rowsize_dof; j++)
          lhs[i + j] = rhs[half_size_dof + i - dof + j];

        // Even y row boundary. Becomes MPI.
        for (int j = 0; j < dof; j++)
        {
          lhs[i + j] = rhs[half_size_dof + i + half_rowsize_dof - dof + j];
        }
      }
      break;
    case QMG_CSHIFT_FROM_YP1:
      // Loop over all but periodic y-row.
      for (int i = 0; i < half_size_dof - half_rowsize_dof; i++)
      {
        lhs[i] = rhs[half_size_dof + half_rowsize_dof + i];
      }

      // Loop over periodic y-row. Becomes MPI.
      for (int i = 0; i < half_rowsize_dof; i++)
      {
        lhs[half_size_dof - half_rowsize_dof + i] = rhs[half_size_dof + i];
      }
      break;
    case QMG_CSHIFT_FROM_YM1:
      // Loop over all but periodic y-row.
      for (int i = half_rowsize_dof; i < half_size_dof; i++)
      {
        lhs[i] = rhs[half_size_dof - half_rowsize_dof + i];
      }

      // Loop over periodic y row. Becomes MPI.
      for (int i = 0; i < half_rowsize_dof; i++)
      {
        lhs[i] = rhs[2*half_size_dof - half_rowsize_dof + i];
      }
      break; 
    case QMG_CSHIFT_FROM_XP2:
    case QMG_CSHIFT_FROM_XP1YP1:
    case QMG_CSHIFT_FROM_YP2:
    case QMG_CSHIFT_FROM_XM1YP1:
    case QMG_CSHIFT_FROM_XM2:
    case QMG_CSHIFT_FROM_XM1YM1:
    case QMG_CSHIFT_FROM_YM2:
    case QMG_CSHIFT_FROM_XP1YM1:
      cout << "[ERROR-QMG]: cshift_from_odd does not support distance two stencils yet.\n";
      break; 
  }
}

// Generic cshift function.
template<typename T> void cshift(T* lhs, T* rhs, qmg_cshift_dir cdir, qmg_eo eo, int dof_per_site, Lattice2D* lat)
{
  if (eo & QMG_EO_FROM_EVEN)
  {
    cshift_from_even(lhs, rhs, cdir, dof_per_site, lat);
  }
  if (eo & QMG_EO_FROM_ODD)
  {
    cshift_from_odd(lhs, rhs, cdir, dof_per_site, lat);
  }

}

#endif