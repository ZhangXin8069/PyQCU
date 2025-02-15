// Copyright (c) 2017 Evan S Weinberg
// Header file for generic 2d stencils, the rock of the code. 

#ifndef QMG_STENCIL_2D
#define QMG_STENCIL_2D

#include <iostream>
#include <complex>
#include "../lattice/lattice.h"
#include "blas/generic_vector.h"
#include "blas/generic_local_matrix.h"
#include "blas/generic_matrix.h"
#include "../cshift/cshift_2d.h"

#ifndef QLINALG_FCN_POINTER
#define QLINALG_FCN_POINTER
typedef void (*matrix_op_real)(double*,double*,void*);
typedef void (*matrix_op_cplx)(complex<double>*,complex<double>*,void*);
#endif

// Indexing offset of, for ex, XP1, is
// lat->get_size_cm() * QMG_DIR_INDEX_XP1.
// Helpful for accessing certain sub-directions
// as a LatticeColorMatrix.
enum stencil_dir_index
{
  QMG_DIR_INDEX_0 = 0,
  QMG_DIR_INDEX_XP1 = 0,
  QMG_DIR_INDEX_YP1 = 1,
  QMG_DIR_INDEX_XM1 = 2,
  QMG_DIR_INDEX_YM1 = 3,
  QMG_DIR_INDEX_XP2 = 0,
  QMG_DIR_INDEX_YP2 = 1,
  QMG_DIR_INDEX_XM2 = 2,
  QMG_DIR_INDEX_YM2 = 3,
  QMG_DIR_INDEX_XP1YP1 = 0,
  QMG_DIR_INDEX_XM1YP1 = 1,
  QMG_DIR_INDEX_XM1YM1 = 2,
  QMG_DIR_INDEX_XP1YM1 = 3,
};

// Enum for pieces of stencil that may exist. 
enum stencil_pieces
{
  QMG_PIECE_CLOVER = 1,
  QMG_PIECE_HOPPING = 2,
  QMG_PIECE_TWOLINK = 4,
  QMG_PIECE_CORNER = 8,
  QMG_PIECE_CLOVER_HOPPING = 3,
  QMG_PIECE_TWOLINK_CORNER = 12,
  QMG_PIECE_ALL = 15,
};

// Enum for if a stencil has chirality or not.
enum chirality_state
{
  QMG_CHIRAL_NO = 0,
  QMG_CHIRAL_YES = 1,
  QMG_CHIRAL_UNKNOWN = 2 // used for coarse operator.
};

// enum for what type of matrix op to prepare, apply, reconstruct.
enum QMGStencilType
{
  QMG_MATVEC_ORIGINAL = 0, // apply original op
  QMG_MATVEC_DAGGER = 1, // apply op dagger
  QMG_MATVEC_RIGHT_JACOBI = 2, // apply right block jacobi
  QMG_MATVEC_RIGHT_SCHUR = 3, // apply schur eo right block jacobi
  QMG_MATVEC_M_MDAGGER = 4, // apply M M^dagger
  QMG_MATVEC_MDAGGER_M = 5, // apply M^dagger M
  QMG_MATVEC_RBJ_DAGGER = 6, // Apply rbj dagger.
  QMG_MATVEC_RBJ_M_MDAGGER = 7, // Apply rbj M M^dagger
  QMG_MATVEC_RBJ_MDAGGER_M = 8, // Apply rbj M^dagger
};

// What's the default naive chirality op, gamma_5 or sigma_1?
enum QMGDefaultChirality
{
  QMG_CHIRALITY_NONE = 0,
  QMG_CHIRALITY_GAMMA_5 = 1,
  QMG_CHIRALITY_SIGMA_1 = 2,
};

// Enum for "what type of" gamma 5 to apply:
// gamma_5, sigma_1, gamma_5^{L/R}, etc.
enum QMGSigmaType
{
  QMG_SIGMA_NONE = 0, // just copy
  QMG_SIGMA_DEFAULT = 1, // default of either gamma_5 or sigma_1
  QMG_GAMMA_5 = 2, // gamma_5. may be overriden by derived functions.
  QMG_SIGMA_1 = 3, // sigma_1
  QMG_GAMMA_5_L_RBJ = 4, // Transfer doubling is via projection. RBJ only. Equals B \gamma_5.
  QMG_GAMMA_5_R_RBJ = 5, // Transfer doubling is via projection. RBJ only. Equals \gamma_5 B^{-1}
};


// Special C function wrappers for stencil applications.
void apply_stencil_2D_M(complex<double>* lhs, complex<double>* rhs, void* extra_data);

void apply_stencil_2D_M_dagger(complex<double>* lhs, complex<double>* rhs, void* extra_data);

void apply_stencil_2D_M_dagger_M(complex<double>* lhs, complex<double>* rhs, void* extra_data);

void apply_stencil_2D_M_M_dagger(complex<double>* lhs, complex<double>* rhs, void* extra_data);

void apply_stencil_2D_M_rbjacobi(complex<double>* lhs, complex<double>* rhs, void* extra_data);

void apply_stencil_2D_M_rbjacobi_schur(complex<double>* lhs, complex<double>* rhs, void* extra_data);

void apply_stencil_2D_M_rbj_dagger(complex<double>* lhs, complex<double>* rhs, void* extra_data);

void apply_stencil_2D_M_rbjacobi_MMD(complex<double>* lhs, complex<double>* rhs, void* extra_data);

void apply_stencil_2D_M_rbjacobi_MDM(complex<double>* lhs, complex<double>* rhs, void* extra_data);


struct Stencil2D
{
protected:
  // Get rid of copy, assignment operator.
  Stencil2D(Stencil2D const &);
  Stencil2D& operator=(Stencil2D const &);

  // Internal memory for cshifts.
  complex<double>* priv_cmatrix;
  complex<double>* priv_cvector;

  // Exposed extra cvector.
  complex<double>* extra_cvector;

  // Temporary space for even-odd solve. Only allocated if needed.
  complex<double>* eo_cvector; 

  // Backups of shift, eo_shift, and dof_shift.
  std::complex<double> shift_backup;
  std::complex<double> eo_shift_backup;
  std::complex<double> dof_shift_backup;

  // Have we swapped to daggered stencils?
  bool swap_dagger;

  // Have we swapped to rbjacobi stencils?
  bool swap_rbjacobi;

  // Have we swapped to rbjacobi dagger stencils?
  bool swap_rbj_dagger;

public:
  // Associated lattice!
  Lattice2D* lat; 
  
  // The following pieces are either allocated or set equal to 0. 
  
  // Nc x Nc x X x Y
  complex<double>* clover;
  
  // Nc x Nc x X x Y x {+X,+Y,-X,-Y} 
  complex<double>* hopping;
  
  // X x Y x Nc x Nc x {+2X, +2Y, -2X, -2Y}
  complex<double>* twolink;
  
  // X x Y x Nc x Nc x {+X+Y, -X+Y, -X-Y, +X-Y}
  complex<double>* corner;  
  
  // Have we generated a lattice?
  bool generated;
  
  // Identity shift, think local mass term. 
  complex<double> shift; 
  
  // Even/odd shift (minus sign on odd), think mass term in g5_staggered
  complex<double> eo_shift; 
  
  // Top/bottom half shift (minus sign on bottom half of color dof, think mass term on coarse g5_staggered.
  // i.e., top half of dof gets a lhs[i] += dof_shift*rhs[i], bottom half gets a rhs[i] -= dof_shift*rhs[i]
  complex<double> dof_shift;

  // Variables related to the dagger stencil. These only get filled
  // on request, by calling "build_dagger_stencil". They have the same
  // size as the other variables defined above.
  bool built_dagger;
  complex<double>* dagger_clover;
  complex<double>* dagger_hopping;
  complex<double>* dagger_twolink;
  complex<double>* dagger_corner;

  // Variables related to the right block jacobi stencil. These only get filled on
  // request by calling "build_right_jacobi_stencil".
  // The "clover" is the trivial identity. Could be made more efficient.
  bool built_rbjacobi;
  complex<double>* rbjacobi_clover;
  complex<double>* rbjacobi_hopping;
  complex<double>* rbjacobi_twolink;
  complex<double>* rbjacobi_corner;

  // We also need an extra spot to hold the inverse of the original clover.
  // We need this for the reconstruct. 
  complex<double>* rbjacobi_cinv;

  // Variables related to the rbj_dagger stencil. These only get filled on
  // request by calling "build_right_jacobi_dagger_stencil".
  // The "clover" is the trivial identity. Could be made more efficient.
  bool built_rbj_dagger;
  complex<double>* rbj_dagger_clover;
  complex<double>* rbj_dagger_hopping;
  complex<double>* rbj_dagger_twolink;
  complex<double>* rbj_dagger_corner;

  complex<double>* rbj_dagger_cinv;

  // Base constructor
  Stencil2D(Lattice2D* in_lat, int pieces, complex<double> in_shift = 0.0, complex<double> in_eo_shift = 0.0, complex<double> in_dof_shift = 0.0)
    : lat(in_lat), shift(in_shift), eo_shift(in_eo_shift), dof_shift(in_dof_shift)
  {
    generated = false;
    
    if (pieces & QMG_PIECE_CLOVER)
    {
      clover = allocate_vector<complex<double>>(lat->get_size_cm());
    }
    else
    {
      clover = 0;
    }
      
    if (pieces & QMG_PIECE_HOPPING)
    {
      hopping = allocate_vector<complex<double>>(lat->get_size_hopping());
    }
    else
    {
      hopping = 0;
    }
    
    if (pieces & QMG_PIECE_TWOLINK)
    {
      twolink = allocate_vector<complex<double>>(lat->get_size_hopping());
    }
    else
    {
      twolink = 0;
    }
      
    if (pieces & QMG_PIECE_CORNER)
    {
      corner = allocate_vector<complex<double>>(lat->get_size_corner());
    }
    else
    {
      corner = 0;
    }

    // Allocate private memory.
    priv_cmatrix = allocate_vector<complex<double>>(lat->get_size_cm());
    priv_cvector = allocate_vector<complex<double>>(lat->get_size_cv());

    // Allocate extra memory.
    extra_cvector = allocate_vector<complex<double>>(lat->get_size_cv());

    // Only allocate the even-odd vector if needed.
    eo_cvector = 0;
    
    // Set all dagger variables to zero.
    built_dagger = false;
    dagger_clover = 0;
    dagger_hopping = 0;
    dagger_twolink = 0;
    dagger_corner = 0;

    // Set all rbjacobi variables to zero.
    built_rbjacobi = false;
    rbjacobi_clover = 0;
    rbjacobi_hopping = 0;
    rbjacobi_twolink = 0;
    rbjacobi_corner = 0;
    rbjacobi_cinv = 0;

    // Set all rbjacobi dagger variables to zero.
    built_rbj_dagger = false;
    rbj_dagger_clover = 0;
    rbj_dagger_hopping = 0;
    rbj_dagger_twolink = 0;
    rbj_dagger_corner = 0;
    rbj_dagger_cinv = 0;

    // Set swaps and backups.
    shift_backup = shift;
    eo_shift_backup = eo_shift;
    dof_shift_backup = dof_shift;
    swap_dagger = false;
    swap_rbjacobi = false; 
    swap_rbj_dagger = false;

  }
  
  virtual ~Stencil2D()
  {
    if (clover != 0) { deallocate_vector(&clover); }
    if (hopping != 0) { deallocate_vector(&hopping); }
    if (twolink != 0) { deallocate_vector(&twolink); }
    if (corner != 0) { deallocate_vector(&corner); }

    // Deallocate private memory.
    deallocate_vector(&priv_cmatrix);
    deallocate_vector(&priv_cvector);

    // Deallocate extra memory.
    deallocate_vector(&extra_cvector);

    // Deallocate eo vector.
    if (eo_cvector != 0) { deallocate_vector(&eo_cvector); }

    if (dagger_clover != 0) { deallocate_vector(&dagger_clover); }
    if (dagger_hopping != 0) { deallocate_vector(&dagger_hopping); }
    if (dagger_twolink != 0) { deallocate_vector(&dagger_twolink); }
    if (dagger_corner != 0) { deallocate_vector(&dagger_corner); }
    built_dagger = false; 

    if (rbjacobi_clover != 0) { deallocate_vector(&rbjacobi_clover); }
    if (rbjacobi_hopping != 0) { deallocate_vector(&rbjacobi_hopping); }
    if (rbjacobi_twolink != 0) { deallocate_vector(&rbjacobi_twolink); }
    if (rbjacobi_corner != 0) { deallocate_vector(&rbjacobi_corner); }
    if (rbjacobi_cinv != 0) { deallocate_vector(&rbjacobi_cinv); }
    built_rbjacobi = false; 

    if (rbj_dagger_clover != 0) { deallocate_vector(&rbj_dagger_clover); }
    if (rbj_dagger_hopping != 0) { deallocate_vector(&rbj_dagger_hopping); }
    if (rbj_dagger_twolink != 0) { deallocate_vector(&rbj_dagger_twolink); }
    if (rbj_dagger_corner != 0) { deallocate_vector(&rbj_dagger_corner); }
    if (rbj_dagger_cinv != 0) { deallocate_vector(&rbj_dagger_cinv); }
    built_rbjacobi = false; 
    
    generated = false; 

  }
  
  // Clear out stencils!
  void clear_stencils()
  {
    if (clover != 0) { zero_vector(clover, lat->get_size_cm()); }
    if (hopping != 0) { zero_vector(hopping, lat->get_size_hopping()); }
    if (twolink != 0) { zero_vector(twolink, lat->get_size_hopping()); }
    if (corner != 0) { zero_vector(corner, lat->get_size_corner()); }

    if (built_dagger)
    {
      if (dagger_clover != 0) { zero_vector(dagger_clover, lat->get_size_cm()); }
      if (dagger_hopping != 0) { zero_vector(dagger_hopping, lat->get_size_hopping()); }
      if (dagger_twolink != 0) { zero_vector(dagger_twolink, lat->get_size_hopping()); }
      if (dagger_corner != 0) { zero_vector(dagger_corner, lat->get_size_corner()); }
      built_dagger = false; 
    }

    if (built_rbjacobi)
    {
      if (rbjacobi_clover != 0) { zero_vector(rbjacobi_clover, lat->get_size_cm()); }
      if (rbjacobi_hopping != 0) { zero_vector(rbjacobi_hopping, lat->get_size_hopping()); }
      if (rbjacobi_twolink != 0) { zero_vector(rbjacobi_twolink, lat->get_size_hopping()); }
      if (rbjacobi_corner != 0) { zero_vector(rbjacobi_corner, lat->get_size_corner()); }
      if (rbjacobi_cinv != 0) { zero_vector(rbjacobi_cinv, lat->get_size_cm()); }
      built_rbjacobi = false; 
    }

    if (built_rbj_dagger)
    {
      if (rbj_dagger_clover != 0) { zero_vector(rbj_dagger_clover, lat->get_size_cm()); }
      if (rbj_dagger_hopping != 0) { zero_vector(rbj_dagger_hopping, lat->get_size_hopping()); }
      if (rbj_dagger_twolink != 0) { zero_vector(rbj_dagger_twolink, lat->get_size_hopping()); }
      if (rbj_dagger_corner != 0) { zero_vector(rbj_dagger_corner, lat->get_size_corner()); }
      built_rbjacobi = false; 
    }
    
    generated = false; 
  }
  
  // Prune pieces of the stencil. This deletes pieces.
  // Need to figure out what that means for dagger stencils...
  void prune_stencils(int pieces)
  {
    if (pieces & QMG_PIECE_CLOVER)
    {
      if (clover != 0) { deallocate_vector(&clover); }
    }
      
    if (pieces & QMG_PIECE_HOPPING)
    {
      if (hopping != 0) { deallocate_vector(&hopping); }
    }
    
    if (pieces & QMG_PIECE_TWOLINK)
    {
      if (twolink != 0) { deallocate_vector(&twolink); }
    }
      
    if (pieces & QMG_PIECE_CORNER)
    {
      if (corner != 0) { deallocate_vector(&corner); }
    }
    
    if (clover == 0 && hopping == 0 && twolink == 0 && corner == 0)
      generated = false;
      
  }
  
  // Prune pieces of the stencil if the max norm is below some tolerance.
  void try_prune_stencils(int pieces, double tol)
  {
    if (pieces & QMG_PIECE_CLOVER)
    {
      if (clover != 0 && norminf(clover, lat->get_size_cm()) < tol) { deallocate_vector(&clover); }
    }
      
    if (pieces & QMG_PIECE_HOPPING)
    {
      if (hopping != 0 && norminf(hopping, lat->get_size_hopping()) < tol) { deallocate_vector(&hopping); }
    }
    
    if (pieces & QMG_PIECE_TWOLINK)
    {
      if (twolink != 0 && norminf(twolink, lat->get_size_hopping()) < tol) { deallocate_vector(&twolink); }
    }
      
    if (pieces & QMG_PIECE_CORNER)
    {
      if (corner != 0 && norminf(corner, lat->get_size_corner()) < tol) { deallocate_vector(&corner); }
    }
    
    if (clover == 0 && hopping == 0 && twolink == 0 && corner == 0)
      generated = false;
  }

  // Get the lattice.
  Lattice2D* get_lattice()
  {
    return lat;
  }

  // Expose internal memory.
  complex<double>* expose_internal_cvector()
  {
    return extra_cvector;
  }

  
  // Print the full stencil at one site.
  void print_stencil_site(int x, int y, string prefix = "")
  {
    int i,j;
    int index; 
    const int nc = lat->get_nc();
      
    if (shift != 0.0)
    {
      cout << prefix << "Shift " << shift << "\n";
    }
      
    if (eo_shift != 0.0)
    {
      cout << prefix << "EO-Shift " << eo_shift << "\n";
    }
      
    if (dof_shift != 0.0)
    {
      cout << prefix << "DOF-Shift " << dof_shift << "\n";
    }
    
    if (clover != 0)
    {
      cout << prefix << "Clover\n";
      index = lat->cm_coord_to_index(x, y, 0, 0);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << clover[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
    }
    
    if (hopping != 0)
    {
      cout << prefix << "Hopping +x\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XP1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << hopping[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Hopping +y\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_YP1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << hopping[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Hopping -x\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XM1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << hopping[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Hopping -y\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_YM1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << hopping[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
    }
      
    if (twolink != 0)
    {
      cout << prefix << "Twolink +2x\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XP2);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << twolink[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Hopping +2y\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_YP2);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << twolink[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Hopping -2x\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XM2);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << twolink[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Hopping -2y\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_YM2);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << twolink[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
    }
      
    if (corner != 0)
    {
      cout << prefix << "Corner +x+y\n";
      index = lat->corner_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XP1YP1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << corner[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Corner -x+y\n";
      index = lat->corner_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XM1YP1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << corner[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Corner -x-y\n";
      index = lat->corner_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XM1YM1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << corner[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Corner +x-y\n";
      index = lat->corner_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XP1YM1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << corner[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
    }
  }

  ////////////////////////
  // Update the shifts. //
  ////////////////////////
  void update_shifts(complex<double> in_shift, complex<double> in_eo_shift, complex<double> in_dof_shift)
  {
    shift = shift_backup = in_shift;
    eo_shift = eo_shift_backup = in_eo_shift;
    dof_shift = dof_shift_backup = in_dof_shift;
  }

  void update_shift(complex<double> in_shift)
  {
    shift = shift_backup = in_shift;
  }

  void update_eo_shift(complex<double> in_eo_shift)
  {
    eo_shift = eo_shift_backup = in_eo_shift;
  }

  void update_dof_shift(complex<double> in_dof_shift)
  {
    dof_shift = dof_shift_backup = in_dof_shift;
  }
      
  // Need functions to apply M_{clover}, M_{eo}, M_{oe}, M_{twolink}, M_{corner}, M_{shift}
  //   and, of course, all.

  // note: doesn't apply all shifts...
  void apply_M_ee(complex<double>* lhs, complex<double>* rhs)
  {
    if (clover == 0)
      return;

    const int nc = lat->get_nc();
    const int half_vol = lat->get_volume()/2;
    const int half_size_cv = lat->get_size_cv()/2;

    cMATxpy(clover, rhs, lhs, half_vol, nc, nc);
    caxpy(shift, rhs, lhs, half_size_cv);
  }

  // note: doesn't apply all shifts...
  void apply_M_oo(complex<double>* lhs, complex<double>* rhs)
  {
    if (clover == 0)
      return;

    const int nc = lat->get_nc();
    const int half_vol = lat->get_volume()/2;
    const int half_size_cv = lat->get_size_cv()/2;
    const int half_size_cm = lat->get_size_cm()/2;

    cMATxpy(clover+half_size_cm, rhs+half_size_cv, lhs+half_size_cv, half_vol, nc, nc);
    caxpy(shift, rhs+half_size_cv, lhs+half_size_cv, half_size_cv);
  }

  void apply_M_clover(complex<double>* lhs, complex<double>* rhs)
  {
    if (clover == 0)
      return;

    const int nc = lat->get_nc();
    const int vol = lat->get_volume();

    cMATxpy(clover, rhs, lhs, vol, nc, nc);
  }

  // Apply the eo part of the stencil.
  void apply_M_eo(complex<double>* lhs, complex<double>* rhs)
  {
    if (hopping == 0)
    {
      cout << "[QMG-WARNING]: Attempt to call 'apply_M_eo' without hopping term.\n";
      return;
    }

    const int nc = lat->get_nc();
    const int half_vol = lat->get_volume()/2;
    const int size_cm = lat->get_size_cm();

    // + xhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_ODD, nc, lat);
    cMATxpy(hopping, priv_cvector, lhs, half_vol, nc, nc);

    // + yhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_ODD, nc, lat);
    cMATxpy(hopping + size_cm, priv_cvector, lhs, half_vol, nc, nc);

    // - xhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_ODD, nc, lat);
    cMATxpy(hopping + 2*size_cm, priv_cvector, lhs, half_vol, nc, nc);

    // - yhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_ODD, nc, lat);
    cMATxpy(hopping + 3*size_cm, priv_cvector, lhs, half_vol, nc, nc);
  }

  // Apply the eo part of the stencil for one direction.
  void apply_M_eo(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (hopping == 0)
    {
      cout << "[QMG-WARNING]: Attempt to call 'apply_M_eo' without hopping term.\n";
      return;
    }

    const int nc = lat->get_nc();
    const int half_vol = lat->get_volume()/2;
    const int size_cm = lat->get_size_cm();

    switch (dir)
    {
      case QMG_DIR_INDEX_XP1: // + xhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_ODD, nc, lat);
        cMATxpy(hopping, priv_cvector, lhs, half_vol, nc, nc);
        break; 

      case QMG_DIR_INDEX_YP1: // + yhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_ODD, nc, lat);
        cMATxpy(hopping + size_cm, priv_cvector, lhs, half_vol, nc, nc);
        break;

      case QMG_DIR_INDEX_XM1: // - xhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_ODD, nc, lat);
        cMATxpy(hopping + 2*size_cm, priv_cvector, lhs, half_vol, nc, nc);
        break;

      case QMG_DIR_INDEX_YM1: // - yhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_ODD, nc, lat);
        cMATxpy(hopping + 3*size_cm, priv_cvector, lhs, half_vol, nc, nc);
        break;
    }
  }
  
  // Apply the oe part of the stencil.
  void apply_M_oe(complex<double>* lhs, complex<double>* rhs)
  {
    if (hopping == 0)
    {
      cout << "[QMG-WARNING]: Attempt to call 'apply_M_oe' without hopping term.\n";
      return;
    }

    const int nc = lat->get_nc();
    const int half_vol = lat->get_volume()/2;
    const int half_cv = lat->get_size_cv()/2;
    const int size_cm = lat->get_size_cm();
    const int half_cm = lat->get_size_cm()/2;

    // + xhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVEN, nc, lat);
    cMATxpy(hopping + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);

    // + yhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVEN, nc, lat);
    cMATxpy(hopping + size_cm + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);

    // - xhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVEN, nc, lat);
    cMATxpy(hopping + 2*size_cm + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);

    // - yhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVEN, nc, lat);
    cMATxpy(hopping + 3*size_cm + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);
  }

  // Apply the oe part of the stencil for one direction.
  void apply_M_oe(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (hopping == 0)
    {
      cout << "[QMG-WARNING]: Attempt to call 'apply_M_oe' without hopping term.\n";
      return;
    }

    const int nc = lat->get_nc();
    const int half_vol = lat->get_volume()/2;
    const int half_cv = lat->get_size_cv()/2;
    const int size_cm = lat->get_size_cm();
    const int half_cm = lat->get_size_cm()/2;

    switch (dir)
    {
      case QMG_DIR_INDEX_XP1: // + xhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVEN, nc, lat);
        cMATxpy(hopping + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);
        break; 

      case QMG_DIR_INDEX_YP1: // + yhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVEN, nc, lat);
        cMATxpy(hopping + size_cm + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);
        break;

      case QMG_DIR_INDEX_XM1: // - xhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVEN, nc, lat);
        cMATxpy(hopping + 2*size_cm + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);
        break;

      case QMG_DIR_INDEX_YM1: // - yhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVEN, nc, lat);
        cMATxpy(hopping + 3*size_cm + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);
        break;
    }
  }

  void apply_M_hopping(complex<double>* lhs, complex<double>* rhs)
  {
    if (hopping != 0)
    {
      apply_M_eo(lhs, rhs);
      apply_M_oe(lhs, rhs);
    }
  }

  void apply_M_hopping(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (hopping != 0)
    {
      apply_M_eo(lhs, rhs, dir);
      apply_M_oe(lhs, rhs, dir);
    }
  }
  
  // void apply_M_twolink(complex<double>* lhs, complex<double>* rhs);
  // void apply_M_corner(complex<double>* lhs, complex<double>* rhs);
  
  // Apply the shifts (think a mass term, maybe with signs a la \gamma_5 D)
  void apply_M_shift(complex<double>* lhs, complex<double>* rhs)
  {
    // Get size of a LatticeColorVector
    int cv = lat->get_size_cv();

    if (lat->get_volume() == 1) // This corner case is annoying.
    {
      if (lat->get_nc() % 2 == 0)
      {
        for (int c = 0; c < lat->get_nc()/2; c++)
        {
          lhs[c] += (shift+eo_shift+dof_shift)*rhs[c];
          lhs[c+lat->get_nc()/2] += (shift+eo_shift-dof_shift)*rhs[c+lat->get_nc()/2];
        }
      }
      else
      {
        for (int c = 0; c < lat->get_nc(); c++)
        {
          lhs[c] += (shift+eo_shift)*rhs[c];
        }
      }
      return;
    }

    // Apply shift and eo_shift to even sites.
    caxpy(shift+eo_shift, rhs, lhs, cv/2);

    // Apply shift and eo_shift to odd sites.
    caxpy(shift-eo_shift, rhs+cv/2, lhs+cv/2, cv/2);

    // There's no good way to do the dof shift with the current mem layout...
    if (dof_shift != 0.0 && lat->get_nc() % 2 == 0) // is dof_shift valid?
    {
      int nc = lat->get_nc();
      for (int c = 0; c < nc/2; c++)
      {
        // Shift top half of degrees of freedom by shift_dof.
        caxpy_stride(dof_shift, rhs, lhs, lat->get_size_cv(), c, nc);

        // Shift bottom half of degrees of freedom by -shift_dof.
        caxpy_stride(-dof_shift, rhs, lhs, lat->get_size_cv(), c+nc/2, nc);
      }
    }
  }

  // lhs = A rhs
  void apply_M(complex<double>* lhs, complex<double>* rhs)
  {
    if (clover != 0)
    {
      apply_M_clover(lhs, rhs);
    }

    if (hopping != 0)
    {
      apply_M_eo(lhs, rhs);
      apply_M_oe(lhs, rhs);
    }

    if (twolink != 0)
    {
      cout << "[QMG-WARNING]: two link stencil not yet supported.\n";
    }

    if (corner != 0)
    {
      cout << "[QMG-WARNING]: corner stencil not yet supported.\n";
    }

    apply_M_shift(lhs, rhs);
  }
    

  // void build_M_dagger_M_stencil(Stencil2D* orig_stenc);


  // Functions to return the shifts.
  complex<double> get_shift()
  {
    return shift;
  }

  complex<double> get_shift_eo()
  {
    return eo_shift;
  }

  complex<double> get_shift_dof()
  {
    return dof_shift;
  }

public:
  // Abstract static functions.

  // Return the number of degrees of freedom.
  // 'i' is for the case of Ls where the number
  // of degrees of freedom may depend on some value
  // passed in.
  static int get_dof(int i = 0)
  {
    return -1;
  }

  // Return true if the operator has a sense of "chirality",
  // false otherwise.
  static chirality_state has_chirality()
  {
    return QMG_CHIRAL_UNKNOWN;
  }

  // Apply gamma5 in place. Default does nothing.
  virtual void gamma5(complex<double>* vec)
  {
    return;
  }

  // Apply gamma5 saved in a vector. 
  virtual void gamma5(complex<double>* g5_vec, complex<double>* vec)
  {
    copy_vector(g5_vec, vec, lat->get_size_cv());
  }
  // A few ways to perform chiral projections.

  // In place project onto up (true), down (false)
  virtual void chiral_projection(complex<double>* vector, bool is_up) = 0;

  // Copy projection onto up, down.
  virtual void chiral_projection_copy(complex<double>* orig, complex<double>* dest, bool is_up) = 0;

  // Copy the down projection into a new vector, perform the up in place.
  virtual void chiral_projection_both(complex<double>* orig_to_up, complex<double>* down) = 0;

  // Apply sigma1 in place. Default does nothing.
  virtual void sigma1(complex<double>* vec)
  {
    return;
  }

  // Apply sigma1 saved in a vector
  virtual void sigma1(complex<double>* s1_vec, complex<double>* vec)
  {
    copy_vector(s1_vec, vec, lat->get_size_cv());
  }

  // What's the default chirality?
  virtual QMGDefaultChirality get_default_chirality() = 0;

  // Apply a certain chiral op.
  void apply_sigma(complex<double>* output, complex<double>* input, QMGSigmaType type = QMG_SIGMA_DEFAULT)
  {
    switch (type)
    {
      case QMG_SIGMA_NONE:
        copy_vector(output, input, lat->get_size_cv());
        break;
      case QMG_SIGMA_DEFAULT:
        switch (get_default_chirality())
        {
          case QMG_CHIRALITY_NONE:
            copy_vector(output, input, lat->get_size_cv());
            break;
          case QMG_CHIRALITY_SIGMA_1:
            sigma1(output, input);
            break;
          case QMG_CHIRALITY_GAMMA_5:
            gamma5(output, input);
            break;
          default:
            copy_vector(output, input, lat->get_size_cv());
            break;
        }
        break;
      case QMG_GAMMA_5:
        gamma5(output, input);
        break;
      case QMG_SIGMA_1:
        sigma1(output, input);
        break;
      case QMG_GAMMA_5_R_RBJ:
        if (!built_rbjacobi)
        {
          std::cout << "[QMG-ERROR]: In apply_sigma, cannot apply QMG_GAMMA_5_L_RBJ without rbjacobi stencil.\n";
          copy_vector(output, input, lat->get_size_cv());
        }
        else
        {
          // Apply B \gamma_5. 
          gamma5(extra_cvector, input);
          cMATxy(clover, extra_cvector, output, lat->get_volume(), lat->get_nc(), lat->get_nc());
          caxpy(shift, extra_cvector, output, lat->get_size_cv()); // also add the mass.
        }
        break;
      case QMG_GAMMA_5_L_RBJ:
        if (!built_rbj_dagger)
        {
          std::cout << "[QMG-ERROR]: In apply_sigma, cannot apply QMG_GAMMA_5_R_RBJ without rbjacobi stencil.\n";
          copy_vector(output, input, lat->get_size_cv());
        }
        else
        {
          // Apply B^{-dagger} \gamma_5. (since we need to left apply \gamma_5 B^{-1})
          gamma5(extra_cvector, input);
          cMATxy(rbj_dagger_cinv, extra_cvector, output, lat->get_volume(), lat->get_nc(), lat->get_nc());
        }
        break;
    }
  }

public:
  //////////////////////////////
  // DAGGER STENCIL FUNCTIONS //
  //////////////////////////////

  void build_dagger_stencil()
  {
    if (built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call build_dagger_stencil, but it's already been called once.\n";
      return;
    }

    const int nc = lat->get_nc();
    const int nc2 = nc*nc;
    const int vol = lat->get_volume();
    const int size_cm = lat->get_size_cm();

    // As with anything else, this only supports one-link stencils for now.
    if (clover != 0)
    {
      dagger_clover = allocate_vector<complex<double>>(lat->get_size_cm());
      cMATcopy_conjtrans_square(clover, dagger_clover, vol, nc);
    }
      
    if (hopping != 0)
    {
      dagger_hopping = allocate_vector<complex<double>>(lat->get_size_hopping());

      // +x
      // The right link is the dagger of the left link of the site from the right.
      cshift(priv_cmatrix, hopping + 2*size_cm, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, nc2, lat);
      cMATcopy_conjtrans_square(priv_cmatrix, dagger_hopping, vol, nc);

      // +y
      // The up link is the dagger of the down link of the site from above.
      cshift(priv_cmatrix, hopping + 3*size_cm, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, nc2, lat);
      cMATcopy_conjtrans_square(priv_cmatrix, dagger_hopping + size_cm, vol, nc);

      // -x
      // The left link is the dagger of the right link of the site from the left.
      cshift(priv_cmatrix, hopping, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, nc2, lat);
      cMATcopy_conjtrans_square(priv_cmatrix, dagger_hopping + 2*size_cm, vol, nc);

      // -y
      // The down link is the dagger of the up link of the site below. 
      cshift(priv_cmatrix, hopping + size_cm, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, nc2, lat);
      cMATcopy_conjtrans_square(priv_cmatrix, dagger_hopping + 3*size_cm, vol, nc);
    }
    
    if (twolink != 0)
    {
      //dagger_twolink = allocate_vector<complex<double>>(lat->get_size_hopping());
      cout << "[QMG-WARNING]: two link stencil not yet supported.\n";
    }
      
    if (corner != 0)
    {
      //dagger_corner = allocate_vector<complex<double>>(lat->get_size_corner());
      cout << "[QMG-WARNING]: corner stencil not yet supported.\n";
    }

    built_dagger = true; 

  }

  // Perform swaps to put dagger stencils in/out of place.
  bool perform_swap_dagger()
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call perform_swap_dagger, but the dagger stencil has not been allocated.\n";
      return false;
    }

    if (!swap_dagger)
    {
      // Swap dagger stencils in.
      // Pointer swaaaap.
      std::swap(clover, dagger_clover);
      std::swap(hopping, dagger_hopping);
      std::swap(twolink, dagger_twolink);
      std::swap(corner, dagger_corner);

      shift = std::conj(shift);
      eo_shift = std::conj(eo_shift);
      dof_shift = std::conj(dof_shift);
      swap_dagger = true;
    }
    else
    {
      // Swap back.
      std::swap(clover, dagger_clover);
      std::swap(hopping, dagger_hopping);
      std::swap(twolink, dagger_twolink);
      std::swap(corner, dagger_corner);
      shift = shift_backup;
      eo_shift = eo_shift_backup;
      dof_shift = dof_shift_backup;
      swap_dagger = false;
    }

    return swap_dagger;
  }

  // Gettin' lazy over here!

  void print_stencil_dagger_site(int x, int y, string prefix = "")
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call print_stencil_dagger_site, but the dagger stencil has not been allocated.\n";
      return;
    }

    perform_swap_dagger();
    print_stencil_site(x, y, prefix);
    perform_swap_dagger();
  }

  
  void apply_M_dagger_clover(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_dagger_clover, but the dagger stencil has not been allocated.\n";
      return;
    }

    if (dagger_clover == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_dagger_clover, but the dagger clover does not exist.\n";
      return;
    }

    perform_swap_dagger();
    apply_M_clover(lhs, rhs);
    perform_swap_dagger();
  }

  void apply_M_dagger_ee(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_dagger_ee, but the dagger stencil has not been allocated.\n";
      return;
    }

    if (dagger_clover == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_dagger_ee, but the dagger clover does not exist.\n";
      return;
    }

    perform_swap_dagger();
    apply_M_ee(lhs, rhs);
    perform_swap_dagger();
  }

  void apply_M_dagger_oo(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_dagger_oo, but the dagger stencil has not been allocated.\n";
      return;
    }

    if (dagger_clover == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_dagger_oo, but the dagger clover does not exist.\n";
      return;
    }

    perform_swap_dagger();
    apply_M_oo(lhs, rhs);
    perform_swap_dagger();
  }

  void apply_M_dagger_eo(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_dagger_eo, but the dagger stencil has not been allocated.\n";
      return;
    }

    if (dagger_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_dagger_eo, but the dagger hopping term does not exist.\n";
      return;
    }

    perform_swap_dagger();
    apply_M_eo(lhs, rhs);
    perform_swap_dagger();
  }

  void apply_M_dagger_eo(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_dagger_eo, but the dagger stencil has not been allocated.\n";
      return;
    }

    if (dagger_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_dagger_eo, but the dagger hopping term does not exist.\n";
      return;
    }

    perform_swap_dagger();
    apply_M_eo(lhs, rhs, dir);
    perform_swap_dagger();
  }
  
  void apply_M_dagger_oe(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_dagger_oe, but the dagger stencil has not been allocated.\n";
      return;
    }

    if (dagger_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_dagger_oe, but the dagger hopping term does not exist.\n";
      return;
    }

    perform_swap_dagger();
    apply_M_oe(lhs, rhs);
    perform_swap_dagger();
  }

  void apply_M_dagger_oe(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_dagger_oe, but the dagger stencil has not been allocated.\n";
      return;
    }

    if (dagger_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_dagger_oe, but the dagger hopping term does not exist.\n";
      return;
    }

    perform_swap_dagger();
    apply_M_oe(lhs, rhs, dir);
    perform_swap_dagger();
  }

  void apply_M_dagger_hopping(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_dagger_hopping, but the dagger stencil has not been allocated.\n";
      return;
    }

    if (dagger_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_dagger_hopping, but the dagger hopping term does not exist.\n";
      return;
    }

    perform_swap_dagger();
    apply_M_hopping(lhs, rhs);
    perform_swap_dagger();
  }

  void apply_M_dagger_hopping(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_dagger_hopping, but the dagger stencil has not been allocated.\n";
      return;
    }

    if (dagger_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_dagger_hopping, but the dagger hopping term does not exist.\n";
      return;
    }

    perform_swap_dagger();
    apply_M_hopping(lhs, rhs);
    perform_swap_dagger();
  }
  
  // void apply_M_dagger_twolink(complex<double>* lhs, complex<double>* rhs);
  // void apply_M_dagger_corner(complex<double>* lhs, complex<double>* rhs);

  void apply_M_dagger_shift(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_dagger_shift, but the dagger stencil has not been allocated.\n";
      return;
    }

    perform_swap_dagger();
    apply_M_shift(lhs, rhs);
    perform_swap_dagger();
  }

  void apply_M_dagger(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_dagger, but the dagger stencil has not been allocated.\n";
      return;
    }

    perform_swap_dagger();
    apply_M(lhs, rhs);
    perform_swap_dagger();
  }

  ///////////////////////////////
  // NORMAL EQUATION FUNCTIONS //
  ///////////////////////////////

  void apply_M_dagger_M(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_dagger_M, but the dagger stencil has not been built.\n";
      return;
    }

    zero_vector(extra_cvector, lat->get_size_cv());
    apply_M(extra_cvector, rhs);
    apply_M_dagger(lhs, extra_cvector);
  }

  void prepare_M_dagger_M(complex<double>* Mdagger_b, complex<double>* b)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call prepare_M_dagger_M, but the dagger stencil has not been built.\n";
      return;
    }

    apply_M_dagger(Mdagger_b, b);
  }

  void apply_M_M_dagger(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_M_dagger, but the dagger stencil has not been built.\n";
      return;
    }

    zero_vector(extra_cvector, lat->get_size_cv());
    apply_M_dagger(extra_cvector, rhs);
    apply_M(lhs, extra_cvector);
  }

  void reconstruct_M_M_dagger(complex<double>* x, complex<double>* y)
  {
    if (!built_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call reconstruct_M_M_dagger, but the dagger stencil has not been built.\n";
      return;
    }

    apply_M_dagger(x, y);
  }

  //////////////////////////////////////////
  // RIGHT BLOCK JACOBI STENCIL FUNCTIONS //
  //////////////////////////////////////////

  void build_rbjacobi_stencil()
  {
    if (built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call build_rbjacobi_stencil, but it's already been called once.\n";
      return;
    }

    const int nc = lat->get_nc();
    const int nc2 = nc*nc;
    const int vol = lat->get_volume();
    const int size_cm = lat->get_size_cm();

    // As with anything else, this only supports one-link stencils for now.

    // First: we need to compute the inverse of the clover.
    // If the clover doesn't exist AND there isn't a mass,
    // we error out.

    if (clover == 0 && shift == 0.0 && eo_shift == 0.0 && dof_shift == 0.0)
    {
      std::cout << "[QMG-ERROR]: Tried to call build_rbjacobi_stencil, but there is no clover term or shift.\n";
      return;
    }

    // Allocate cinv.
    rbjacobi_cinv = allocate_vector<complex<double>>(lat->get_size_cm());

    // Use priv_cmatrix to build the clover plus mass.
    if (clover == 0)
    {
      zero_vector(priv_cmatrix, lat->get_size_cm());
    }
    else
    {
      copy_vector(priv_cmatrix, clover, lat->get_size_cm());
    }

    // Add in the mass.
    // Set up a mass identity stencil.
    complex<double> even_mass_pattern[nc2];
    complex<double> odd_mass_pattern[nc2];
    for (int i = 0; i < nc2; i++)
    {
      even_mass_pattern[i] = odd_mass_pattern[i] = 0.0;
      if (i % (nc+1) == 0) // if on the diagonal
      {
        if (nc % 2 == 0) // if we support dof...
        {
          if (i < nc2/2)
          {
            even_mass_pattern[i] = shift+eo_shift+dof_shift;
            odd_mass_pattern[i] = shift-eo_shift+dof_shift;
          }
          else
          {
            even_mass_pattern[i] = shift+eo_shift-dof_shift;
            odd_mass_pattern[i] = shift-eo_shift-dof_shift;
          }
        } 
        else // no dof.
        {
          even_mass_pattern[i] = shift+eo_shift;
          odd_mass_pattern[i] = shift-eo_shift; 
        }
      }
    }

    if (lat->get_volume() == 1)
    {
      capx_pattern(even_mass_pattern, nc2, priv_cmatrix, 1);
    }
    else
    {
      capx_pattern(even_mass_pattern, nc2, priv_cmatrix, vol/2);
      capx_pattern(odd_mass_pattern, nc2, priv_cmatrix+size_cm/2, vol/2);
    }

    // Good, we've built the clover matrix.
    // Allocate some QR space.
    complex<double>* Qtmp = allocate_vector<complex<double> >(size_cm);
    complex<double>* Rtmp = allocate_vector<complex<double> >(size_cm);

    // Construct rbjacobi_cinv via batch QR.
    cMATx_do_qr_square(priv_cmatrix, Qtmp, Rtmp, vol, nc);
    cMATqr_do_xinv_square(Qtmp, Rtmp, rbjacobi_cinv, vol, nc);

    // Clean up QR. We keep "Rtmp" around.
    deallocate_vector(&Qtmp);

    // Next: the clover is just the identity.
    rbjacobi_clover = allocate_vector<complex<double> >(size_cm);
    zero_vector(rbjacobi_clover, size_cm);
    // Create an identity pattern.
    complex<double> identity_pattern[nc2];
    for (int i = 0; i < nc2; i++)
    {
      identity_pattern[i] = 0.0;
      if (i % (nc+1) == 0)
        identity_pattern[i] = 1.0;
    }
    capx_pattern(identity_pattern, nc2, rbjacobi_clover, vol);

    // Now just go through the rest and right block precondition.
    if (hopping != 0)
    {
      rbjacobi_hopping = allocate_vector<complex<double> >(lat->get_size_hopping());

      // +x
      // Grab the +xhat from the site to the left, right multiply inverse, shift.
      // Use Rtmp as a temporary.
      cshift(priv_cmatrix, hopping, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, nc2, lat);
      cMATxtMATyMATz_square(priv_cmatrix, rbjacobi_cinv, Rtmp, vol, nc);
      cshift(rbjacobi_hopping, Rtmp, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, nc2, lat);

      // + y
      cshift(priv_cmatrix, hopping + size_cm, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, nc2, lat);
      cMATxtMATyMATz_square(priv_cmatrix, rbjacobi_cinv, Rtmp, vol, nc);
      cshift(rbjacobi_hopping + size_cm, Rtmp, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, nc2, lat);
      
      // -x
      cshift(priv_cmatrix, hopping + 2*size_cm, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, nc2, lat);
      cMATxtMATyMATz_square(priv_cmatrix, rbjacobi_cinv, Rtmp, vol, nc);
      cshift(rbjacobi_hopping + 2*size_cm, Rtmp, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, nc2, lat);
      
      // -y
      cshift(priv_cmatrix, hopping + 3*size_cm, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, nc2, lat);
      cMATxtMATyMATz_square(priv_cmatrix, rbjacobi_cinv, Rtmp, vol, nc);
      cshift(rbjacobi_hopping + 3*size_cm, Rtmp, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, nc2, lat);
      
    }

    // Finish cleanup.
    deallocate_vector(&Rtmp);
    
    if (twolink != 0)
    {
      //rbjacobi_twolink = allocate_vector<complex<double>>(lat->get_size_hopping());
      cout << "[QMG-WARNING]: two link stencil not yet supported.\n";
    }
      
    if (corner != 0)
    {
      //rbjacobi_corner = allocate_vector<complex<double>>(lat->get_size_corner());
      cout << "[QMG-WARNING]: corner stencil not yet supported.\n";
    }

    built_rbjacobi = true; 

  }

  /// Perform swaps to put rbjacobi stencils in/out of place.
  bool perform_swap_rbjacobi()
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call perform_swap_rbjacobi, but the rbjacobi stencil has not been allocated.\n";
      return false;
    }

    if (!swap_rbjacobi)
    {
      // Swap rbjacobi stencils in.
      // Pointer swaaaap.
      std::swap(clover, rbjacobi_clover);
      std::swap(hopping, rbjacobi_hopping);
      std::swap(twolink, rbjacobi_twolink);
      std::swap(corner, rbjacobi_corner);
      shift = 0.0;
      eo_shift = 0.0;
      dof_shift = 0.0;
      swap_rbjacobi = true;
    }
    else
    {
      // Swap back.
      std::swap(clover, rbjacobi_clover);
      std::swap(hopping, rbjacobi_hopping);
      std::swap(twolink, rbjacobi_twolink);
      std::swap(corner, rbjacobi_corner);
      shift = shift_backup;
      eo_shift = eo_shift_backup;
      dof_shift = dof_shift_backup;
      swap_rbjacobi = false;
    }

    return swap_rbjacobi;
  }


  void print_stencil_rbjacobi_site(int x, int y, string prefix = "")
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call print_stencil_rbjacobi_site, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    perform_swap_rbjacobi();
    print_stencil_site(x, y, prefix);
    if (clover != 0)
    {
      cout << prefix << "Right Block Jacobi Inv Clover\n";
      int index = lat->cm_coord_to_index(x, y, 0, 0);
      for (int i = 0; i < lat->get_nc(); i++)
      {
        cout << prefix;
        for (int j = 0; j < lat->get_nc(); j++)
        {
          cout << rbjacobi_cinv[index+i*lat->get_nc()+j] << " ";
        }
        cout << "\n";
      }
    }
    perform_swap_rbjacobi();
  }

  
  void apply_M_rbjacobi_clover(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_clover, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    if (rbjacobi_clover == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_clover, but the rbjacobi clover does not exist.\n";
      return;
    }

    // The rbjacobi clover is the identity.
    cxpy(rhs, lhs, lat->get_size_cv());
  }

  void apply_M_rbjacobi_eo(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_eo, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    if (rbjacobi_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_eo, but the rbjacobi hopping term does not exist.\n";
      return;
    }

    perform_swap_rbjacobi();
    apply_M_eo(lhs, rhs);
    perform_swap_rbjacobi();
  }

  void apply_M_rbjacobi_eo(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_eo, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    if (rbjacobi_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_eo, but the rbjacobi hopping term does not exist.\n";
      return;
    }

    perform_swap_rbjacobi();
    apply_M_eo(lhs, rhs, dir);
    perform_swap_rbjacobi();
  }
  
  void apply_M_rbjacobi_oe(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_oe, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    if (rbjacobi_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_oe, but the rbjacobi hopping term does not exist.\n";
      return;
    }

    perform_swap_rbjacobi();
    apply_M_oe(lhs, rhs);
    perform_swap_rbjacobi();
  }

  void apply_M_rbjacobi_oe(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_oe, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    if (rbjacobi_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_oe, but the rbjacobi hopping term does not exist.\n";
      return;
    }

    perform_swap_rbjacobi();
    apply_M_oe(lhs, rhs, dir);
    perform_swap_rbjacobi();
  }

  void apply_M_rbjacobi_hopping(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_hopping, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    if (rbjacobi_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_hopping, but the rbjacobi hopping term does not exist.\n";
      return;
    }

    perform_swap_rbjacobi();
    apply_M_hopping(lhs, rhs);
    perform_swap_rbjacobi();
  }

  void apply_M_rbjacobi_hopping(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_hopping, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    if (rbjacobi_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_hopping, but the rbjacobi hopping term does not exist.\n";
      return;
    }

    perform_swap_rbjacobi();
    apply_M_hopping(lhs, rhs);
    perform_swap_rbjacobi();
  }
  
  // void apply_M_rbjacobi_twolink(complex<double>* lhs, complex<double>* rhs);
  // void apply_M_rbjacobi_corner(complex<double>* lhs, complex<double>* rhs);

  void apply_M_rbjacobi_shift(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_shift, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    // rbj has no shift.

    return;
  }

  void apply_M_rbjacobi(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    apply_M_rbjacobi_clover(lhs, rhs);

    if (hopping != 0)
    {
      apply_M_rbjacobi_eo(lhs, rhs);
      apply_M_rbjacobi_oe(lhs, rhs);
    }

    if (twolink != 0)
    {
      cout << "[QMG-WARNING]: two link stencil not yet supported.\n";
    }

    if (corner != 0)
    {
      cout << "[QMG-WARNING]: corner stencil not yet supported.\n";
    }

    apply_M_rbjacobi_shift(lhs, rhs);
  }

  // Special for the right block jacobi, apply the clover inverse.
  void apply_M_rbjacobi_cinv(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_cinv, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    if (rbjacobi_cinv == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_cinv, but the rbjacobi cinv does not exist.\n";
      return;
    }

    // Special swaaaap.
    std::swap(clover, rbjacobi_cinv);
    apply_M_clover(lhs, rhs);
    std::swap(clover, rbjacobi_cinv);
  }

  // Reconstruct right block jacobi solve, which is just applying the above function.
  // Sort of redundant. 
  void reconstruct_M_rbjacobi(complex<double>* x, complex<double>* y)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call reconstruct_M_rbjacobi_cinv, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    apply_M_rbjacobi_cinv(x, y);
  }

  ////////////////////////////////////////
  // RIGHT BLOCK JACOBI SCHUR FUNCTIONS //
  ////////////////////////////////////////

  // There are no versions of this that perform left, right, clover only, etc.
  void apply_M_rbjacobi_schur(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_schur, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    // Apply (1 - D_{eo} D^{-1}_{oo} D_{oe} D^{-1}_{ee}))
    
    // Allocate if needed, zero the temporary vector. 
    if (eo_cvector == 0) { eo_cvector = allocate_vector<complex<double>>(lat->get_size_cv()); }
    zero_vector(eo_cvector, lat->get_size_cv()); 

    // Apply D_{oe} D^{-1}_{ee}
    apply_M_rbjacobi_oe(eo_cvector, rhs);

    // Apply D_{eo} D^{-1}_{oo}. 
    apply_M_rbjacobi_eo(eo_cvector, eo_cvector);

    // Form the final solution.
    caxpbyz(1.0, rhs, -1.0, eo_cvector, lhs, lat->get_size_cv()/2);
  }

  // Prepare for a right block jacobi schur solve.
  // Forms b_r = b_e - D_{eo} D^{-1}_{oo} b_{o}
  void prepare_M_rbjacobi_schur(complex<double>* b_r, complex<double>* b)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call prepare_M_rbjacobi_schur, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    // Apply D_{eo} D^{-1}_{oo}
    apply_M_rbjacobi_eo(b_r, b);

    // Put in b_e contribution.
    cxpay(b, -1.0, b_r, lat->get_size_cv()/2);

    // Zero b_o contribution
    zero_vector(b_r+lat->get_size_cv()/2, lat->get_size_cv()/2);
  }

  // Reconstruct a right block jacobi schur solve.
  // x_e = D_{ee}^{-1} y_e, x_o = D^{-1}_{oo}(b_o - D_{oe} D^{-1}_{ee} y_e)
  void reconstruct_M_rbjacobi_schur(complex<double>* x, complex<double>* y_e, complex<double>* b)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call reconstruct_M_rbjacobi_schur, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    // Allocate if needed, zero the temporary vector. 
    if (eo_cvector == 0) { eo_cvector = allocate_vector<complex<double>>(lat->get_size_cv()); }
    zero_vector(eo_cvector, lat->get_size_cv()); 

    // We'll form x_o first, since x_e is easy.

    // Apply D_{oe} D^{-1}_{ee} y_e
    apply_M_rbjacobi_oe(eo_cvector, y_e);

    // Form b_o - D_{oe} D^{-1}_ee y_e
    cxpay(b+lat->get_size_cv()/2, -1.0, eo_cvector+lat->get_size_cv()/2, lat->get_size_cv()/2);

    // Copy y_e into eo_cvector so we can do the cinv in one pass.
    copy_vector(eo_cvector, y_e, lat->get_size_cv()/2);

    // Apply D_{ee}^{-1}, D_{oo}^{-1}
    apply_M_rbjacobi_cinv(x, eo_cvector);
  }

  // Reconstruct a right block jacobi schur solve to a rbjacobi solve.
  // x_e = y_e, x_o = (b_o - D_{oe} D^{-1}_{ee} y_e)
  void reconstruct_M_rbjacobi_schur_to_rbjacobi(complex<double>* x, complex<double>* y_e, complex<double>* b)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call reconstruct_M_rbjacobi_schur_to_rbjacobi, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    // Allocate if needed, zero the temporary vector. 
    if (eo_cvector == 0) { eo_cvector = allocate_vector<complex<double>>(lat->get_size_cv()); }
    zero_vector(eo_cvector, lat->get_size_cv()); 

    // We'll form x_o first, since x_e is easy.

    // Apply D_{oe} D^{-1}_{ee} y_e
    apply_M_rbjacobi_oe(eo_cvector, y_e);

    // Form b_o - D_{oe} D^{-1}_ee y_e
    caxpbyz(1.0, b+lat->get_size_cv()/2, -1.0, eo_cvector+lat->get_size_cv()/2, x+lat->get_size_cv()/2, lat->get_size_cv()/2);

    // Copy y_e into eo_cvector so we can do the cinv in one pass.
    copy_vector(x, y_e, lat->get_size_cv()/2);
  }

  //////////////////////////////
  // DAGGER STENCIL FUNCTIONS //
  //////////////////////////////

  void build_rbj_dagger_stencil()
  {
    if (built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call build_rbj_dagger_stencil, but it's already been called once.\n";
      return;
    }

    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call build_rbj_dagger_stencil, but the right jacobi stencil has not been built yet.\n";
      return;
    }

    const int nc = lat->get_nc();
    const int nc2 = nc*nc;
    const int vol = lat->get_volume();
    const int size_cm = lat->get_size_cm();

    // As with anything else, this only supports one-link stencils for now.
    if (rbjacobi_clover != 0)
    {
      rbj_dagger_clover = allocate_vector<complex<double>>(lat->get_size_cm());
      cMATcopy_conjtrans_square(rbjacobi_clover, rbj_dagger_clover, vol, nc);
    }

    if (rbjacobi_cinv != 0)
    {
      rbj_dagger_cinv = allocate_vector<complex<double>>(lat->get_size_cm());
      cMATcopy_conjtrans_square(rbjacobi_cinv, rbj_dagger_cinv, vol, nc);
    }
      
    if (rbjacobi_hopping != 0)
    {
      rbj_dagger_hopping = allocate_vector<complex<double>>(lat->get_size_hopping());

      // +x
      // The right link is the dagger of the left link of the site from the right.
      cshift(priv_cmatrix, rbjacobi_hopping + 2*size_cm, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, nc2, lat);
      cMATcopy_conjtrans_square(priv_cmatrix, rbj_dagger_hopping, vol, nc);

      // +y
      // The up link is the dagger of the down link of the site from above.
      cshift(priv_cmatrix, rbjacobi_hopping + 3*size_cm, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, nc2, lat);
      cMATcopy_conjtrans_square(priv_cmatrix, rbj_dagger_hopping + size_cm, vol, nc);

      // -x
      // The left link is the dagger of the right link of the site from the left.
      cshift(priv_cmatrix, rbjacobi_hopping, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, nc2, lat);
      cMATcopy_conjtrans_square(priv_cmatrix, rbj_dagger_hopping + 2*size_cm, vol, nc);

      // -y
      // The down link is the dagger of the up link of the site below. 
      cshift(priv_cmatrix, rbjacobi_hopping + size_cm, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, nc2, lat);
      cMATcopy_conjtrans_square(priv_cmatrix, rbj_dagger_hopping + 3*size_cm, vol, nc);
    }
    
    if (twolink != 0)
    {
      //dagger_twolink = allocate_vector<complex<double>>(lat->get_size_hopping());
      cout << "[QMG-WARNING]: two link stencil not yet supported.\n";
    }
      
    if (corner != 0)
    {
      //dagger_corner = allocate_vector<complex<double>>(lat->get_size_corner());
      cout << "[QMG-WARNING]: corner stencil not yet supported.\n";
    }

    built_rbj_dagger = true; 

  }

  // Perform swaps to put dagger stencils in/out of place.
  bool perform_swap_rbj_dagger()
  {
    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call perform_swap_rbj_dagger, but the right jacobi dagger stencil has not been allocated.\n";
      return false;
    }

    if (!swap_rbj_dagger)
    {
      // Swap dagger stencils in.
      // Pointer swaaaap.
      std::swap(clover, rbj_dagger_clover);
      std::swap(hopping, rbj_dagger_hopping);
      std::swap(twolink, rbj_dagger_twolink);
      std::swap(corner, rbj_dagger_corner);
      shift = 0.0;
      eo_shift = 0.0;
      dof_shift = 0.0;
      swap_rbj_dagger = true;
    }
    else
    {
      // Swap back.
      std::swap(clover, rbj_dagger_clover);
      std::swap(hopping, rbj_dagger_hopping);
      std::swap(twolink, rbj_dagger_twolink);
      std::swap(corner, rbj_dagger_corner);
      shift = shift_backup;
      eo_shift = eo_shift_backup;
      dof_shift = dof_shift_backup;
      swap_rbj_dagger = false;
    }

    return swap_rbj_dagger;
  }

  // Gettin' lazy over here!

  void print_stencil_rbj_dagger_site(int x, int y, string prefix = "")
  {
    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call print_stencil_rbj_dagger_site, but the right jacobi dagger stencil has not been allocated.\n";
      return;
    }

    perform_swap_rbj_dagger();
    print_stencil_site(x, y, prefix);
    perform_swap_rbj_dagger();
  }

  
  void apply_M_rbj_dagger_clover(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_clover, but the right jacobi dagger stencil has not been allocated.\n";
      return;
    }

    if (rbj_dagger_clover == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_clover, but the right jacobi dagger clover does not exist.\n";
      return;
    }

    perform_swap_rbj_dagger();
    apply_M_clover(lhs, rhs);
    perform_swap_rbj_dagger();
  }

  void apply_M_rbj_dagger_eo(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_eo, but the right jacobi dagger stencil has not been allocated.\n";
      return;
    }

    if (rbj_dagger_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_eo, but the right jacobi dagger hopping term does not exist.\n";
      return;
    }

    perform_swap_rbj_dagger();
    apply_M_eo(lhs, rhs);
    perform_swap_rbj_dagger();
  }

  void apply_M_rbj_dagger_eo(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_eo, but the right jacobi dagger stencil has not been allocated.\n";
      return;
    }

    if (rbj_dagger_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_eo, but the right jacobi dagger hopping term does not exist.\n";
      return;
    }

    perform_swap_rbj_dagger();
    apply_M_eo(lhs, rhs, dir);
    perform_swap_rbj_dagger();
  }
  
  void apply_M_rbj_dagger_oe(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_oe, but the right jacobi dagger stencil has not been allocated.\n";
      return;
    }

    if (rbj_dagger_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_oe, but the right jacobi dagger hopping term does not exist.\n";
      return;
    }

    perform_swap_rbj_dagger();
    apply_M_oe(lhs, rhs);
    perform_swap_rbj_dagger();
  }

  void apply_M_rbj_dagger_oe(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_oe, but the right jacobi dagger stencil has not been allocated.\n";
      return;
    }

    if (rbj_dagger_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_oe, but the right jacobi dagger hopping term does not exist.\n";
      return;
    }

    perform_swap_rbj_dagger();
    apply_M_oe(lhs, rhs, dir);
    perform_swap_rbj_dagger();
  }

  void apply_M_rbj_dagger_hopping(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_hopping, but the right jacobi dagger stencil has not been allocated.\n";
      return;
    }

    if (rbj_dagger_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_hopping, but the right jacobi dagger hopping term does not exist.\n";
      return;
    }

    perform_swap_rbj_dagger();
    apply_M_hopping(lhs, rhs);
    perform_swap_rbj_dagger();
  }

  void apply_M_rbj_dagger_hopping(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_hopping, but the right jacobi dagger stencil has not been allocated.\n";
      return;
    }

    if (rbj_dagger_hopping == 0)
    {
      cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_hopping, but the right jacobi dagger hopping term does not exist.\n";
      return;
    }

    perform_swap_rbj_dagger();
    apply_M_hopping(lhs, rhs);
    perform_swap_rbj_dagger();
  }
  
  // void apply_M_rbj_dagger_twolink(complex<double>* lhs, complex<double>* rhs);
  // void apply_M_rbj_dagger_corner(complex<double>* lhs, complex<double>* rhs);

  void apply_M_rbj_dagger_shift(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger_shift, but the right jacobi dagger stencil has not been allocated.\n";
      return;
    }

    perform_swap_rbj_dagger();
    apply_M_shift(lhs, rhs);
    perform_swap_rbj_dagger();
  }

  void apply_M_rbj_dagger(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbj_dagger, but the right jacobi dagger stencil has not been allocated.\n";
      return;
    }

    perform_swap_rbj_dagger();
    apply_M(lhs, rhs);
    perform_swap_rbj_dagger();
  }

  ////////////////////////////////////////////
  // RIGHT JACOBI NORMAL EQUATION FUNCTIONS //
  ////////////////////////////////////////////

  void apply_M_rbjacobi_MDM(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_MDM, but the right jacobi stencil has not been built.\n";
      return;
    }

    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_MDM, but the right jacobi dagger stencil has not been built.\n";
      return;
    }

    zero_vector(extra_cvector, lat->get_size_cv());
    apply_M_rbjacobi(extra_cvector, rhs);
    apply_M_rbj_dagger(lhs, extra_cvector);
  }

  void prepare_M_rbjacobi_MDM(complex<double>* Mdagger_b, complex<double>* b)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call prepare_M_rbjacobi_MDM, but the right jacobi stencil has not been built.\n";
      return;
    }

    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call prepare_M_rbjacobi_MDM, but the right jacobi dagger stencil has not been built.\n";
      return;
    }

    apply_M_rbj_dagger(Mdagger_b, b);
  }

  // Reconstruct right block jacobi solve, which is just applying the above function.
  void reconstruct_M_rbjacobi_MDM(complex<double>* x, complex<double>* y)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call reconstruct_M_rbjacobi_MDM, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call reconstruct_M_rbjacobi_MDM, but the right jacobi dagger stencil has not been built.\n";
      return;
    }

    apply_M_rbjacobi_cinv(x, y);
  }

  // Reconstruct right block jacobi solve, which is just applying the above function.
  void reconstruct_M_rbjacobi_MDM_to_rbjacobi(complex<double>* x, complex<double>* y)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call reconstruct_M_rbjacobi_MDM, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call reconstruct_M_rbjacobi_MDM, but the right jacobi dagger stencil has not been built.\n";
      return;
    }

    copy_vector(y, x, lat->get_size_cv());
  }

  void apply_M_rbjacobi_MMD(complex<double>* lhs, complex<double>* rhs)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_MMD, but the right jacobi stencil has not been built.\n";
      return;
    }

    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call apply_M_rbjacobi_MMD, but the right jacobi dagger stencil has not been built.\n";
      return;
    }

    zero_vector(extra_cvector, lat->get_size_cv());
    apply_M_rbj_dagger(extra_cvector, rhs);
    apply_M_rbjacobi(lhs, extra_cvector);
  }

  void reconstruct_M_rbjacobi_MMD(complex<double>* x, complex<double>* y)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call reconstruct_M_rbjacobi_MMD, but the right jacobi stencil has not been built.\n";
      return;
    }

    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call reconstruct_M_rbjacobi_MMD, but the right jacobi dagger stencil has not been built.\n";
      return;
    }

    zero_vector(x, lat->get_size_cv());
    apply_M_rbj_dagger(x, y);
    zero_vector(extra_cvector, lat->get_size_cv());
    apply_M_rbjacobi_cinv(extra_cvector, x);
    copy_vector(x, extra_cvector, lat->get_size_cv());
  }

  // Reconstruct right block jacobi solve, which is just applying the above function.
  void reconstruct_M_rbjacobi_MMD_to_rbjacobi(complex<double>* x, complex<double>* y)
  {
    if (!built_rbjacobi)
    {
      std::cout << "[QMG-WARNING]: Tried to call reconstruct_M_rbjacobi_MDM, but the rbjacobi stencil has not been allocated.\n";
      return;
    }

    if (!built_rbj_dagger)
    {
      std::cout << "[QMG-WARNING]: Tried to call reconstruct_M_rbjacobi_MDM, but the right jacobi dagger stencil has not been built.\n";
      return;
    }

    zero_vector(x, lat->get_size_cv());
    apply_M_rbj_dagger(x, y);
  }


  //////////////////////////
  // CONVENIENT FUNCTIONS //
  //////////////////////////

  void apply_M(complex<double>* lhs, complex<double>* rhs, QMGStencilType stencil)
  {
    switch (stencil)
    {
      case QMG_MATVEC_ORIGINAL:
        apply_M(lhs, rhs);
        break;
      case QMG_MATVEC_DAGGER:
        apply_M_dagger(lhs, rhs);
        break;
      case QMG_MATVEC_RIGHT_JACOBI:
        apply_M_rbjacobi(lhs, rhs);
        break;
      case QMG_MATVEC_RIGHT_SCHUR:
        apply_M_rbjacobi_schur(lhs, rhs);
        break;
      case QMG_MATVEC_M_MDAGGER:
        apply_M_M_dagger(lhs, rhs);
        break;
      case QMG_MATVEC_MDAGGER_M:
        apply_M_dagger_M(lhs, rhs);
        break;
      case QMG_MATVEC_RBJ_DAGGER:
        apply_M_rbj_dagger(lhs, rhs);
        break;
      case QMG_MATVEC_RBJ_M_MDAGGER:
        apply_M_rbjacobi_MMD(lhs, rhs);
        break;
      case QMG_MATVEC_RBJ_MDAGGER_M:
        apply_M_rbjacobi_MDM(lhs, rhs);
        break;
      default:
        cout << "[QMG-ERROR]: Tried to call apply_M with invalid stencil type.\n";
        break;
    }
  }

  void prepare_M(complex<double>* b_prep, complex<double>* b, QMGStencilType stencil)
  {
    switch (stencil)
    {
      case QMG_MATVEC_ORIGINAL:
        copy_vector(b_prep, b, lat->get_size_cv());
        break;
      case QMG_MATVEC_DAGGER:
        copy_vector(b_prep, b, lat->get_size_cv());
        break;
      case QMG_MATVEC_RIGHT_JACOBI:
        copy_vector(b_prep, b, lat->get_size_cv());
        break;
      case QMG_MATVEC_RIGHT_SCHUR:
        prepare_M_rbjacobi_schur(b_prep, b);
        break;
      case QMG_MATVEC_M_MDAGGER:
        copy_vector(b_prep, b, lat->get_size_cv());
        break;
      case QMG_MATVEC_MDAGGER_M:
        prepare_M_dagger_M(b_prep, b);
        break;
      case QMG_MATVEC_RBJ_DAGGER:
        copy_vector(b_prep, b, lat->get_size_cv());
        break;
      case QMG_MATVEC_RBJ_M_MDAGGER:
        copy_vector(b_prep, b, lat->get_size_cv());
        break;
      case QMG_MATVEC_RBJ_MDAGGER_M:
        prepare_M_rbjacobi_MDM(b_prep, b);
        break;
      default:
        cout << "[QMG-ERROR]: Tried to call prepare_M with invalid stencil type.\n";
        break;
    }
  }

  void reconstruct_M(complex<double>* x, complex<double>* y, complex<double>* b, QMGStencilType stencil)
  {
    switch (stencil)
    {
      case QMG_MATVEC_ORIGINAL:
        copy_vector(x, y, lat->get_size_cv());
        break;
      case QMG_MATVEC_DAGGER:
        copy_vector(x, y, lat->get_size_cv());
        break;
      case QMG_MATVEC_RIGHT_JACOBI:
        reconstruct_M_rbjacobi(x, y);
        break;
      case QMG_MATVEC_RIGHT_SCHUR:
        reconstruct_M_rbjacobi_schur(x, y, b);
        break;
      case QMG_MATVEC_M_MDAGGER:
        reconstruct_M_M_dagger(x, y);
        break;
      case QMG_MATVEC_MDAGGER_M:
        copy_vector(x, y, lat->get_size_cv());
        break;
      case QMG_MATVEC_RBJ_DAGGER:
        copy_vector(x, y, lat->get_size_cv());
        break;
      case QMG_MATVEC_RBJ_M_MDAGGER:
        reconstruct_M_rbjacobi_MMD(x, y);
        break;
      case QMG_MATVEC_RBJ_MDAGGER_M:
        reconstruct_M_rbjacobi_MDM(x, y);
        break;
      default:
        cout << "[QMG-ERROR]: Tried to call reconstruct_M with invalid stencil type.\n";
        break;
    }
  }

  // Function which maps between QMGStencilType and a function pointer.
  static matrix_op_cplx get_apply_function(QMGStencilType stencil)
  {
    switch (stencil)
    {
      case QMG_MATVEC_ORIGINAL:
        return apply_stencil_2D_M;
        break;
      case QMG_MATVEC_DAGGER:
        return apply_stencil_2D_M_dagger;
        break;
      case QMG_MATVEC_RIGHT_JACOBI:
        return apply_stencil_2D_M_rbjacobi;
        break;
      case QMG_MATVEC_RIGHT_SCHUR:
        return apply_stencil_2D_M_rbjacobi_schur;
        break;
      case QMG_MATVEC_M_MDAGGER:
        return apply_stencil_2D_M_M_dagger;
        break;
      case QMG_MATVEC_MDAGGER_M:
        return apply_stencil_2D_M_dagger_M;
        break;
      case QMG_MATVEC_RBJ_DAGGER:
        return apply_stencil_2D_M_rbj_dagger;
        break;
      case QMG_MATVEC_RBJ_M_MDAGGER:
        return apply_stencil_2D_M_rbjacobi_MMD;
        break;
      case QMG_MATVEC_RBJ_MDAGGER_M:
        return apply_stencil_2D_M_rbjacobi_MDM;
        break;
      default:
        cout << "[QMG-ERROR]: Tried to call get_apply_function with invalid stencil type.\n";
        return 0;
        break;
    }
  }

};

// Special C function wrappers for stencil applications.
void apply_stencil_2D_M(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  zero_vector(lhs, stenc->lat->get_size_cv());
  stenc->apply_M(lhs, rhs); // lhs = M rhs
}

// Apply clover only.
void apply_stencil_2D_M_piece_clover(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  zero_vector(lhs, stenc->lat->get_size_cv());
  stenc->apply_M_clover(lhs, rhs); // lhs = M rhs
}

// Apply hopping only.
void apply_stencil_2D_M_piece_hopping(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  zero_vector(lhs, stenc->lat->get_size_cv());
  stenc->apply_M_hopping(lhs, rhs); // lhs = M rhs
}

void apply_stencil_2D_M_dagger(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  zero_vector(lhs, stenc->lat->get_size_cv());
  if (!stenc->built_dagger)
  {
    std::cout << "[QMG-WARNING]: Tried to call apply_stencil_2D_M_dagger, but the dagger stencil has not been built.\n";
    return;
  }
  stenc->apply_M_dagger(lhs, rhs); // lhs = M rhs
}


void apply_stencil_2D_M_dagger_M(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  if (!stenc->built_dagger)
  {
    std::cout << "[QMG-WARNING]: Tried to call apply_stencil_2D_M_dagger_M, but the dagger stencil has not been built.\n";
    return;
  }
  zero_vector(stenc->expose_internal_cvector(), stenc->lat->get_size_cv());
  stenc->apply_M(stenc->expose_internal_cvector(), rhs); // stenc->expose_internal_cvector() = M rhs
  zero_vector(lhs, stenc->lat->get_size_cv());
  stenc->apply_M_dagger(lhs, stenc->expose_internal_cvector()); // lhs = M^\dagger stenc->expose_internal_cvector()
}

void apply_stencil_2D_M_M_dagger(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  if (!stenc->built_dagger)
  {
    std::cout << "[QMG-WARNING]: Tried to call apply_stencil_2D_M_M_dagger, but the dagger stencil has not been built.\n";
    return;
  }
  zero_vector(stenc->expose_internal_cvector(), stenc->lat->get_size_cv());
  stenc->apply_M_dagger(stenc->expose_internal_cvector(), rhs); // lhs = M rhs
  zero_vector(lhs, stenc->lat->get_size_cv());
  stenc->apply_M(lhs, stenc->expose_internal_cvector()); // lhs = M rhs
}

void apply_stencil_2D_M_rbjacobi(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  if (!stenc->built_rbjacobi)
  {
    std::cout << "[QMG-WARNING]: Tried to call apply_stencil_2D_M_rbjacobi, but the rbjacobi stencil has not been built.\n";
    return;
  }
  zero_vector(lhs, stenc->lat->get_size_cv());
  stenc->apply_M_rbjacobi(lhs, rhs); // lhs = M rhs
}

// function for rbjacobi reconstruct.
void apply_stencil_2D_M_rbjacobi_cinv(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  if (!stenc->built_rbjacobi)
  {
    std::cout << "[QMG-WARNING]: Tried to call apply_stencil_2D_M_rbjacobi_cinv, but the rbjacobi stencil has not been built.\n";
    return;
  }
  zero_vector(lhs, stenc->lat->get_size_cv());
  stenc->apply_M_rbjacobi_cinv(lhs, rhs); // lhs = M rhs
}

void apply_stencil_2D_M_rbjacobi_schur(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  if (!stenc->built_rbjacobi)
  {
    std::cout << "[QMG-WARNING]: Tried to call apply_stencil_2D_M_rbjacobi_schur, but the rbjacobi stencil has not been built.\n";
    return;
  }
  zero_vector(lhs, stenc->lat->get_size_cv()/2);
  stenc->apply_M_rbjacobi_schur(lhs, rhs); // lhs = M rhs
}

void apply_stencil_2D_M_rbj_dagger(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  if (!stenc->built_rbj_dagger)
  {
    std::cout << "[QMG-WARNING]: Tried to call apply_stencil_2D_M_rbj_dagger, but the rbjacobi dagger stencil has not been built.\n";
    return;
  }
  zero_vector(lhs, stenc->lat->get_size_cv());
  stenc->apply_M_rbj_dagger(lhs, rhs); // lhs = M rhs
}

void apply_stencil_2D_M_rbjacobi_MMD(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  if (!stenc->built_rbjacobi)
  {
    std::cout << "[QMG-WARNING]: Tried to call apply_stencil_2D_M_rbjacobi_MMD, but the rbjacobi stencil has not been built.\n";
    return;
  }
  if (!stenc->built_rbj_dagger)
  {
    std::cout << "[QMG-WARNING]: Tried to call apply_stencil_2D_M_rbjacobi_MMD, but the rbjacobi dagger stencil has not been built.\n";
    return;
  }
  zero_vector(lhs, stenc->lat->get_size_cv());
  stenc->apply_M_rbjacobi_MMD(lhs, rhs); // lhs = M rhs
}

void apply_stencil_2D_M_rbjacobi_MDM(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  if (!stenc->built_rbjacobi)
  {
    std::cout << "[QMG-WARNING]: Tried to call apply_stencil_2D_M_rbjacobi_MDM, but the rbjacobi stencil has not been built.\n";
    return;
  }
  if (!stenc->built_rbj_dagger)
  {
    std::cout << "[QMG-WARNING]: Tried to call apply_stencil_2D_M_rbjacobi_MDM, but the rbjacobi dagger stencil has not been built.\n";
    return;
  }
  zero_vector(lhs, stenc->lat->get_size_cv());
  stenc->apply_M_rbjacobi_MDM(lhs, rhs); // lhs = M rhs
}

#endif // QMG_STENCIL_2D