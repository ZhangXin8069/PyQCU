// Copyright (c) 2017 Evan S Weinberg
// Create a coarse operator from another stencil and a transfer object. 
// If chiral, top half of dof should be top chirality,
// bottom half should be bottom chirality.

#ifndef QMG_COARSE
#define QMG_COARSE

#include <iostream>
#include <complex>
using std::complex;

#include "blas/generic_vector.h"
#include "lattice/lattice.h"
#include "stencil/stencil_2d.h"
#include "transfer/transfer.h"

// Extended gamma_5 types for coarse operators.
enum QMGSigmaTypeCoarse
{
  QMG_SIGMA_1_L = 6, // Coarsened original op. Transfer doubling is via applying an op. Left apply U^{-\dagger} \sigma_1 L.
  QMG_SIGMA_1_R = 7, // Coarsened original op. Transfer doubling is via applying an op. Right apply \sigma_1 L^{-\dagger}.
  QMG_SIGMA_1_L_RBJ = 8, // Coarsened rbj op. Transfer doubling is via applying an op. Left apply \sigma_1^L B^{-1}.
  QMG_SIGMA_1_R_RBJ = 9, // Coarsened rbj op. Transfer doubling is via applying an op. Right apply B \sigma_1^R.
};

// Build a coarse operator from another stencil and a
// a transfer object. 
struct CoarseOperator2D : public Stencil2D
{
protected:
  // Get rid of copy, assignment.
  CoarseOperator2D(CoarseOperator2D const &);
  CoarseOperator2D& operator=(CoarseOperator2D const &);

  // Temporary storage.
  complex<double>* tmp_coarse;
  complex<double>* tmp_fine;
  complex<double>* tmp_Afine;

  // Save the fine lattice.
  Lattice2D* fine_lat; 

  // Is it a chiral stencil?
  bool is_chiral; 

  // Is this the coarse version of a right block jacobi stencil?
  bool use_rbjacobi;

  // Save the transfer object, if it exists.
  TransferMG* in_transfer;

  // Save the default chirality.
  QMGDefaultChirality default_chirality;

  // Places to store sigma_1^{L/R} maybe prime, if we need it.
  complex<double>* sigma_1_L;
  complex<double>* sigma_1_R;

public:
  // Enum for if we should build the dagger and/or rbjacobi stencil.
  enum QMGCoarseBuildStencil
  {
    QMG_COARSE_BUILD_ORIGINAL = 0, // build coarse stencil only.
    QMG_COARSE_BUILD_DAGGER = 1, // also build dagger stencil
    QMG_COARSE_BUILD_RBJACOBI = 2, // also build rbjacobi stencil
    QMG_COARSE_BUILD_DAGGER_RBJACOBI = 3, // build both dagger, rbjacobi stencil.
    QMG_COARSE_BUILD_RBJDAGGER = 4, // build rbjacobi and rbjacobi dagger stencil.
    QMG_COARSE_BUILD_ALL = 5, // build all types of stencils
  };

public:

  // Base constructor to set up a bare stencil.
  CoarseOperator2D(Lattice2D* in_lat, int pieces, bool is_chiral, QMGDefaultChirality def_chiral = QMG_CHIRALITY_NONE, complex<double> in_shift = 0.0, complex<double> in_eo_shift = 0.0, complex<double> in_dof_shift = 0.0)
    : Stencil2D(in_lat, pieces, in_shift, in_eo_shift, in_dof_shift), is_chiral(is_chiral), use_rbjacobi(false), in_transfer(0), default_chirality(def_chiral), sigma_1_L(0), sigma_1_R(0)
  {
    tmp_coarse = nullptr;
    tmp_fine = nullptr;
    tmp_Afine = nullptr;
  }

  // Base constructor to build a coarse stencil from a fine stencil.
  // NOTE! Need a way to determine QMG_PIECE_... based on the
  // input stencil. Maybe there needs to be a function in each stencil
  // to determine the size, then some smart function that figures
  // out what the stencil will look like after coarsening?
  // Also need some smart way to deal with the mass (for \gamma_5 ops)
  // Currently this function only transfers identity shifts.
  CoarseOperator2D(Lattice2D* in_lat, Stencil2D* fine_stencil, Lattice2D* fine_lattice, TransferMG* transfer, bool is_chiral = false, bool use_rbjacobi = false, QMGCoarseBuildStencil build_extra = QMG_COARSE_BUILD_ORIGINAL)
    : Stencil2D(in_lat, QMG_PIECE_CLOVER_HOPPING, 0.0, 0.0, 0.0), fine_lat(fine_lattice), is_chiral(is_chiral), use_rbjacobi(use_rbjacobi), in_transfer(transfer), sigma_1_L(0), sigma_1_R(0)
  {
    const int coarse_vol = lat->get_volume();
    const int coarse_size = lat->get_size_cv();
    const int coarse_nc = lat->get_nc();
    const int fine_size = fine_lat->get_size_cv();

    // Allocate temporary space.
    tmp_coarse = allocate_vector<complex<double>>(coarse_size);
    tmp_fine = allocate_vector<complex<double>>(fine_size);
    tmp_Afine = allocate_vector<complex<double>>(fine_size);

    // Learn about chirality from transfer object.
    QMGDoublingType doubling = in_transfer->get_doubling();

    switch(doubling)
    {
      case QMG_DOUBLE_NONE:
        default_chirality = QMG_CHIRALITY_NONE;
        break;
      case QMG_DOUBLE_PROJECTION:
        default_chirality = QMG_CHIRALITY_GAMMA_5;
        break;
      case QMG_DOUBLE_OPERATOR:
        default_chirality = QMG_CHIRALITY_SIGMA_1;
        break;
    }

    // Prepare for rbjacobi build. 
    if (use_rbjacobi)
    {
      fine_stencil->perform_swap_rbjacobi();
    }

    ///////////////////////////////////
    // Step 0: Transfer shifts over. //
    ///////////////////////////////////

    // We need some set of flags concerning transfering
    // eo and dof shifts over...
    shift = fine_stencil->get_shift();

    ///////////////////////////////////////////////////////////////////////////
    // Step 1: learn about (some of) the coarse clover from the fine clover. //
    ///////////////////////////////////////////////////////////////////////////

    zero_vector(clover, lat->get_size_cm());
    
    // Take matrix elements of the fine clover term.
    for (int color = 0; color < coarse_nc; color++)
    {
      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      if (coarse_vol == 1)
      {
        tmp_coarse[lat->cv_coord_to_index(0, color)] = 1.0;
        transfer->prolong_c2f(tmp_coarse, tmp_fine);
        fine_stencil->apply_M_clover(tmp_Afine, tmp_fine);
        zero_vector(tmp_coarse, coarse_size);
        transfer->restrict_f2c(tmp_Afine, tmp_coarse);
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(0, c, color)] += tmp_coarse[lat->cv_coord_to_index(0,c)];
        continue; 
      }

      // Set a 1 at each coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol; i++)
        tmp_coarse[lat->cv_coord_to_index(i, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_clover(tmp_Afine, tmp_fine);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // Fill matrix elements into clover.
      for (int i = 0; i < coarse_vol; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];
    }

    //////////////////////////////////////////////////////////////////
    // Step 2: learn about some of the coarse clover (all if it's a //
    //         distance one fine stencil), and some of the hopping  //
    //         term (all if it's a distance one fine stencil)       //
    //////////////////////////////////////////////////////////////////

    // Take matrix elements of the fine hopping term. This gets tweaked
    // if the coarse lattice has dimension 1 or 2 in some direction.

    zero_vector(hopping, lat->get_size_hopping());

    for (int color = 0; color < coarse_nc; color++)
    {
      

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // If the coarse volume is 1, there's no meaningful
      // hopping term.
      if (coarse_vol == 1)
      {
        tmp_coarse[lat->cv_coord_to_index(0, color)] = 1.0;
        transfer->prolong_c2f(tmp_coarse, tmp_fine);
        fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine);
        zero_vector(tmp_coarse, coarse_size);
        transfer->restrict_f2c(tmp_Afine, tmp_coarse);
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(0, c, color)] += tmp_coarse[lat->cv_coord_to_index(0,c)];
        continue; 
      }

      /////////////////////////////////////////////////
      // Learn from the +x part of the fine stencil. //
      /////////////////////////////////////////////////

      // Set a 1 at each even coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_XP1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // even coarse sites update coarse clover, odd coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

      if (lat->get_dim_mu(0) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i+coarse_vol/2, c, color, QMG_DIR_INDEX_XP1)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each odd coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_XP1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // odd coarse sites update coarse clover, even coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      if (lat->get_dim_mu(0) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i, c, color, QMG_DIR_INDEX_XP1)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

      /////////////////////////////////////////////////
      // Learn from the +y part of the fine stencil. //
      /////////////////////////////////////////////////

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each even coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_YP1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // even coarse sites update coarse clover, odd coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

      if (lat->get_dim_mu(1) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i+coarse_vol/2, c, color, QMG_DIR_INDEX_YP1)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each odd coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_YP1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // odd coarse sites update coarse clover, even coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      if (lat->get_dim_mu(1) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i, c, color, QMG_DIR_INDEX_YP1)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

      /////////////////////////////////////////////////
      // Learn from the -x part of the fine stencil. //
      /////////////////////////////////////////////////

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each even coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_XM1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // even coarse sites update coarse clover, odd coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

      if (lat->get_dim_mu(0) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i+coarse_vol/2, c, color, QMG_DIR_INDEX_XM1)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each odd coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_XM1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // odd coarse sites update coarse clover, even coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      if (lat->get_dim_mu(0) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i, c, color, QMG_DIR_INDEX_XM1)] += tmp_coarse[lat->cv_coord_to_index(i, c)];


      /////////////////////////////////////////////////
      // Learn from the -y part of the fine stencil. //
      /////////////////////////////////////////////////

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each even coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_YM1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // even coarse sites update coarse clover, odd coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

      if (lat->get_dim_mu(1) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i+coarse_vol/2, c, color, QMG_DIR_INDEX_YM1)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each odd coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_YM1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // odd coarse sites update coarse clover, even coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      if (lat->get_dim_mu(1) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i, c, color, QMG_DIR_INDEX_YM1)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

    }

    // Undo rbjacobi build. 
    if (use_rbjacobi)
    {
      fine_stencil->perform_swap_rbjacobi();
    }

    // Build dagger, rbjacobi stencils

    if (build_extra == QMG_COARSE_BUILD_DAGGER || build_extra == QMG_COARSE_BUILD_DAGGER_RBJACOBI || build_extra == QMG_COARSE_BUILD_ALL)
    {
      build_dagger_stencil();
    }

    if (build_extra == QMG_COARSE_BUILD_RBJACOBI || build_extra == QMG_COARSE_BUILD_DAGGER_RBJACOBI || build_extra == QMG_COARSE_BUILD_RBJDAGGER || build_extra == QMG_COARSE_BUILD_ALL)
    {
      build_rbjacobi_stencil();
    }

    if (build_extra == QMG_COARSE_BUILD_RBJDAGGER || build_extra == QMG_COARSE_BUILD_ALL)
    {
      build_rbj_dagger_stencil();
    }

    // Still need to coarsen in 2-link, corner terms...
    // you know, when I ever actually implement them.
  }

  ~CoarseOperator2D()
  {
    if (tmp_coarse != 0) deallocate_vector(&tmp_coarse);
    if (tmp_fine != 0) deallocate_vector(&tmp_fine);
    if (tmp_Afine != 0) deallocate_vector(&tmp_Afine);

    if (sigma_1_L != 0)
      deallocate_vector(&sigma_1_L);

    if (sigma_1_R != 0)
      deallocate_vector(&sigma_1_R);
  }

  // The coarse operator could have any number of dof per sites.
  static int get_dof(int i = 0)
  {
    return -1;
  }

  // The coarse operator might have a sense of chirality.
  static chirality_state has_chirality()
  {
    return QMG_CHIRAL_UNKNOWN; 
  }

  virtual void gamma5(complex<double>* vec)
  {
    if (is_chiral)
    {
      const int nc = lat->get_nc();
      {
        for (int c = nc/2; c < nc; c++)
          cax_blas(-1.0, vec+c, nc, lat->get_size_cv()/nc);
      }
    }
  }

  virtual void gamma5(complex<double>* g5_vec, complex<double>* vec)
  {
    if (is_chiral)
    {
      const int nc = lat->get_nc();
      {
        for (int c = 0; c < nc/2; c++)
          caxy_blas(1.0, vec+c, nc, g5_vec+c, nc, lat->get_size_cv()/nc);

        for (int c = nc/2; c < nc; c++)
          caxy_blas(-1.0, vec+c, nc, g5_vec+c, nc, lat->get_size_cv()/nc);
      }
    }
  }

  // Apply sigma1 in place. Default does nothing.
  virtual void sigma1(complex<double>* vec)
  {
    const int my_nc = lat->get_nc();
    if (my_nc % 2) { return ; }
    double scale[my_nc];
    int shuffle[my_nc];
    for (int i = 0; i < my_nc/2; i++)
    {
      scale[i] = 1.0;
      scale[i+my_nc/2] = 1.0;
      shuffle[i] = i+my_nc/2;
      shuffle[i+my_nc/2] = i;
    }
    caxy_shuffle_pattern(scale, shuffle, my_nc, vec, extra_cvector, lat->get_volume());
    copy_vector(vec, extra_cvector, lat->get_size_cv());
  }

  // Apply sigma1 saved in a vector
  virtual void sigma1(complex<double>* s1_vec, complex<double>* vec)
  {
    const int my_nc = lat->get_nc();
    if (my_nc % 2) { return ; }
    double scale[my_nc];
    int shuffle[my_nc];
    for (int i = 0; i < my_nc/2; i++)
    {
      scale[i] = 1.0;
      scale[i+my_nc/2] = 1.0;
      shuffle[i] = i+my_nc/2;
      shuffle[i+my_nc/2] = i;
    }
    caxy_shuffle_pattern(scale, shuffle, my_nc, vec, s1_vec, lat->get_volume());
  }

  // Chirality either does not exist or is internal dof.
  // Applies either gamma_5 or sigma_1 depending on default chirality.
  virtual void chiral_projection(complex<double>* vector, bool is_up)
  {
    if (is_chiral)
    {
      if (default_chirality == QMG_CHIRALITY_GAMMA_5)
      {
        const int nc = lat->get_nc();
        if (is_up)
        {
          for (int c = 0; c < nc/2; c++)
            zero_vector_blas(vector+nc/2+c, nc, lat->get_size_cv()/nc);
        }
        else
        {
          for (int c = 0; c < nc/2; c++)
            zero_vector_blas(vector+c, nc, lat->get_size_cv()/nc);
        }
      }
      else if (default_chirality == QMG_CHIRALITY_SIGMA_1)
      {
        // Apply sigma1 into a temporary vector.
        sigma1(extra_cvector, vector);
        caxpby(is_up ? 0.5 : -0.5, extra_cvector, 0.5, vector, lat->get_size_cv());
      }
    }
  }

  // Copy projection onto up, down.
  virtual void chiral_projection_copy(complex<double>* orig, complex<double>* dest, bool is_up)
  {
    if (is_chiral)
    {
      if (default_chirality == QMG_CHIRALITY_GAMMA_5)
      {
        const int nc = lat->get_nc();
        if (is_up)
        {
          for (int c = 0; c < nc/2; c++)
          {
            copy_vector_blas(dest+c, orig+c, nc, lat->get_size_cv()/nc);
            zero_vector_blas(dest+nc/2+c, nc, lat->get_size_cv()/nc);
          }
        }
        else
        {
          for (int c = 0; c < nc/2; c++)
          {
            copy_vector_blas(dest+nc/2+c, orig+nc/2+c, nc, lat->get_size_cv()/nc);
            zero_vector_blas(dest+c, nc, lat->get_size_cv()/nc);
          }
        }
      }
      else if (default_chirality == QMG_CHIRALITY_SIGMA_1)
      {
        // Apply sigma1 into a temporary vector.
        sigma1(extra_cvector, orig);
        caxpbyz(is_up ? 0.5 : -0.5, extra_cvector, 0.5, orig, dest, lat->get_size_cv());
      }
    }
  }

  // Copy the down projection into a new vector, perform the up in place.
  virtual void chiral_projection_both(complex<double>* orig_to_up, complex<double>* down)
  {
    if (is_chiral)
    {
      if (default_chirality == QMG_CHIRALITY_GAMMA_5)
      {
        const int nc = lat->get_nc();
        for (int c = 0; c < nc/2; c++)
        {
          copy_vector_blas(down+nc/2+c, orig_to_up+nc/2+c, nc, lat->get_size_cv()/nc);
          zero_vector_blas(down+c, nc, lat->get_size_cv()/nc);
          zero_vector_blas(orig_to_up+nc/2+c, nc, lat->get_size_cv()/nc);
        }
      }
      else if (default_chirality == QMG_CHIRALITY_SIGMA_1)
      {
        // Apply sigma1 into a temporary vector.
        sigma1(extra_cvector, orig_to_up);

        // Put down into... down.
        caxpbyz(0.5, orig_to_up, -0.5, extra_cvector, down, lat->get_size_cv());

        // Aaand get up going.
        caxpy(-1.0, down, orig_to_up, lat->get_size_cv());
      }
    }
  }

  

  virtual QMGDefaultChirality get_default_chirality()
  {
    return default_chirality;
  }


  // Extended application of a certain chiral op.
  void apply_sigma(complex<double>* output, complex<double>* input, QMGSigmaTypeCoarse type)
  {
    if (!in_transfer->has_decompositions())
    {
      std::cout << "[QMG-ERROR]: In CoarseOperator2D, cannot apply apply_sigma() if the transfer op does not have factorizations.\n";
      return;
    }

    if (in_transfer->is_symmetric()) // R = P^\dagger
    {
      // U = \Sigma, L = \Sigma^\dagger.
    
      if (sigma_1_L == 0 || sigma_1_R == 0) // We haven't built these yet.
      {
        // Allocate the storage.
        sigma_1_L = allocate_vector<complex<double>>(lat->get_size_cm());
        sigma_1_R = allocate_vector<complex<double>>(lat->get_size_cm());

        // In the cholesky case, they're actually equal, we only need
        // sigma and sigma_inv.
        complex<double>* sigma = allocate_vector<complex<double>>(lat->get_size_cm());
        complex<double>* sigma_inv = allocate_vector<complex<double>>(lat->get_size_cm());

        // Grab sigma from the transfer operator.
        in_transfer->copy_cholesky(sigma);

        // Get inverse of sigma.
        complex<double>* Qmat = allocate_vector<complex<double>>(lat->get_size_cm());
        complex<double>* Rmat = allocate_vector<complex<double>>(lat->get_size_cm());
        zero_vector(Qmat, lat->get_size_cm());
        zero_vector(Rmat, lat->get_size_cm());
        cMATx_do_qr_square(sigma, Qmat, Rmat, lat->get_volume(), lat->get_nc());
        cMATqr_do_xinv_square(Qmat, Rmat, sigma_inv, lat->get_volume(), lat->get_nc());

        // Clean up.
        deallocate_vector(&Qmat);
        deallocate_vector(&Rmat);

        // Build up a matrix of sigma_1's. 
        const int nc = lat->get_nc();
        const int nc2 = lat->get_nc()*lat->get_nc();
        double sigma1_single[nc2];
        for (int i = 0; i < nc2; i++)
        {
          sigma1_single[i] = 0.0;
        }
        for (int i = 0; i < nc; i++)
        {
          if (i < nc/2)
          {
            sigma1_single[i*nc + i + (nc/2)] = 1.0;
          }
          else
          {
            sigma1_single[i*nc + i - (nc/2)] = 1.0;
          }
        }
        complex<double>* sigma1_repeat = allocate_vector<complex<double>>(lat->get_size_cm());
        zero_vector(sigma1_repeat, lat->get_size_cm());
        capx_pattern(sigma1_single, nc2, sigma1_repeat, lat->get_volume());

        // sigma_1_L is Sigma^{-\dagger} \sigma_1 \Sigma^\dagger,
        // but when we left apply we apply the dagger, which is sigma_1_r,
        // which is Sigma \sigma_1 \Sigma^{-1}
        complex<double>* extra_cmatrix = allocate_vector<complex<double>>(lat->get_size_cm());
        cMATxtMATyMATz_square(sigma, sigma1_repeat, extra_cmatrix, lat->get_volume(), lat->get_nc());
        cMATxtMATyMATz_square(extra_cmatrix, sigma_inv, sigma_1_L, lat->get_volume(), lat->get_nc());
        deallocate_vector(&extra_cmatrix);
        copy_vector(sigma_1_R, sigma_1_L, lat->get_size_cm());

        // Aaaaand we're ready!
        deallocate_vector(&sigma1_repeat);
        deallocate_vector(&sigma);
        deallocate_vector(&sigma_inv);
      }

      switch (type)
      {
        case QMG_SIGMA_1_L:
        case QMG_SIGMA_1_R: // The same in the symmetric case.
          cMATxy(sigma_1_L, input, output, lat->get_volume(), lat->get_nc(), lat->get_nc());
          break;
        case QMG_SIGMA_1_L_RBJ:
          if (!built_rbj_dagger)
          {
            std::cout << "[QMG-ERROR]: In apply_sigma, cannot apply QMG_SIGMA_1_L_RBJ without rbjacobi dagger stencil.\n";
            copy_vector(output, input, lat->get_size_cv());
          }
          else
          {
            // Apply B^{-dagger} \sigma_1^L. (since we need to left apply \gamma_5 B^{-1})
            cMATxy(sigma_1_L, input, extra_cvector, lat->get_volume(), lat->get_nc(), lat->get_nc());
            cMATxy(rbj_dagger_cinv, extra_cvector, output, lat->get_volume(), lat->get_nc(), lat->get_nc());
          }
          break;
        case QMG_SIGMA_1_R_RBJ:
          if (!built_rbjacobi)
          {
            std::cout << "[QMG-ERROR]: In apply_sigma, cannot apply QMG_SIGMA_1_R_RBJ without rbjacobi stencil.\n";
            copy_vector(output, input, lat->get_size_cv());
          }
          else
          {
            // Apply B \sigma_1^R.
            cMATxy(sigma_1_R, input, extra_cvector, lat->get_volume(), lat->get_nc(), lat->get_nc());
            cMATxy(clover, extra_cvector, output, lat->get_volume(), lat->get_nc(), lat->get_nc());
            caxpy(shift, extra_cvector, output, lat->get_size_cv());
          }
          break;
      }
    }
    else // R != P^\dagger
    {
      if (sigma_1_L == 0 || sigma_1_R == 0) // We haven't built these yet.
      {
        // Allocate the storage.
        sigma_1_L = allocate_vector<complex<double>>(lat->get_size_cm());
        sigma_1_R = allocate_vector<complex<double>>(lat->get_size_cm());

        // We need U and its inverse, L^\dagger and its inverse.
        complex<double>* U_mat = allocate_vector<complex<double>>(lat->get_size_cm());
        complex<double>* U_mat_inv = allocate_vector<complex<double>>(lat->get_size_cm());
        complex<double>* Ldag_mat = allocate_vector<complex<double>>(lat->get_size_cm());
        complex<double>* Ldag_mat_inv = allocate_vector<complex<double>>(lat->get_size_cm());

        // Grab L, U from the transfer operator, transpose L.
        in_transfer->copy_LU(Ldag_mat, U_mat);
        cMATconjtrans_square(Ldag_mat, lat->get_volume(), lat->get_nc());

        // Prepare to get inverses.
        complex<double>* Qmat = allocate_vector<complex<double>>(lat->get_size_cm());
        complex<double>* Rmat = allocate_vector<complex<double>>(lat->get_size_cm());

        // Get inverse of U.
        zero_vector(Qmat, lat->get_size_cm());
        zero_vector(Rmat, lat->get_size_cm());
        cMATx_do_qr_square(U_mat, Qmat, Rmat, lat->get_volume(), lat->get_nc());
        cMATqr_do_xinv_square(Qmat, Rmat, U_mat_inv, lat->get_volume(), lat->get_nc());

        // Get inverse of L^\dagger
        zero_vector(Qmat, lat->get_size_cm());
        zero_vector(Rmat, lat->get_size_cm());
        cMATx_do_qr_square(Ldag_mat, Qmat, Rmat, lat->get_volume(), lat->get_nc());
        cMATqr_do_xinv_square(Qmat, Rmat, Ldag_mat_inv, lat->get_volume(), lat->get_nc());

        // Clean up.
        deallocate_vector(&Qmat);
        deallocate_vector(&Rmat);

        // Build up a matrix of sigma_1's. 
        const int nc = lat->get_nc();
        const int nc2 = lat->get_nc()*lat->get_nc();
        double sigma1_single[nc2];
        for (int i = 0; i < nc2; i++)
        {
          sigma1_single[i] = 0.0;
        }
        for (int i = 0; i < nc; i++)
        {
          if (i < nc/2)
          {
            sigma1_single[i*nc + i + (nc/2)] = 1.0;
          }
          else
          {
            sigma1_single[i*nc + i - (nc/2)] = 1.0;
          }
        }
        complex<double>* sigma1_repeat = allocate_vector<complex<double>>(lat->get_size_cm());
        zero_vector(sigma1_repeat, lat->get_size_cm());
        capx_pattern(sigma1_single, nc2, sigma1_repeat, lat->get_volume());

        // Prepare to form sigma_1^L and sigma_1^R
        complex<double>* extra_cmatrix = allocate_vector<complex<double>>(lat->get_size_cm());

        // sigma_1_L is U^{-\dagger} \sigma-1 L
        // but when we left apply we apply the dagger, 
        // which is L^\dagger \sigma_1 U^{-1}
        cMATxtMATyMATz_square(Ldag_mat, sigma1_repeat, extra_cmatrix, lat->get_volume(), lat->get_nc());
        cMATxtMATyMATz_square(extra_cmatrix, U_mat_inv, sigma_1_L, lat->get_volume(), lat->get_nc());

        // sigma_1^R is U \sigma_1 L^{-\dagger}
        cMATxtMATyMATz_square(U_mat, sigma1_repeat, extra_cmatrix, lat->get_volume(), lat->get_nc());
        cMATxtMATyMATz_square(extra_cmatrix, Ldag_mat_inv, sigma_1_R, lat->get_volume(), lat->get_nc());


        // Aaaaand we're ready to clean all the things!
        deallocate_vector(&extra_cmatrix);
        deallocate_vector(&sigma1_repeat);
        deallocate_vector(&Ldag_mat);
        deallocate_vector(&Ldag_mat_inv);
        deallocate_vector(&U_mat);
        deallocate_vector(&U_mat_inv);
      }

      switch (type)
      {
        case QMG_SIGMA_1_L:
        cMATxy(sigma_1_L, input, output, lat->get_volume(), lat->get_nc(), lat->get_nc());
        break;
        case QMG_SIGMA_1_R:
          cMATxy(sigma_1_R, input, output, lat->get_volume(), lat->get_nc(), lat->get_nc());
          break;
        case QMG_SIGMA_1_L_RBJ:
          if (!built_rbj_dagger)
          {
            std::cout << "[QMG-ERROR]: In apply_sigma, cannot apply QMG_SIGMA_1_L_RBJ without rbjacobi dagger stencil.\n";
            copy_vector(output, input, lat->get_size_cv());
          }
          else
          {
            // Apply B^{-dagger} \sigma_1^L. (since we need to left apply \gamma_5 B^{-1})
            cMATxy(sigma_1_L, input, extra_cvector, lat->get_volume(), lat->get_nc(), lat->get_nc());
            cMATxy(rbj_dagger_cinv, extra_cvector, output, lat->get_volume(), lat->get_nc(), lat->get_nc());
          }
          break;
        case QMG_SIGMA_1_R_RBJ:
          if (!built_rbjacobi)
          {
            std::cout << "[QMG-ERROR]: In apply_sigma, cannot apply QMG_SIGMA_1_R_RBJ without rbjacobi stencil.\n";
            copy_vector(output, input, lat->get_size_cv());
          }
          else
          {
            // Apply B \sigma_1^R.
            cMATxy(sigma_1_R, input, extra_cvector, lat->get_volume(), lat->get_nc(), lat->get_nc());
            cMATxy(clover, extra_cvector, output, lat->get_volume(), lat->get_nc(), lat->get_nc());
            caxpy(shift, extra_cvector, output, lat->get_size_cv());
          }
          break;
      }
    }
    
  }


};

#endif
