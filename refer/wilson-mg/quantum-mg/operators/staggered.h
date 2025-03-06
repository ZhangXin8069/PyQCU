// Copyright (c) 2017 Evan S Weinberg
// Test of a gauged staggered op implementation.

#ifndef QMG_STAGGERED
#define QMG_STAGGERED

#include <iostream>
#include <complex>
using std::complex;

#include "blas/generic_vector.h"
#include "lattice/lattice.h"
#include "stencil/stencil_2d.h"

// Pre-declare an inline function to set eta phases.
inline void staggered_set_eta_y(int i, complex<double>& elem, void* extra_data);

// Set up a gauged staggered operator.
// This largely just requires overloading the 
// constructor, plus supplying a few convenience functions. 
struct Staggered2D : public Stencil2D
{
protected:
  // Get rid of copy, assignment.
  Staggered2D(Staggered2D const &);
  Staggered2D& operator=(Staggered2D const &);

  // Temporary space for eo solve.
  complex<double>* tmp_eo_space;

public:

  // Base constructor.
  Staggered2D(Lattice2D* in_lat, complex<double> mass, complex<double>* gauge_links)
    : Stencil2D(in_lat, QMG_PIECE_HOPPING, mass, 0.0, 0.0)
  {
    if (lat->get_nc() != 1)
    {
      std::cout << "[QMG-ERROR]: Staggered2D only supports Nc = 1.\n";
      return;
    }

    // Allocate temporary space.
    tmp_eo_space = allocate_vector<complex<double>>(lat->get_size_cv());

    // The gauged laplace doesn't have a self-interaction term besides the
    // mass, so the clover term never gets allocated.

    // The hopping term is a bit more complicated, but not by much!
    zero_vector(hopping, lat->get_size_hopping());

    const int cm_size = lat->get_size_cm();

    // The +x, +y hopping terms are easy.
    
    // +x
    caxpy(-0.5, gauge_links, hopping, cm_size); // Copy rescaled +x links in.

    // +y, set phases.
    caxpy(-0.5, gauge_links + cm_size, hopping + cm_size, cm_size);
    arb_local_function_vector(hopping + cm_size, staggered_set_eta_y, (void*)lat, cm_size);

    // -x requires a cshift, conj.
    cshift(priv_cmatrix, gauge_links, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, 1, lat);
    conj_vector(priv_cmatrix, cm_size);
    caxpy(0.5, priv_cmatrix, hopping + 2*cm_size, cm_size);

    // -y requires a cshift, conj, set phases.
    cshift(priv_cmatrix, gauge_links + cm_size, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, 1, lat);
    conj_vector(priv_cmatrix, cm_size);
    caxpy(0.5, priv_cmatrix, hopping + 3*cm_size, cm_size);
    arb_local_function_vector(hopping + 3*cm_size, staggered_set_eta_y, (void*)lat, cm_size);
  }

  ~Staggered2D()
  {
    deallocate_vector(&tmp_eo_space);
  }

  // update gauge links.
  void update_links(complex<double>* gauge_links)
  {
    // Only need to update hopping term. 
    zero_vector(hopping, lat->get_size_hopping());

    const int cm_size = lat->get_size_cm();

    // The +x, +y hopping terms are easy.
    
    // +x
    caxpy(-0.5, gauge_links, hopping, cm_size); // Copy rescaled +x links in.

    // +y, set phases.
    caxpy(-0.5, gauge_links + cm_size, hopping + cm_size, cm_size);
    arb_local_function_vector(hopping + cm_size, staggered_set_eta_y, (void*)lat, cm_size);

    // -x requires a cshift, conj.
    cshift(priv_cmatrix, gauge_links, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, 1, lat);
    conj_vector(priv_cmatrix, cm_size);
    caxpy(0.5, priv_cmatrix, hopping + 2*cm_size, cm_size);

    // -y requires a cshift, conj, set phases.
    cshift(priv_cmatrix, gauge_links + cm_size, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, 1, lat);
    conj_vector(priv_cmatrix, cm_size);
    caxpy(0.5, priv_cmatrix, hopping + 3*cm_size, cm_size);
    arb_local_function_vector(hopping + 3*cm_size, staggered_set_eta_y, (void*)lat, cm_size);

    // Kill rbjacobi, dagger links if they exist.
    if (built_dagger)
    {
      deallocate_vector(&dagger_clover);
      deallocate_vector(&dagger_hopping);
      built_dagger = false;
    }

    if (built_rbjacobi)
    {
      deallocate_vector(&rbjacobi_cinv);
      deallocate_vector(&rbjacobi_clover);
      deallocate_vector(&rbjacobi_hopping);
      built_rbjacobi = false;
    }
  }

public:
  // Abstract static functions.

  // Staggered has one dof per site.
  static int get_dof(int i = 0)
  {
    return 1;
  }

  // Staggered has a sense of chirality.
  static chirality_state has_chirality()
  {
    return QMG_CHIRAL_YES; 
  }

  virtual void gamma5(complex<double>* vec)
  {
    cax(-1.0, vec+lat->get_size_cv()/2, lat->get_size_cv()/2);
  }

  virtual void gamma5(complex<double>* g5_vec, complex<double>* vec)
  {
    copy_vector(g5_vec, vec, lat->get_size_cv()/2);
    caxy(-1.0, vec+lat->get_size_cv()/2, g5_vec+lat->get_size_cv()/2, lat->get_size_cv()/2);
  }

  // Chirality is even/odd. 
  virtual void chiral_projection(complex<double>* vector, bool is_up)
  {
    if (is_up)
      zero_vector(vector+(lat->get_size_cv()/2), lat->get_size_cv()/2); // zero odd
    else
      zero_vector(vector, lat->get_size_cv()/2); // zero even.
  }

  // Copy projection onto up, down.
  virtual void chiral_projection_copy(complex<double>* orig, complex<double>* dest, bool is_up)
  {
    if (is_up)
    {
      zero_vector(dest+lat->get_size_cv()/2, lat->get_size_cv()/2);
      copy_vector(dest, orig, lat->get_size_cv()/2);
    }
    else
    {
      zero_vector(dest, lat->get_size_cv()/2);
      copy_vector(dest+lat->get_size_cv()/2, orig+lat->get_size_cv()/2, lat->get_size_cv()/2);
    }
  }

  // Copy the down projection into a new vector, perform the up in place.
  virtual void chiral_projection_both(complex<double>* orig_to_up, complex<double>* down)
  {
    zero_vector(down, lat->get_size_cv()/2);
    copy_vector(down+lat->get_size_cv()/2, orig_to_up+lat->get_size_cv()/2, lat->get_size_cv()/2);
    zero_vector(orig_to_up+lat->get_size_cv()/2, lat->get_size_cv()/2);
  }

  virtual QMGDefaultChirality get_default_chirality()
  {
    return QMG_CHIRALITY_GAMMA_5;
  }

  // Custom functions to prepare for eo preconditioned solve.
  // b_new = (4 + m^2) b_e - D_{eo} b_o
  void prepare_b(complex<double>* b_new, complex<double>* b)
  {
    int cv_size = lat->get_size_cv();

    // Zero even part of b_new.
    zero_vector(b_new, cv_size/2);

    // Apply D_{eo}.
    apply_M_eo(b_new, b);

    // Form shift b_e - D_{eo} b_0
    caxpby(shift, b, complex<double>(-1.0), b_new, cv_size/2); // even only.
  }

  // Custom function for eo preconditioned op application.
  // Destroys odd piece of lhs.
  void apply_eo_prec_M(complex<double>* lhs, complex<double>* rhs)
  {
    int cv_size = lat->get_size_cv();

    // Zero lhs.
    zero_vector(lhs, cv_size/2);

    // Zero temp vector.
    zero_vector(tmp_eo_space, cv_size);

    // Apply D_{oe}
    apply_M_oe(tmp_eo_space, rhs);
    
    // Apply D_{eo}
    apply_M_eo(tmp_eo_space, tmp_eo_space);

    // Form (m^2 - D_{eo}D_{oe})
    caxpbyz(shift*shift, rhs, complex<double>(-1.0), tmp_eo_space, lhs, cv_size/2);
  }

  // Custom function to reconstruct after eo preconditioned solve.
  // x_0 = 1/m[b_o - D_{oe} x_e]
  void reconstruct_x(complex<double>* x, complex<double>* b)
  {
    int cv_size = lat->get_size_cv();

    // Zero odd part of x.
    zero_vector(x + cv_size/2, cv_size/2);

    // Apply D_{oe}
    apply_M_oe(x, x);

    // Form 1/m[b_o - D_{oe} x_e]
    caxpby(1.0/shift, b + cv_size/2, -1.0/shift, x + cv_size/2, cv_size/2);
  }

};

// Special C function wrapper for e-o stencil application.
void apply_eo_staggered_2D_M(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Staggered2D* stenc = (Staggered2D*)extra_data;
  stenc->apply_eo_prec_M(lhs, rhs); // internally zeroes lhs.
}

// Special C function to set eta phases. Expects a LatticeColorMatrix. Should get
// called twice: once for +y, once for -y.
inline void staggered_set_eta_y(int i, complex<double>& elem, void* extra_data)
{
  Lattice2D* lat = (Lattice2D*)extra_data;
  int x, y, c1, c2;
  lat->cm_index_to_coord(i, x, y, c1, c2);
  elem *= (double)(1.0-2.0*(x%2));
}

#endif
