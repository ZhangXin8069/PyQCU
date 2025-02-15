// Copyright (c) 2017 Evan S Weinberg
// Test of a gauged square laplace implementation.

#ifndef QMG_LAPLACE
#define QMG_LAPLACE

#include <iostream>
#include <complex>
using std::complex;

#include "blas/generic_vector.h"
#include "lattice/lattice.h"
#include "stencil/stencil_2d.h"

// Set up a gauged laplace.
// This largely just requires overloading the 
// constructor.
struct GaugedLaplace2D : public Stencil2D
{
protected:
  // Get rid of copy, assignment.
  GaugedLaplace2D(GaugedLaplace2D const &);
  GaugedLaplace2D& operator=(GaugedLaplace2D const &);

  // Temporary space for eo solve.
  complex<double>* tmp_eo_space;

public:

  // Base constructor.
  GaugedLaplace2D(Lattice2D* in_lat, complex<double> mass_sq, complex<double>* gauge_links)
    : Stencil2D(in_lat, QMG_PIECE_CLOVER_HOPPING, mass_sq, 0.0, 0.0)
  {
    if (lat->get_nc() != 1)
    {
      std::cout << "[QMG-ERROR]: GaugedLaplace2D only supports Nc = 1.\n";
      return;
    }

    // Allocate temporary space.
    tmp_eo_space = allocate_vector<complex<double>>(lat->get_size_cv());

    // The gauged laplace doesn't have a self-interaction term besides the
    // mass, so it's just a 4 on the clover.
    constant_vector(clover, 4.0, lat->get_size_cm());

    // The hopping term is a bit more complicated, but not by much!
    zero_vector(hopping, lat->get_size_hopping());

    const int cm_size = lat->get_size_cm();

    // The +x, +y hopping terms are easy.
    
    // +x
    caxpy(-1.0, gauge_links, hopping, cm_size); // Copy rescaled +x links in.

    // +y
    caxpy(-1.0, gauge_links + cm_size, hopping + cm_size, cm_size);

    // -x requires a cshift, conj.
    cshift(priv_cmatrix, gauge_links, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, 1, lat);
    conj_vector(priv_cmatrix, cm_size);
    caxpy(-1.0, priv_cmatrix, hopping + 2*cm_size, cm_size);

    // -y requires a cshift, conj.
    cshift(priv_cmatrix, gauge_links + cm_size, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, 1, lat);
    conj_vector(priv_cmatrix, cm_size);
    caxpy(-1.0, priv_cmatrix, hopping + 3*cm_size, cm_size);
  }

  ~GaugedLaplace2D()
  {
    deallocate_vector(&tmp_eo_space);
  }

  // update gauge links.
  void update_links(complex<double>* gauge_links)
  {
    // Only need to update the hopping term.
    zero_vector(hopping, lat->get_size_hopping());

    const int cm_size = lat->get_size_cm();

    // The +x, +y hopping terms are easy.
    
    // +x
    caxpy(-1.0, gauge_links, hopping, cm_size); // Copy rescaled +x links in.

    // +y
    caxpy(-1.0, gauge_links + cm_size, hopping + cm_size, cm_size);

    // -x requires a cshift, conj.
    cshift(priv_cmatrix, gauge_links, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, 1, lat);
    conj_vector(priv_cmatrix, cm_size);
    caxpy(-1.0, priv_cmatrix, hopping + 2*cm_size, cm_size);

    // -y requires a cshift, conj.
    cshift(priv_cmatrix, gauge_links + cm_size, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, 1, lat);
    conj_vector(priv_cmatrix, cm_size);
    caxpy(-1.0, priv_cmatrix, hopping + 3*cm_size, cm_size);

    // Kill rbjacobi, dagger links if they exist.
    if (built_dagger)
    {
      deallocate_vector(&dagger_hopping);
      built_dagger = false;
    }

    if (built_rbjacobi)
    {
      deallocate_vector(&rbjacobi_cinv);
      deallocate_vector(&rbjacobi_hopping);
      built_rbjacobi = false;
    }
  }

  // Square laplace has one dof per site.
  static int get_dof(int i = 0)
  {
    return 1;
  }

  // Square laplace has no sense of chirality.
  static chirality_state has_chirality()
  {
    return QMG_CHIRAL_NO; 
  }

  // Chiral projections are uninteresting.
  virtual void chiral_projection(complex<double>* vector, bool is_up)
  {
    return;
  }

  // Copy projection onto up, down.
  virtual void chiral_projection_copy(complex<double>* orig, complex<double>* dest, bool is_up)
  {
    return;
  }

  // Copy the down projection into a new vector, perform the up in place.
  virtual void chiral_projection_both(complex<double>* orig_to_up, complex<double>* down)
  {
    return; 
  }

  virtual QMGDefaultChirality get_default_chirality()
  {
    return QMG_CHIRALITY_NONE;
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

    // Form (4 + m^2) b_e - D_{eo} b_0
    caxpby(4.0+shift, b, complex<double>(-1.0), b_new, cv_size/2); // even only.
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

    // Form ((4+m^2)^2 - D_{eo}D_{oe})
    caxpbyz((4.0+shift)*(4.0+shift), rhs, complex<double>(-1.0), tmp_eo_space, lhs, cv_size/2);
  }

  // Custom function to reconstruct after eo preconditioned solve.
  // x_0 = 1/(4+m^2)[b_o - D_{oe} x_e]
  void reconstruct_x(complex<double>* x, complex<double>* b)
  {
    int cv_size = lat->get_size_cv();

    // Zero odd part of x.
    zero_vector(x + cv_size/2, cv_size/2);

    // Apply D_{oe}
    apply_M_oe(x, x);

    // Form 1/(4+m^2)[b_o - D_{oe} x_e]
    caxpby(1.0/(4.0+shift), b + cv_size/2, -1.0/(4.0+shift), x + cv_size/2, cv_size/2);
  }

};

// Special C function wrapper for e-o stencil application.
void apply_eo_gauge_laplace_2D_M(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  GaugedLaplace2D* stenc = (GaugedLaplace2D*)extra_data;
  stenc->apply_eo_prec_M(lhs, rhs); // internally zeroes lhs.
}

#endif
