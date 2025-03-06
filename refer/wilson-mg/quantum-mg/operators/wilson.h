// Copyright (c) 2017 Evan S Weinberg
// Test of a gauged wilson op implementation.

#ifndef QMG_WILSON
#define QMG_WILSON

#include <iostream>
#include <complex>
using std::complex;

#include "blas/generic_vector.h"
#include "lattice/lattice.h"
#include "stencil/stencil_2d.h"


// Set up a gauged Wilson operator.
// This largely just requires overloading the 
// constructor, plus supplying a few convenience functions. 
struct Wilson2D : public Stencil2D
{
protected:
  // Get rid of copy, assignment.
  Wilson2D(Wilson2D const &);
  Wilson2D& operator=(Wilson2D const &);

  // Temporary space for eo solve.
  complex<double>* tmp_eo_space;

  double wilson_coeff;

public:

  // Predeclare function to assign links.
  void update_links(complex<double>* gauge_links);

  // Base constructor.
  Wilson2D(Lattice2D* in_lat, complex<double> mass, complex<double>* gauge_links, double wilson_coeff = 1.0)
    : Stencil2D(in_lat, QMG_PIECE_CLOVER_HOPPING, mass, 0.0, 0.0), wilson_coeff(wilson_coeff)
  {
    if (lat->get_nc() != 2)
    {
      std::cout << "[QMG-ERROR]: Wilson2D only supports Nc = 2.\n";
      return;
    }

    // Allocate temporary space.
    tmp_eo_space = allocate_vector<complex<double>>(lat->get_size_cv());

    // Prepare links.
    update_links(gauge_links);
  }

  ~Wilson2D()
  {
    deallocate_vector(&tmp_eo_space);
  }

public:

  // Abstract static functions.

  // Wilson has two dof per site.
  static int get_dof(int i = 0)
  {
    return 2;
  }

  // Wilson has a sense of chirality.
  static chirality_state has_chirality()
  {
    return QMG_CHIRAL_YES; 
  }

  virtual void gamma5(complex<double>* vec)
  {
    const int nc = lat->get_nc();
    {
      for (int c = nc/2; c < nc; c++)
        cax_blas(-1.0, vec+c, nc, lat->get_size_cv()/nc);
    }
  }

  virtual void gamma5(complex<double>* g5_vec, complex<double>* vec)
  {
    const int nc = lat->get_nc();
    {
      for (int c = 0; c < nc/2; c++)
        caxy_blas(1.0, vec+c, nc, g5_vec+c, nc, lat->get_size_cv()/nc);

      for (int c = nc/2; c < nc; c++)
        caxy_blas(-1.0, vec+c, nc, g5_vec+c, nc, lat->get_size_cv()/nc);
    }
  }

  // First component per site is up, second component per site is down.
  virtual void chiral_projection(complex<double>* vector, bool is_up)
  {
    if (is_up)
      zero_vector_blas(vector+1, 2, lat->get_size_cv()/2);
    else
      zero_vector_blas(vector, 2, lat->get_size_cv()/2);
  }

  // Copy projection onto up, down.
  virtual void chiral_projection_copy(complex<double>* orig, complex<double>* dest, bool is_up)
  {
    if (is_up)
    {
      zero_vector_blas(dest+1, 2, lat->get_size_cv()/2);
      copy_vector_blas(dest, orig, 2, lat->get_size_cv()/2);
    }
    else
    {
      zero_vector_blas(dest, 2, lat->get_size_cv()/2);
      copy_vector_blas(dest+1, orig+1, 2, lat->get_size_cv()/2);
    }
  }

  // Copy the down projection into a new vector, perform the up in place.
  virtual void chiral_projection_both(complex<double>* orig_to_up, complex<double>* down)
  {
    zero_vector_blas(down, 2, lat->get_size_cv()/2);
    copy_vector_blas(down+1, orig_to_up+1, 2, lat->get_size_cv()/2);
    zero_vector_blas(orig_to_up+1, 2, lat->get_size_cv()/2);
  }

  // Apply sigma1 in place. Default does nothing.
  virtual void sigma1(complex<double>* vec)
  {
    double scale[2] = { 1.0, 1.0 };
    int shuffle[2] = { 1, 0 };
    caxy_shuffle_pattern(scale, shuffle, 2, vec, extra_cvector, lat->get_volume());
    copy_vector(vec, extra_cvector, lat->get_size_cv());
    return;
  }

  // Apply sigma1 saved in a vector
  virtual void sigma1(complex<double>* s1_vec, complex<double>* vec)
  {
    double scale[2] = { 1.0, 1.0 };
    int shuffle[2] = { 1, 0 };
    caxy_shuffle_pattern(scale, shuffle, 2, vec, s1_vec, lat->get_volume());
  }

  virtual QMGDefaultChirality get_default_chirality()
  {
    return QMG_CHIRALITY_GAMMA_5;
  }

};

// update gauge links.
void Wilson2D::update_links(complex<double>* gauge_links)
{
  // Prepare for complex numbers.
  const complex<double> cplxI(0.0,1.0);

  // Get the color matrix size. We stride with this when we
  // use the blas'.
  const int nc2 = lat->get_nc()*lat->get_nc();

  // Get the volume (number of sites).
  const int volume = lat->get_volume();


  // The clover term is the 2*wilson_term on the identity.
  constant_vector_blas(clover, nc2, 2.0*wilson_coeff, volume);
  constant_vector_blas(clover+1, nc2, 0.0, volume);
  constant_vector_blas(clover+2, nc2, 0.0, volume);
  constant_vector_blas(clover+3, nc2, 2.0*wilson_coeff, volume);


  // The hopping term is a bit more complicated, but not by much!
  const int cm_size = lat->get_size_cm();

  // Get the step for the nc = 1 gauge fields... oi.
  const int u1_cm_size = cm_size/nc2; 

  // +x
  // structure: 0.5*{{-w, 1},{1,-w}}*U_x(x)
  caxy_blas(-0.5*wilson_coeff, gauge_links, 1, hopping, nc2, volume);
  caxy_blas(0.5, gauge_links, 1, hopping+1, nc2, volume);
  caxy_blas(0.5, gauge_links, 1, hopping+2, nc2, volume);
  caxy_blas(-0.5*wilson_coeff, gauge_links, 1, hopping+3, nc2, volume);

  // +y
  // structure: 0.5*{{-w, -I},{I,-w}}
  caxy_blas(-0.5*wilson_coeff, gauge_links+u1_cm_size, 1, hopping+cm_size, nc2, volume);
  caxy_blas(-0.5*cplxI, gauge_links+u1_cm_size, 1, hopping+cm_size+1, nc2, volume);
  caxy_blas(0.5*cplxI, gauge_links+u1_cm_size, 1, hopping+cm_size+2, nc2, volume);
  caxy_blas(-0.5*wilson_coeff, gauge_links+u1_cm_size, 1, hopping+cm_size+3, nc2, volume);

  // -x requires a cshift, conj.
  // structure: 0.5*{{-w, -1},{-1, -w}}
  cshift(priv_cmatrix, gauge_links, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, 1, lat);
  conj_vector(priv_cmatrix, u1_cm_size);
  caxy_blas(-0.5*wilson_coeff, priv_cmatrix, 1, hopping+2*cm_size, nc2, volume);
  caxy_blas(-0.5, priv_cmatrix, 1, hopping+2*cm_size+1, nc2, volume);
  caxy_blas(-0.5, priv_cmatrix, 1, hopping+2*cm_size+2, nc2, volume);
  caxy_blas(-0.5*wilson_coeff, priv_cmatrix, 1, hopping+2*cm_size+3, nc2, volume);

  // -y requires a cshift, conj
  // structure: 0.5*{{-w, I},{-I,-w}}
  cshift(priv_cmatrix, gauge_links + u1_cm_size, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, 1, lat);
  conj_vector(priv_cmatrix, u1_cm_size);
  caxy_blas(-0.5*wilson_coeff, priv_cmatrix, 1, hopping+3*cm_size, nc2, volume);
  caxy_blas(0.5*cplxI, priv_cmatrix, 1, hopping+3*cm_size+1, nc2, volume);
  caxy_blas(-0.5*cplxI, priv_cmatrix, 1, hopping+3*cm_size+2, nc2, volume);
  caxy_blas(-0.5*wilson_coeff, priv_cmatrix, 1, hopping+3*cm_size+3, nc2, volume);

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


#endif
