// Copyright (c) 2017 Evan S Weinberg
// Test of a gauged dwf op implementation.

#ifndef QMG_DWF
#define QMG_DWF

#include <iostream>
#include <complex>
using std::complex;

#include "blas/generic_vector.h"
#include "lattice/lattice.h"
#include "stencil/stencil_2d.h"


// Set up a gauged "Shamir" domain wall operator.
// This largely just requires overloading the 
// constructor, plus supplying a few convenience functions. 
template <int Ls>
struct Dwf2D : public Stencil2D
{
protected:
  // Get rid of copy, assignment.
  Dwf2D(Dwf2D const &);
  Dwf2D& operator=(Dwf2D const &);

  // Temporary space for eo solve.
  complex<double>* tmp_eo_space;

  complex<double> mass;
  double M5; 

public:

  // gamma5 shuffles
  double a[2*Ls];
  int shuffle[2*Ls];

  // It's not efficient, but pre-build Gamma5.
  // Should just use one of the shuffle blas routines...
  complex<double>* gamma5_dense_mat;

public:

  // Base constructor.
  Dwf2D(Lattice2D* in_lat, complex<double> mass, complex<double>* gauge_links, double M5 = -1.0)
    : Stencil2D(in_lat, QMG_PIECE_CLOVER_HOPPING, M5, 0.0, 0.0), mass(mass), M5(M5)
  {
    if (lat->get_nc() != 2*Ls)
    {
      std::cout << "[QMG-ERROR]: Dwf2D only supports Nc = 2 Ls.\n";
      return;
    }

    // Allocate temporary space.
    tmp_eo_space = allocate_vector<complex<double>>(lat->get_size_cv());

    // Prepare links.
    update_links(gauge_links);

    // Set up the gamma5 shuffles
    for (int i = 0; i < Ls; i++) {
      a[2*i] = 1.0;
      a[2*i+1] = -1.0;
      shuffle[2*i] = 2*(Ls-1-i);
      shuffle[2*i+1] = 2*(Ls-1-i)+1;
    }

    // Build the dense 
    gamma5_dense_mat = allocate_vector<complex<double>>(lat->get_nc()*lat->get_nc());
    for (int i = 0; i < 4*Ls*Ls; i++) {
      gamma5_dense_mat[i] = 0.0;
    }

    // Assign +/- 1 values
    for (int i = 0; i < Ls; i++) {
      gamma5_dense_mat[2*Ls-2 + i*(4*Ls-2)] = 1.0;
      gamma5_dense_mat[4*Ls-1 + i*(4*Ls-2)] = -1.0;
    }
  }

  ~Dwf2D()
  {
    deallocate_vector(&tmp_eo_space);
    deallocate_vector(&gamma5_dense_mat);
  }

public:

  // Abstract static functions.

  // Wilson has two dof per site.
  static int get_dof()
  {
    return 2*Ls;
  }

  // Wilson has a sense of chirality.
  static chirality_state has_chirality()
  {
    return QMG_CHIRAL_YES; 
  }

  virtual void gamma5(complex<double>* vec)
  {
    cMAT_single_xy(gamma5_dense_mat, vec, tmp_eo_space, lat->get_volume(), 2*Ls, 2*Ls);
    copy_vector(vec, tmp_eo_space, lat->get_size_cv());
  }

  virtual void gamma5(complex<double>* g5_vec, complex<double>* vec)
  {
    caxy_shuffle_pattern(a, shuffle, 2*Ls, vec, g5_vec, lat->get_volume());
    //cMAT_single_xy(gamma5_dense_mat, vec, g5_vec, lat->get_volume(), 2*Ls, 2*Ls);
  }

  // First component per site is up, second component per site is down.
  virtual void chiral_projection(complex<double>* vector, bool is_up)
  {
    /*if (is_up)
      zero_vector_blas(vector+1, 2, lat->get_size_cv()/2);
    else
      zero_vector_blas(vector, 2, lat->get_size_cv()/2);*/
  }

  // Copy projection onto up, down.
  virtual void chiral_projection_copy(complex<double>* orig, complex<double>* dest, bool is_up)
  {
    /*if (is_up)
    {
      zero_vector_blas(dest+1, 2, lat->get_size_cv()/2);
      copy_vector_blas(dest, orig, 2, lat->get_size_cv()/2);
    }
    else
    {
      zero_vector_blas(dest, 2, lat->get_size_cv()/2);
      copy_vector_blas(dest+1, orig+1, 2, lat->get_size_cv()/2);
    }*/
  }

  // Copy the down projection into a new vector, perform the up in place.
  virtual void chiral_projection_both(complex<double>* orig_to_up, complex<double>* down)
  {
    /*zero_vector_blas(down, 2, lat->get_size_cv()/2);
    copy_vector_blas(down+1, orig_to_up+1, 2, lat->get_size_cv()/2);
    zero_vector_blas(orig_to_up+1, 2, lat->get_size_cv()/2);*/
  }

  virtual QMGDefaultChirality get_default_chirality()
  {
    return QMG_CHIRALITY_GAMMA_5;
  }

  // update gauge links.
  void update_links(complex<double>* gauge_links)
  {
    // Iterators.
    int j;

    // Prepare for complex numbers.
    const complex<double> cplxI(0.0,1.0);

    // Yup.
    const double wilson_coeff = 1.0;

    // Get the color matrix size. We stride with this when we
    // use the blas'.
    const int nc2 = lat->get_nc()*lat->get_nc();

    // Get the volume (number of sites).
    const int volume = lat->get_volume();
    const int cm_size = lat->get_size_cm();
    const int u1_cm_size = cm_size/nc2; 

    // Zero everything out since the Dwf operator is rather sparse.
    zero_vector(clover, cm_size);
    zero_vector(hopping, 4*cm_size);

    // M5 goes into the shift with this operator.

    // Since this is Shamir, there are Ls copies of the Wilson op along the 2x2 block diagonal.
    for (j = 0; j < Ls; j++)
    {
      // Clover term.
      constant_vector_blas(clover+j*(4*Ls+2), nc2, 3.0*wilson_coeff, volume);
      constant_vector_blas(clover+j*(4*Ls+2)+2*Ls+1, nc2, 3.0*wilson_coeff, volume);

      // Hopping.

      // +x
      caxpy_blas(-0.5*wilson_coeff, gauge_links, 1, hopping+j*(4*Ls+2), nc2, volume);
      caxpy_blas(0.5, gauge_links, 1, hopping+j*(4*Ls+2)+1, nc2, volume);
      caxpy_blas(0.5, gauge_links, 1, hopping+j*(4*Ls+2)+2*Ls, nc2, volume);
      caxpy_blas(-0.5*wilson_coeff, gauge_links, 1, hopping+j*(4*Ls+2)+2*Ls+1, nc2, volume);

      // +y
      caxpy_blas(-0.5*wilson_coeff, gauge_links+u1_cm_size, 1, hopping+cm_size+j*(4*Ls+2), nc2, volume);
      caxpy_blas(-0.5*cplxI, gauge_links+u1_cm_size, 1, hopping+cm_size+j*(4*Ls+2)+1, nc2, volume);
      caxpy_blas(0.5*cplxI, gauge_links+u1_cm_size, 1, hopping+cm_size+j*(4*Ls+2)+2*Ls, nc2, volume);
      caxpy_blas(-0.5*wilson_coeff, gauge_links+u1_cm_size, 1, hopping+cm_size+j*(4*Ls+2)+2*Ls+1, nc2, volume);

      // -x requires a cshift, conj.
      // structure: 0.5*{{-w, -1},{-1, -w}}
      cshift(priv_cmatrix, gauge_links, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, 1, lat);
      conj_vector(priv_cmatrix, u1_cm_size);
      caxpy_blas(-0.5*wilson_coeff, priv_cmatrix, 1, hopping+2*cm_size+j*(4*Ls+2), nc2, volume);
      caxpy_blas(-0.5, priv_cmatrix, 1, hopping+2*cm_size+j*(4*Ls+2)+1, nc2, volume);
      caxpy_blas(-0.5, priv_cmatrix, 1, hopping+2*cm_size+j*(4*Ls+2)+2*Ls, nc2, volume);
      caxpy_blas(-0.5*wilson_coeff, priv_cmatrix, 1, hopping+2*cm_size+j*(4*Ls+2)+2*Ls+1, nc2, volume);

      // -y requires a cshift, conj
      // structure: 0.5*{{-w, I},{-I,-w}}
      cshift(priv_cmatrix, gauge_links + u1_cm_size, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, 1, lat);
      conj_vector(priv_cmatrix, u1_cm_size);
      caxpy_blas(-0.5*wilson_coeff, priv_cmatrix, 1, hopping+3*cm_size+j*(4*Ls+2), nc2, volume);
      caxpy_blas(0.5*cplxI, priv_cmatrix, 1, hopping+3*cm_size+j*(4*Ls+2)+1, nc2, volume);
      caxpy_blas(-0.5*cplxI, priv_cmatrix, 1, hopping+3*cm_size+j*(4*Ls+2)+2*Ls, nc2, volume);
      caxpy_blas(-0.5*wilson_coeff, priv_cmatrix, 1, hopping+3*cm_size+j*(4*Ls+2)+2*Ls+1, nc2, volume);
    }

    // Then the clover has a few additional pieces.
    // Off diagonal P_+ and P_-.
    const double signfix = -1.;
    for (j = 0; j < Ls-1; j++)
    {
      // -P_+
      constant_vector_blas(clover+j*(4*Ls+2)+4*Ls, nc2, signfix*1.0, volume);

      // -P_-
      constant_vector_blas(clover+j*(4*Ls+2)+2*Ls+3, nc2, signfix*1.0, volume);
    }

    // And the beautiful mass.
    // mP_-
    constant_vector_blas(clover+(2*Ls-1)*2*Ls+1, nc2, -signfix*mass, volume);

    // m P_+
    constant_vector_blas(clover+(2*Ls-2), nc2, -signfix*mass, volume);


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


};

// Make a fixed Ls dwf operator
static Stencil2D* createDwfLs(Lattice2D* in_lat, complex<double> mass, complex<double>* gauge_links, int Ls, double M5 = -1.0)
{
  switch (Ls)
  {
    case 2:
      return new Dwf2D<2>(in_lat, mass, gauge_links, M5);
      break;
    case 4:
      return new Dwf2D<4>(in_lat, mass, gauge_links, M5);
      break;
    case 6:
      return new Dwf2D<6>(in_lat, mass, gauge_links, M5);
      break;
    case 8:
      return new Dwf2D<8>(in_lat, mass, gauge_links, M5);
      break;
    case 12:
      return new Dwf2D<12>(in_lat, mass, gauge_links, M5);
      break;
    case 16:
      return new Dwf2D<16>(in_lat, mass, gauge_links, M5);
      break;
    case 24:
      return new Dwf2D<24>(in_lat, mass, gauge_links, M5);
      break;
    case 32:
      return new Dwf2D<32>(in_lat, mass, gauge_links, M5);
      break;
    default:
      std::cout << "[QMG-ERROR]: Unsupported Ls " << Ls << " for domain wall operator. Add a template to dwf.h.\n";
      return nullptr;
  }
}


#endif
