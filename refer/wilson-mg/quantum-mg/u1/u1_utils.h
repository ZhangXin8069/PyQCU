// Copyright (c) 2017 Evan S Weinberg

#ifndef QMG_U1_UTILS
#define QMG_U1_UTILS

#include <fstream>
#include <cmath>
#include <complex>
#include <random>

using std::complex; 
using std::fstream;
using std::ios;
using std::ios_base;
using std::string;

#include "blas/generic_vector.h"
#include "lattice/lattice.h"
#include "cshift/cshift_2d.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Different gauge field types. 
enum qmg_gauge_create_type
{
  GAUGE_LOAD = 0,      // Load a gauge field.
  GAUGE_RANDOM = 1,      // Create a gauge field with deviation 1/sqrt(beta)
  GAUGE_UNIT = 2       // Use a unit gauge field.
};


// Load complex gauge field from file. 
// Based on Rich Brower's u1 gauge routines. 
// Reads in a U1 phase lattice from file, returns complex fields. 
// Rich's code has 'y' as the fast direction. Need to convert to eo ordering.
void read_gauge_u1(complex<double>* gauge_field, Lattice2D* lat, string input_file)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  int x_len = lat->get_dim_mu(0);
  int y_len = lat->get_dim_mu(1);

  double phase_field;  
  fstream in_file;

  in_file.open(input_file,ios::in); 
  for(int x =0;x< x_len;x++)
  {
    for(int y =0;y< y_len;y++)
    {
      for(int mu=0; mu<2; mu++)
      {
        in_file >> phase_field;
        gauge_field[lat->gauge_coord_to_index(x, y, 0, 0, mu)] = polar(1.0,phase_field);
      }
    }
  }
  in_file.close(); 

  return;
}

// Just read the phases as opposed to compactifying them.
void read_phase_u1(double* phase_field, Lattice2D* lat, string input_file)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  int x_len = lat->get_dim_mu(0);
  int y_len = lat->get_dim_mu(1);

  double phase_tmp;  
  fstream in_file;

  in_file.open(input_file,ios::in); 
  for(int x =0;x< x_len;x++)
  {
    for(int y =0;y< y_len;y++)
    {
      for(int mu=0; mu<2; mu++)
      {
        in_file >> phase_tmp;
        phase_field[lat->gauge_coord_to_index(x, y, 0, 0, mu)] = phase_tmp;
      }
    }
  }
  in_file.close(); 

  return;
}

// Write complex gauge field to file.
// Based on Rich Brower's u1 gauge routines.
// Writes a U1 phase lattice from file from complex fields.
// Rich's code has 'y' as the fast direction. Need to transpose!
void write_gauge_u1(complex<double>* gauge_field, Lattice2D* lat, string output_file)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  int x_len = lat->get_dim_mu(0);
  int y_len = lat->get_dim_mu(1);

  double phase_field;
  fstream out_file;

  out_file.open(output_file, ios::in|ios::out|ios::trunc);
  out_file.setf(ios_base::fixed, ios_base::floatfield);
  out_file.precision(20);

  for(int x =0;x< x_len;x++)
  {
    for(int y =0;y< y_len;y++)
    {
      for(int mu=0; mu<2; mu++)
      {
        phase_field = arg(gauge_field[lat->gauge_coord_to_index(x,y,0,0,mu)]);
        out_file << phase_field << "\n";
      }
    }
  }
  out_file.close(); 
}

// Write phase field to file as opposed to the compact links.
void write_gauge_u1(double* phase_field, Lattice2D* lat, string output_file)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  int x_len = lat->get_dim_mu(0);
  int y_len = lat->get_dim_mu(1);

  double phase_tmp;
  fstream out_file;

  out_file.open(output_file, ios::in|ios::out|ios::trunc);
  out_file.setf(ios_base::fixed, ios_base::floatfield);
  out_file.precision(20);

  for(int x =0;x< x_len;x++)
  {
    for(int y =0;y< y_len;y++)
    {
      for(int mu=0; mu<2; mu++)
      {
        phase_tmp = phase_field[lat->gauge_coord_to_index(x,y,0,0,mu)];
        out_file << phase_tmp << "\n";
      }
    }
  }
  out_file.close(); 
}

// Create a unit gauge field.
// Just set everything to 1!
void unit_gauge_u1(complex<double>* gauge_field, Lattice2D* lat)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  constant_vector(gauge_field, 1.0, lat->get_size_gauge());
}

// Create a hot gauge field, uniformly distributed in -Pi -> Pi.
// mt19937 can be created+seeded as: std::mt19937 generator (seed1);
void rand_gauge_u1(complex<double>* gauge_field, Lattice2D* lat, std::mt19937 &generator)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  random_uniform(gauge_field, lat->get_size_gauge(), generator, -PI, PI);
  polar(gauge_field,lat->get_size_gauge());
}

// Create a gaussian gauge field with variance = 1/beta
// beta -> 0 is a hot start, beta -> inf is a cold start. 
// Based on code by Rich Brower, re-written for C++11.
void gauss_gauge_u1(complex<double>* gauge_field, Lattice2D* lat, std::mt19937 &generator, double beta)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  // Take abs value of beta.
  if (beta < 0) { beta = -beta; }

  // If beta is 0, just call a hot start.
  if (beta == 0)
  {
    rand_gauge_u1(gauge_field, lat, generator);
  }
  else
  {
    // create phase.
    gaussian(gauge_field, lat->get_size_gauge(), generator, 1.0/sqrt(beta));
    // promote to U(1).
    polar(gauge_field, lat->get_size_gauge());
  } 
}

// Generate a random gauge transform.
// mt19937 can be created+seeded as: std::mt19937 generator (seed1);
void rand_trans_u1(complex<double>* gauge_trans, Lattice2D* lat, std::mt19937 &generator)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  random_uniform(gauge_trans, lat->get_size_cm(), generator, -PI, PI);
  polar(gauge_trans, lat->get_size_cm());
}

// Apply a gauge transform:
// u_i(x) = g(x) u_i(x) g^\dag(x+\hat{i})
void apply_gauge_trans_u1(complex<double>* gauge_field, complex<double>* gauge_trans, Lattice2D* lat)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  int size_cm = lat->get_size_cm();
  complex<double>* tmp = allocate_vector<complex<double>>(size_cm);
  complex<double>* cm_forward = allocate_vector<complex<double>>(size_cm);

  // Update x link.
  copy_vector(tmp, gauge_trans, size_cm);
  cxty(gauge_field, tmp, size_cm);
  cshift(cm_forward, gauge_trans, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, 1, lat);
  conj_vector(cm_forward, size_cm);
  cxty(cm_forward, tmp, size_cm);
  copy_vector(gauge_field, tmp, size_cm);

  // Update y link.
  copy_vector(tmp, gauge_trans, size_cm);
  cxty(gauge_field + size_cm, tmp, size_cm);
  cshift(cm_forward, gauge_trans, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, 1, lat);
  conj_vector(cm_forward, size_cm);
  cxty(cm_forward, tmp, size_cm);
  copy_vector(gauge_field + size_cm, tmp, size_cm);

  // Clean up. 
  deallocate_vector(&tmp);
  deallocate_vector(&cm_forward);
}

// Apply ape smearing with parameter \alpha, n_iter times.
// Based on code from Rich Brower
void apply_ape_smear_u1(complex<double>* smeared_field, complex<double>* gauge_field, Lattice2D* lat, double alpha, int n_iter)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  int size_cm = lat->get_size_cm();
  complex<double>* smeared_tmp = allocate_vector<complex<double>>(lat->get_size_gauge());
  complex<double>* link_vec = allocate_vector<complex<double>>(size_cm);
  complex<double>* cm_shift = allocate_vector<complex<double>>(size_cm);
  complex<double>* cm_shift2 = allocate_vector<complex<double>>(size_cm);

  copy_vector(smeared_tmp, gauge_field, lat->get_size_gauge());

  for (int i = 0; i < n_iter; i++)
  {
    // x first.

    // ----
    copy_vector(smeared_field, smeared_tmp, size_cm);

    // ----
    // |  |
    
    // U_y(x)
    copy_vector(link_vec, smeared_tmp + size_cm, size_cm);
    // U_x(x+yhat)
    cshift(cm_shift, smeared_tmp, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, 1, lat);
    cxty(cm_shift, link_vec, size_cm);
    // conj(U_y(x+xhat))
    cshift(cm_shift, smeared_tmp + size_cm, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, 1, lat);
    conj_vector(cm_shift, size_cm);
    cxty(cm_shift, link_vec, size_cm);
    // Rescale alpha.
    caxpy(alpha, link_vec, smeared_field, size_cm);

    // |  |
    // ----

    // conj(U_y(x-yhat))
    cshift(link_vec, smeared_tmp + size_cm, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, 1, lat);
    copy_vector(cm_shift2, link_vec, size_cm); // for +x-y.
    conj_vector(link_vec, size_cm);
    // U_x(x-yhat)
    cshift(cm_shift, smeared_tmp, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, 1, lat);
    cxty(cm_shift, link_vec, size_cm);
    // U_y(x+xhat-yhat). Could really use the QMG_CSHIFT_FROM_XP1YM1 here.
    cshift(cm_shift, cm_shift2, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, 1, lat);
    cxty(cm_shift, link_vec, size_cm);
    // Rescale alpha.
    caxpy(alpha, link_vec, smeared_field, size_cm);

    // y second.

    //  |
    //  |
    //  |
    copy_vector(smeared_field + size_cm, smeared_tmp + size_cm, size_cm);

    // ----
    //    |
    // ----
    
    // U_x(x)
    copy_vector(link_vec, smeared_tmp, size_cm);
    // U_y(x+xhat)
    cshift(cm_shift, smeared_tmp + size_cm, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, 1, lat);
    cxty(cm_shift, link_vec, size_cm);
    // conj(U_x(x+yhat))
    cshift(cm_shift, smeared_tmp, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, 1, lat);
    conj_vector(cm_shift, size_cm);
    cxty(cm_shift, link_vec, size_cm);
    // Rescale alpha.
    caxpy(alpha, link_vec, smeared_field, size_cm);

    // ----
    // |
    // ----

    // conj(U_x(x-xhat))
    cshift(link_vec, smeared_tmp, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, 1, lat);
    copy_vector(cm_shift2, link_vec, size_cm); // for -x+y.
    conj_vector(link_vec, size_cm);
    // U_y(x-xhat)
    cshift(cm_shift, smeared_tmp + size_cm, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, 1, lat);
    cxty(cm_shift, link_vec, size_cm);
    // U_x(x-xhat+yhat). Could really use the QMG_CSHIFT_FROM_XM1YP1 here.
    cshift(cm_shift, cm_shift2, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, 1, lat);
    cxty(cm_shift, link_vec, size_cm);
    // Rescale alpha.
    caxpy(alpha, link_vec, smeared_field, size_cm);

    // Project everything back to U(1)
    arg_vector(smeared_field, lat->get_size_gauge());
    polar(smeared_field, lat->get_size_gauge());
    copy_vector(smeared_tmp, smeared_field, lat->get_size_gauge());

  }

  copy_vector(smeared_field, smeared_tmp, lat->get_size_gauge());

  deallocate_vector(&smeared_tmp);
  deallocate_vector(&link_vec);
  deallocate_vector(&cm_shift);
  deallocate_vector(&cm_shift2);
}

// Get the non-compact action
double get_noncompact_action_u1(double* phase_field, double beta, Lattice2D* lat) {
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return -50;
  }


  int size_cm = lat->get_size_cm();
  double* phase_vec = allocate_vector<double>(size_cm);
  double* cm_forward = allocate_vector<double>(size_cm);

  // A_x(x)
  copy_vector(phase_vec, phase_field, size_cm); 

  // A_y(x+xhat)
  cshift(cm_forward, phase_field + size_cm, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, 1, lat);
  caxpy(1.0, cm_forward, phase_vec, size_cm);

  // -A_x(x+yhat)
  cshift(cm_forward, phase_field, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, 1, lat);
  conj_vector(cm_forward, size_cm);
  caxpy(-1.0, cm_forward, phase_vec, size_cm);

  // -A_y(x)
  caxpy(-1.0, phase_field + size_cm, phase_vec, size_cm);

  // Sum and normalize.
  double action = 0.5*beta*norm2sq(phase_vec, size_cm);

  // Clean up.
  deallocate_vector(&phase_vec);
  deallocate_vector(&cm_forward);

  return action;
}

// Get average plaquette. 
complex<double> get_plaquette_u1(complex<double>* gauge_field, Lattice2D* lat)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return -50;
  }


  int size_cm = lat->get_size_cm();
  complex<double>* plaq_vec = allocate_vector<complex<double>>(size_cm);
  complex<double>* cm_forward = allocate_vector<complex<double>>(size_cm);

  // U_x(x)
  copy_vector(plaq_vec, gauge_field, size_cm); 

  // U_y(x+xhat)
  cshift(cm_forward, gauge_field + size_cm, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, 1, lat);
  cxty(cm_forward, plaq_vec, size_cm);

  // U_x^\dagger(x+yhat)
  cshift(cm_forward, gauge_field, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, 1, lat);
  conj_vector(cm_forward, size_cm);
  cxty(cm_forward, plaq_vec, size_cm);

  // U_y(x)
  conj_vector(gauge_field + size_cm, size_cm);
  cxty(gauge_field + size_cm, plaq_vec, size_cm);
  conj_vector(gauge_field + size_cm, size_cm);

  // Sum and normalize.
  complex<double> plaq = sum_vector(plaq_vec, size_cm)/(double)lat->get_volume();

  // Clean up.
  deallocate_vector(&plaq_vec);
  deallocate_vector(&cm_forward);

  return plaq;

}

// Get the topological charge.
double get_topo_u1(complex<double>* gauge_field, Lattice2D* lat)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return -50.1;
  }

  int size_cm = lat->get_size_cm();
  complex<double>* plaq_vec = allocate_vector<complex<double>>(size_cm);
  complex<double>* cm_forward = allocate_vector<complex<double>>(size_cm);

  // U_x(x)
  copy_vector(plaq_vec, gauge_field, size_cm); 

  // U_y(x+xhat)
  cshift(cm_forward, gauge_field + size_cm, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, 1, lat);
  cxty(cm_forward, plaq_vec, size_cm);

  // U_x^\dagger(x+yhat)
  cshift(cm_forward, gauge_field, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, 1, lat);
  conj_vector(cm_forward, size_cm);
  cxty(cm_forward, plaq_vec, size_cm);

  // U_y(x)
  conj_vector(gauge_field + size_cm, size_cm);
  cxty(gauge_field + size_cm, plaq_vec, size_cm);
  conj_vector(gauge_field + size_cm, size_cm);

  // Take arg.
  arg_vector(plaq_vec, size_cm);

  // Sum.
  double topo = real(sum_vector(plaq_vec, size_cm))*0.5/PI;

  // Clean up.
  deallocate_vector(&plaq_vec);
  deallocate_vector(&cm_forward);

  // Return.
  return topo;

}

// Go to Lorentz gauge
void lorentz_gauge_fix_u1(complex<double>* gauge_field, Lattice2D* lat, const double delta, const double tol, const int max_iter)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  const int cm_size = lat->get_size_cm();

  // Oi. Create a U(1) field.
  complex<double>* phi = allocate_vector<complex<double>>(cm_size);
  constant_vector(phi, 1.0, cm_size);

  // Create a derivative field.
  complex<double>* deriv_phi = allocate_vector<complex<double>>(cm_size);
  constant_vector(deriv_phi, 0.0, cm_size);

  // Meh.
  double resid = 1.0;

  while (resid > tol)
  {
    // Build the derivative with respect to theta.
    
  }

  // Clean up.
  deallocate_vector(&phi);
  deallocate_vector(&deriv_phi);

}

// Create an instanton.
void create_instanton_u1(complex<double>* gauge_field, Lattice2D* lat, double Q, const int x0, const int y0)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  const int xlen = lat->get_dim_mu(0);
  const int ylen = lat->get_dim_mu(1);

  // Unfortunately, we break our data parallel setup 
  // when we create an instanton. It'd be too much of a headache
  // (for now) to stick to it.
  for (int x = 0; x < xlen; x++)
  {
    for (int y = 0; y < ylen; y++)
    {
      // This is for an instanton at the origin.
      double rx = x - xlen/2 + 0.5;
      double ry = y - ylen/2 + 0.5;

      // Center the instanton appropriately.
      gauge_field[lat->gauge_coord_to_index((x-xlen/2+x0+3*xlen)%xlen, (y-ylen/2+y0+3*ylen)%ylen, 0, 0, 0)] *= polar(1.0, Q*ry/(rx*rx+ry*ry));
      gauge_field[lat->gauge_coord_to_index((x-xlen/2+x0+3*xlen)%xlen, (y-ylen/2+y0+3*ylen)%ylen, 0, 0, 1)] *= polar(1.0, -Q*rx/(rx*rx+ry*ry));
    }
  }
}

// Create a non-compact instanton.
void create_noncompact_instanton_u1(double* phase_field, Lattice2D* lat, double Q)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  const int xlen = lat->get_dim_mu(0);
  const int ylen = lat->get_dim_mu(1);

  // Unfortunately, we break our data parallel setup 
  // when we create an instanton. It'd be too much of a headache
  // (for now) to stick to it.
  for (int x = 0; x < xlen; x++)
  {
    for (int y = 0; y < ylen; y++)
    {
      // This is for an instanton at the origin.
      //double rx = x - xlen/2; // + 0.5;
      //double ry = y - ylen/2; // + 0.5;

      // Center the instanton appropriately.
      phase_field[lat->gauge_coord_to_index(x, y, 0, 0, 0)] += -Q*3.1415926535*y/(xlen*ylen); //Q*ry/(rx*rx+ry*ry);
      if (y == ylen-1)
        phase_field[lat->gauge_coord_to_index(x, y, 0, 0, 1)] += Q*3.1415926535*x/xlen; //-Q*rx/(rx*rx+ry*ry);
    }
  }
}

// Perform a non-compact heatbath phase update.
// sqrt{PI/beta} exp( -beta * [(theta + 1/2(staple1+ staple2))^2 + const])
void heatbath_noncompact_update(double* phase_field, Lattice2D* lat, double beta, int n_update, std::mt19937 &generator)
{
  if (lat->get_nc() != 1)
  {
    cout << "[QMG-ERROR]: U1 gauge functions require Nc = 1 lattice.\n";
    return;
  }

  // Calculate the width for the heatbath now.
  double width = sqrt(0.5/beta);
  std::normal_distribution<> dist(0.0, width);

  const int xlen = lat->get_dim_mu(0);
  const int ylen = lat->get_dim_mu(1);

  /*
  complex<double>* staple = allocate_vector<complex<double>>(size_cm);
  complex<double>* staple_accum = allocate_vector<complex<double>>(size_cm);
  complex<double>* cm_shift = allocate_vector<complex<double>>(size_cm);
  complex<double>* cm_shift2 = allocate_vector<complex<double>>(size_cm);
  */


  // Should be just a double, but it's all the same.
  double staple; 
  
  for (int i = 0; i < n_update; i++)
  {
    // Unfortunately, we need to hard code loop over sites.
    // This algorithm can't be parallelized as is...
    // We would need subsets. Or to just wait for HMC. 

    // Update x.
    for (int x = 0; x < xlen; x++)
    {
      for (int y = 0; y < ylen; y++)
      {
        staple = phase_field[lat->gauge_coord_to_index((x+1)%xlen, y, 0, 0, 1)];
        staple -= phase_field[lat->gauge_coord_to_index(x, (y+1)%ylen, 0, 0, 0)];
        staple -= phase_field[lat->gauge_coord_to_index(x, y, 0, 0, 1)];
        staple -= phase_field[lat->gauge_coord_to_index((x+1)%xlen, (y-1+ylen)%ylen, 0, 0, 1)];
        staple -= phase_field[lat->gauge_coord_to_index(x, (y-1+ylen)%ylen, 0, 0, 0)];
        staple += phase_field[lat->gauge_coord_to_index(x, (y-1+ylen)%ylen, 0, 0, 1)];
        phase_field[lat->gauge_coord_to_index(x, y, 0, 0, 0)] = dist(generator) - 0.5*staple;
      }
    }

    // Update y.
    for (int x = 0; x < xlen; x++)
    {
      for (int y = 0; y < ylen; y++)
      {
        staple = phase_field[lat->gauge_coord_to_index(x, (y+1)%ylen, 0, 0, 0)];
        staple -= phase_field[lat->gauge_coord_to_index((x+1)%xlen, y, 0, 0, 1)];
        staple -= phase_field[lat->gauge_coord_to_index(x, y, 0, 0, 0)];
        staple -= phase_field[lat->gauge_coord_to_index((x-1+xlen)%xlen, (y+1)%ylen, 0, 0, 0)];
        staple -= phase_field[lat->gauge_coord_to_index((x-1+xlen)%xlen, y, 0, 0, 1)];
        staple += phase_field[lat->gauge_coord_to_index((x-1+xlen)%xlen, y, 0, 0, 0)];
        phase_field[lat->gauge_coord_to_index(x, y, 0, 0, 1)] = dist(generator) - 0.5*staple;
      }
    }

    // Update y.

    /*

    // x first.

    // ----
    // |  |
    
    // +A_y(x)
    copy_vector(staple, phase_field + size_cm, size_cm);
    // +A_x(x+yhat)
    cshift(cm_shift, phase_field, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, 1, lat);
    caxpy(1.0, cm_shift, staple, size_cm);
    // -A_y(x+xhat)
    cshift(cm_shift, phase_field + size_cm, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, 1, lat);
    caxpy(-1.0, cm_shift, staple, size_cm);
    copy_vector(staple_accum, staple, size_cm);

    // |  |
    // ----

    // -A_y(x-yhat)
    cshift(cm_shift2, phase_field + size_cm, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, 1, lat);
    caxy(-1.0, cm_shift2, staple, size_cm); // for +x-y.

    // +A_x(x-yhat)
    cshift(cm_shift, phase_field, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, 1, lat);
    caxpy(1.0, cm_shift, staple, size_cm);

    // A_y(x+xhat-yhat). Could really use the QMG_CSHIFT_FROM_XP1YM1 here.
    cshift(cm_shift, cm_shift2, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, 1, lat);
    caxpy(1.0, cm_shift, staple, size_cm);
    caxpy(1.0, staple, staple_accum, size_cm);

    // Do heatbath update on x links.
    gaussian_real(phase_field, size_cm, generator, width);
    caxpy(0.5, staple_accum, phase_field, size_cm);


    // y second.

    // ----
    //    |
    // ----
    
    // +A_x(x)
    copy_vector(staple, phase_field, size_cm);
    // +A_y(x+xhat)
    cshift(cm_shift, phase_field + size_cm, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVENODD, 1, lat);
    caxpy(1.0, cm_shift, staple, size_cm);
    // -A_x(x+yhat)
    cshift(cm_shift, phase_field, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, 1, lat);
    caxpy(-1.0, cm_shift, staple, size_cm);
    copy_vector(staple_accum, staple, size_cm);

    // ----
    // |
    // ----

    // -A_x(x-xhat)
    cshift(cm_shift2, phase_field, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, 1, lat);
    caxy(-1.0, cm_shift2, staple, size_cm); // for -x+y.

    // +A_y(x-xhat)
    cshift(cm_shift, phase_field + size_cm, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, 1, lat);
    caxpy(1.0, cm_shift, staple, size_cm);

    // A_x(x+yhat-xhat). Could really use the QMG_CSHIFT_FROM_XM1YP1 here.
    cshift(cm_shift, cm_shift2, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVENODD, 1, lat);
    caxpy(1.0, cm_shift, staple, size_cm);
    caxpy(1.0, staple, staple_accum, size_cm);

    // Do heatbath update on y links.
    gaussian_real(phase_field + size_cm, size_cm, generator, width);
    caxpy(0.5, staple_accum, phase_field + size_cm, size_cm);
    */
    // Normalize phases.
    //polar(phase_field, size_gauge);
    //arg_vector(phase_field, size_gauge);
  }
 
  /*
  deallocate_vector(&staple);
  deallocate_vector(&staple_accum);
  deallocate_vector(&cm_shift);
  deallocate_vector(&cm_shift2);
  */
}
	
#endif // U1_UTILS