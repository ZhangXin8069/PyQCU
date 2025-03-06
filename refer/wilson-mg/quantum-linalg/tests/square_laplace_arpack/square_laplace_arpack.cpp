// Copyright (c) 2017 Evan S Weinberg
// Test code for a complex operator.

#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <complex>
#include <random>

#include "blas/generic_vector.h"

#include "interfaces/arpack/generic_arpack.h"


struct laplace_gauged_struct
{
  int length;
  double m_sq;
  std::complex<double>* gauge_links; // size 2*length*length
};

using namespace std; 

// Square laplacian function with gauge links. 
void square_laplacian_gauged(complex<double>* lhs, complex<double>* rhs, void* extra_data);

int main(int argc, char** argv)
{  
  //double *lattice; // Holds the gauge field.
  complex<double> *lhs, *rhs, *check; // For some Kinetic terms.
  complex<double> *gauge_links; 
  std::mt19937 generator (1337u); // RNG, 1337u is the seed. 

  // Basic information about the lattice.
  int length = 16;
  double m_sq = 0.01;
  
  // Create a random compact U(1) link.
  gauge_links = allocate_vector<complex<double>>(2*length*length);
  
  //constant_vector(gauge_links, 1.0, 2*length*length);
  // Remark: fills real and imaginary with gaussian numbers.
  double inv_variance = 1.0; // inverse of variance for gaussian non-compact U(1) links.
  gaussian(gauge_links, 2*length*length, generator, 1.0/inv_variance);
  polar(gauge_links, 2*length*length); // ignores imag part to make compact link.
  
  
  // Structure which gets passed to the function.
  laplace_gauged_struct lapstr;
  lapstr.length = length;
  lapstr.m_sq = m_sq;
  lapstr.gauge_links = gauge_links; 
  
  // Parameters related to solve.
  double tol = 1e-7; // can set to zero to solve it to insanity.
  int max_iter = 4000;
  
  // Some start-up.
  int volume = length*length;
  

  // Vectors. 
  lhs = allocate_vector<complex<double>>(volume);
  rhs = allocate_vector<complex<double>>(volume);
  check = allocate_vector<complex<double>>(volume);
  
  // Prepare an arpack structure for grabbing 5 eigenvectors.
  arpack_dcn* arpack = new arpack_dcn(volume, max_iter, tol,
                            square_laplacian_gauged, (void*)&lapstr,
                            5, 16);

  if(!arpack->prepare_eigensystem(arpack_dcn::ARPACK_SMALLEST_MAGNITUDE, 5))
  {
    cout << "[ERROR]: Znaupd code: " << arpack->get_solve_info().znaupd_code << "\n";

    delete arpack;  

    deallocate_vector(&lhs);
    deallocate_vector(&rhs);
    deallocate_vector(&check);
    deallocate_vector(&gauge_links); 
    return -1;
  }


  complex<double>* eigs = new complex<double>[5];

  arpack->get_eigensystem(eigs, arpack_dcn::ARPACK_SMALLEST_MAGNITUDE);

  for (int i = 0; i < 5; i++)
  {
    std::cout << "Eigenvalue " << i << " has value " << eigs[i] << "\n";
  }

  complex<double>** evecs = new complex<double>*[5];
  for (int i = 0; i < 5; i++)
  {
    evecs[i] = allocate_vector<complex<double> >(volume);
  }

  arpack->get_eigensystem(eigs, evecs, arpack_dcn::ARPACK_SMALLEST_MAGNITUDE);

  std::cout << "First few components of lowest system:\n";
  for (int j = 0; j < 16; j++)
    std::cout << evecs[0][j] << " ";
  std::cout << "\n";

  for (int i = 0; i < 5; i++)
    deallocate_vector(&evecs[i]);

  delete[] evecs; 
  delete[] eigs;
  delete arpack;  

  // Test getting the entire spectrum.
  // If you don't pass in a max eigvals, cv, it prepares you for a full system solve.
  arpack = new arpack_dcn(volume, max_iter, tol, square_laplacian_gauged, (void*)&lapstr);

  eigs = new complex<double>[volume];
  evecs = new complex<double>*[volume];
  for (int i = 0; i < volume; i++)
  {
    evecs[i] = new complex<double>[volume];
  }

  arpack->get_entire_eigensystem(eigs, arpack_dcn::ARPACK_SMALLEST_MAGNITUDE);

  for (int i = 0; i < volume; i++)
    std::cout << "Eigenvalue " << i << " has value " << eigs[i] << "\n";

  for (int i = 0; i < volume; i++)
  {
    delete[] evecs[i];
  }

  delete[] evecs;
  delete[] eigs;

  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&check);
  deallocate_vector(&gauge_links); 
  
  
}

// Square lattice.
// Kinetic term for a 2D laplacian w/ period bc. Applies lhs = A*rhs.
// The unit vectors are e_1 = xhat, e_2 = yhat.
// The "extra_data" allows us to generalize these functions later.
// It would become an internal structure in C++ code.
void square_laplacian_gauged(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  // Declare variables.
  int i;
  int x,y;
  laplace_gauged_struct* lapstr = (laplace_gauged_struct*)extra_data;

  int length = lapstr->length;
  int volume = lapstr->length*lapstr->length;
  double m_sq = lapstr->m_sq;
  complex<double>* gauge_links = lapstr->gauge_links; 
  
  // For a 2D square lattice, the stencil is:
  //     |  0 -1  0 |
  //     | -1 +4 -1 |
  //     |  0 -1  0 |
  //
  // e2 = yhat
  // ^
  // | 
  // |-> e1 = xhat

  // Apply the stencil.
  for (i = 0; i < volume; i++)
  {
    lhs[i] = 0.0;
    x = i%length; // integer mod.
    y = i/length; // integer divide.

    // + e1.
    lhs[i] = lhs[i]-gauge_links[y*length*2+x*2]*rhs[y*length+((x+1)%length)];

    // - e1.
    lhs[i] = lhs[i]-conj(gauge_links[y*length*2+((x+length-1)%length)*2])*rhs[y*length+((x+length-1)%length)];

    // + e2.
    lhs[i] = lhs[i]-gauge_links[y*length*2+x*2+1]*rhs[((y+1)%length)*length+x];

    // - e2.
    lhs[i] = lhs[i]-conj(gauge_links[((y+length-1)%length)*length*2+x*2+1])*rhs[((y+length-1)%length)*length+x];

    // 0
    // Added mass term here.
    lhs[i] = lhs[i]+(4+m_sq)*rhs[i];
  }

}


