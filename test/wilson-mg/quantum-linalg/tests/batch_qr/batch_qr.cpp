// Copyright (c) 2017 Evan S Weinberg
// Test code for batch QR operations. 

#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <complex>
#include <random>

#include "blas/generic_vector.h"
#include "blas/generic_matrix.h"


using namespace std; 


int main(int argc, char** argv)
{  
  
  std::mt19937 generator (1337u); // RNG, 1337u is the seed. 

  // Test the batch QR operations.

  const int num_matrix = 1024;
  const int n_dim = 12;

  complex<double>* A = allocate_vector<complex<double> >(num_matrix*n_dim*n_dim);
  complex<double>* Q = allocate_vector<complex<double> >(num_matrix*n_dim*n_dim);
  complex<double>* R = allocate_vector<complex<double> >(num_matrix*n_dim*n_dim);
  complex<double>* T = allocate_vector<complex<double> >(num_matrix*n_dim*n_dim);
  complex<double>* Ainv = allocate_vector<complex<double> >(num_matrix*n_dim*n_dim);
  complex<double>* Eye = allocate_vector<complex<double> >(num_matrix*n_dim*n_dim);


  // Initialize as appropriate.

  // A with random numbers.
  gaussian(A, num_matrix*n_dim*n_dim, generator);

  // Identity as appropriate.
  zero_vector(Eye, num_matrix*n_dim*n_dim);
  complex<double> identity_pattern[n_dim*n_dim];
  for (int i = 0; i < n_dim*n_dim; i++)
  {
    identity_pattern[i] = 0.0;
    if (i % (n_dim+1) == 0)
      identity_pattern[i] = 1.0;
  }
  capx_pattern(identity_pattern, n_dim*n_dim, Eye, num_matrix);

  // Test a batch QR decomposition.
  cMATx_do_qr_square(A, Q, R, num_matrix, n_dim);

  // Compute product T = QR as check.
  cMATxtMATyMATz_square(Q, R, T, num_matrix, n_dim);

  // Verify
  std::cout << "Checking QR decomposition: " << sqrt(diffnorm2sq(A, T, num_matrix*n_dim*n_dim)/norm2sq(A, num_matrix*n_dim*n_dim)) << "\n";

  // Test batch forming Ainv from the QR decomposition. 
  cMATqr_do_xinv_square(Q, R, Ainv, num_matrix, n_dim);

  // Compute product A Ainv as check.
  cMATxtMATyMATz_square(A, Ainv, T, num_matrix, n_dim);

  // Verify
  std::cout << "Checking QR inversion: " << sqrt(diffnorm2sq(Eye, T, num_matrix*n_dim*n_dim)/norm2sq(Eye, num_matrix*n_dim*n_dim)) << "\n";

  deallocate_vector(&A);
  deallocate_vector(&Q);
  deallocate_vector(&R);
  deallocate_vector(&T);
  deallocate_vector(&Ainv);
  deallocate_vector(&Eye);

  return 0;
}

