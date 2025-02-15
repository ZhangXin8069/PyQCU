#include <iostream>
#include <complex>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

#include "blas/generic_vector.h"

void test_orthogonalize(const int n_ortho, const int length)
{
  // Iterators
  int i, j;

  // Random number generator
  std::mt19937 generator (1337u);  

  // Populate some vectors.
  vector<complex<double>*> vecs(n_ortho);
  for (i = 0; i < n_ortho; i++)
  {
    vecs[i] = allocate_vector<complex<double>>(length);

    gaussian(vecs[i], length, generator);
  }

  // Print <v_i, v_j> for now.
  std::cout << "M\n";
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < n_ortho; j++)
    {
      std::cout << dot(vecs[i], vecs[j], length) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Okay, now ortho.
  std::cout << "Sigma -> M = Sigma^\\dagger Sigma\n\n";
  complex<double> Sigma[n_ortho][n_ortho];
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < n_ortho; j++)
    {
      Sigma[i][j] = 0.0;
    }
  }
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < i; j++)
    {
      // divisor would be one if we normalize.
      complex<double> alpha = dot(vecs[j], vecs[i], length);
      Sigma[j][i] = alpha;
      caxpy(-alpha, vecs[j], vecs[i], length);
    }

    // We have a freedom on how to normalize since we need to take care of the 
    // magnitude and the phase. As a convention we're sticking it on the left.
    double cplx_norm = sqrt(norm2sq(vecs[i], length));
    Sigma[i][i] = cplx_norm;
    cax(1.0/cplx_norm, vecs[i], length);
  }
  // Print Sigma
  std::cout << "Sigma\n";
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < n_ortho; j++)
    {
      std::cout << Sigma[i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Test printing it again.
  std::cout << "Verify\n";
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < n_ortho; j++)
    {
      std::cout << dot(vecs[i], vecs[j], length) << " ";
    }
    std::cout << "\n";
  }

  for (i = 0; i < n_ortho; i++)
  {
    deallocate_vector(&vecs[i]);
  }
}

void test_biorthogonalize(const int n_ortho, const int length)
{
  // Iterators
  int i, j;


  // Random number generator
  std::mt19937 generator (1337u);  

  // Populate some vectors.
  vector<complex<double>*> left_vecs(n_ortho);
  vector<complex<double>*> right_vecs(n_ortho);
  for (i = 0; i < n_ortho; i++)
  {
    left_vecs[i] = allocate_vector<complex<double>>(length);
    right_vecs[i] = allocate_vector<complex<double>>(length);

    gaussian(left_vecs[i], length, generator);
    gaussian(right_vecs[i], length, generator);
  }

  // Print <l_i, r_j> for now.
  std::cout << "M\n";
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < n_ortho; j++)
    {
      std::cout << dot(left_vecs[i], right_vecs[j], length) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Okay, now bi-ortho.
  std::cout << "L and U -> M = L U\n\n";
  complex<double> L_mat[n_ortho][n_ortho];
  complex<double> U_mat[n_ortho][n_ortho];
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < n_ortho; j++)
    {
      L_mat[i][j] = U_mat[i][j] = 0.0;
    }
  }
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < i; j++)
    {
      // divisor would be one if we normalize.
      complex<double> alpha = dot(right_vecs[j], left_vecs[i], length);//dot(left_vecs[j], right_vecs[j], length);
      L_mat[i][j] = conj(alpha);
      caxpy(-alpha, left_vecs[j], left_vecs[i], length);
      complex<double> beta = dot(left_vecs[j], right_vecs[i], length);//dot(left_vecs[j], right_vecs[j], length);
      U_mat[j][i] = beta;
      caxpy(-beta, right_vecs[j], right_vecs[i], length);
    }

    // We have a freedom on how to normalize since we need to take care of the 
    // magnitude and the phase. As a convention we're sticking it on the left.
    complex<double> cplx_dot = dot(left_vecs[i], right_vecs[i], length);
    double cplx_norm = abs(cplx_dot);
    cax(polar(1.0/sqrt(cplx_norm), arg(cplx_dot)), left_vecs[i], length);
    cax(1.0/sqrt(cplx_norm), right_vecs[i], length);
    L_mat[i][i] = polar(sqrt(cplx_norm), arg(cplx_dot));
    U_mat[i][i] = sqrt(cplx_norm);
  }

  // Print L, U
  std::cout << "L\n";
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < n_ortho; j++)
    {
      std::cout << L_mat[i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  std::cout << "U\n";
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < n_ortho; j++)
    {
      std::cout << U_mat[i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";


  // Test printing it again.
  std::cout << "Verify.\n";
  for (i = 0; i < n_ortho; i++)
  {
    for (j = 0; j < n_ortho; j++)
    {
      std::cout << dot(left_vecs[i], right_vecs[j], length) << " ";
    }
    std::cout << "\n";
  }

  for (i = 0; i < n_ortho; i++)
  {
    deallocate_vector(&left_vecs[i]);
    deallocate_vector(&right_vecs[i]);
  }
}

int main(int argc, char** argv)
{
  // How many vectors are we bi-orthogonalizing?
  const int n_ortho = 2;

  // What's the length of these vectors?
  const int length = 1024;

  std::cout << "[BEGIN-ORTHONORMALIZE]\n";
  test_orthogonalize(n_ortho, length);
  std::cout << "[END-ORTHONORMALIZE]\n\n[BEGIN-BI-ORTHONORMALIZE]\n";
  test_biorthogonalize(n_ortho, length);
  std::cout << "[END-BI-ORTHONORMALIZE]\n";

  return 0;
}
