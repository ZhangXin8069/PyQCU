// Copyright (c) 2017 Evan S Weinberg
// Test and make sure orthogonalization and bi-ortho 
// save the appropriate matrix decompositions.
// THE COMPARISONS WITH EXPLICIT ORTHO
// ARE NOT DATA PARALLEL

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <random>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"

// QMG
#include "lattice/lattice.h"
#include "transfer/transfer.h"


int main(int argc, char** argv)
{
  // Set output to not be that long.
  cout << setiosflags(ios::fixed) << setprecision(6);

  //Iterators and such.
  int i,j,k;
  int x_fine,y_fine;

  // Random number generator.
  std::mt19937 generator (1337u);

  // Basic information for fine level.
  const int x_len = 8;
  const int y_len = 8;
  const int dof = 1;

  // Blocking size.
  const int x_block = 4;
  const int y_block = 4;

  const int x_coarse = x_len/x_block;
  const int y_coarse = y_len/y_block;

  // Number of null vectors (equiv. degrees of freedom on coarse level)
  const int coarse_dof = 2;

  // Create a lattice object for the fine lattice.
  Lattice2D* lat = new Lattice2D(x_len, y_len, dof);

  // Create a lattice object for the coarse lattice.
  Lattice2D* coarse_lat = new Lattice2D(x_len/x_block, y_len/y_block, coarse_dof);

  // Coarse, fine size.
  const int fine_size_cv = lat->get_size_cv();
  const int coarse_size_cv = coarse_lat->get_size_cv();

  // Allocate space for vectors.
  complex<double>** prolong_vecs = new complex<double>*[coarse_dof];
  complex<double>** restrict_vecs = new complex<double>*[coarse_dof];
  for (i = 0; i < coarse_dof; i++)
  {
    prolong_vecs[i] = allocate_vector<complex<double>>(fine_size_cv);
    restrict_vecs[i] = allocate_vector<complex<double>>(fine_size_cv);
  }

  // Allocate space for two fine vectors.
  complex<double>* fine_vec_1 = allocate_vector<complex<double>>(fine_size_cv);
  complex<double>* fine_vec_2 = allocate_vector<complex<double>>(fine_size_cv);

  // Allocate space for two coarse vectors.
  complex<double>* coarse_vec_1 = allocate_vector<complex<double>>(coarse_size_cv);
  complex<double>* coarse_vec_2 = allocate_vector<complex<double>>(coarse_size_cv);

  ////////////////////////////////////////////
  // TEST 1: SYMMETRIC PROLONG AND RESTRICT //
  ////////////////////////////////////////////

  // Generate random vectors.
  for (i = 0; i < coarse_dof; i++)
  {
    gaussian(prolong_vecs[i], fine_size_cv, generator);
  }

  // Get the block Cholesky from block ortho.
  TransferMG* symmetric_transfer_obj = new TransferMG(lat, coarse_lat, prolong_vecs, true, true);
  complex<double>* block_Sigma = allocate_vector<complex<double>>(coarse_lat->get_size_cm());
  symmetric_transfer_obj->copy_cholesky(block_Sigma);

  // Do a block Gram Schmidt over each block separately.
  complex<double> M[x_coarse][y_coarse][coarse_dof][coarse_dof];
  complex<double> Sigma[x_coarse][y_coarse][coarse_dof][coarse_dof];

  for (x_fine = 0; x_fine < x_len; x_fine += x_block)
  {
    for (y_fine = 0; y_fine < y_len; y_fine += y_block)
    {
      // Form the outer product on the block.
      for (i = 0; i < coarse_dof; i++) for (j = 0; j < coarse_dof; j++)
      {
        M[x_fine/x_block][y_fine/y_block][i][j] = 0.0;
        for (int x_in = 0; x_in < x_block; x_in++)
        {
          for (int y_in = 0; y_in < y_block; y_in++)
          {
            M[x_fine/x_block][y_fine/y_block][i][j] += conj(prolong_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)])*
                              prolong_vecs[j][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)];
          }
        }
      }

      // Get sigma on each block.
      for (i = 0; i < coarse_dof; i++) for (j = 0; j < coarse_dof; j++)
        Sigma[x_fine/x_block][y_fine/y_block][i][j] = 0.0;

      for (i = 0; i < coarse_dof; i++)
      {
        for (j = 0; j < i; j++)
        {
          // Get Sigma_ij
          for (int x_in = 0; x_in < x_block; x_in++)
          {
            for (int y_in = 0; y_in < y_block; y_in++)
            {
              Sigma[x_fine/x_block][y_fine/y_block][j][i] += conj(prolong_vecs[j][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)])*
                                prolong_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)];
            }
          }

          // i -= alpha*j
          for (int x_in = 0; x_in < x_block; x_in++)
          {
            for (int y_in = 0; y_in < y_block; y_in++)
            {
              prolong_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)] -= 
                Sigma[x_fine/x_block][y_fine/y_block][j][i]*prolong_vecs[j][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)];
            }
          }
        }

        for (int x_in = 0; x_in < x_block; x_in++)
        {
          for (int y_in = 0; y_in < y_block; y_in++)
          {
            Sigma[x_fine/x_block][y_fine/y_block][i][i] += abs(conj(prolong_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)])*
                              prolong_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)]);
          }
        }

        Sigma[x_fine/x_block][y_fine/y_block][i][i] = sqrt(Sigma[x_fine/x_block][y_fine/y_block][i][i]);
        for (int x_in = 0; x_in < x_block; x_in++)
        {
          for (int y_in = 0; y_in < y_block; y_in++)
          {
            prolong_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)] /= Sigma[x_fine/x_block][y_fine/y_block][i][i];
          }
        }
      }

      // Check the explicit decomposition.
      std::cout << "Coarse (" << x_fine/x_block << ", " << y_fine/y_block << ") Frob = ";

      double sym_frob_norm = 0.0;
      for (i = 0; i < coarse_dof; i++)
      {
        for (j = 0; j < coarse_dof; j++)
        {
          complex<double> sig_dag_sig = 0.0;
          for (k = 0; k < coarse_dof; k++)
          {
            sig_dag_sig += conj(Sigma[x_fine/x_block][y_fine/y_block][k][i])*Sigma[x_fine/x_block][y_fine/y_block][k][j];
          }
          sym_frob_norm += abs(M[x_fine/x_block][y_fine/y_block][i][j] - sig_dag_sig)*abs(M[x_fine/x_block][y_fine/y_block][i][j] - sig_dag_sig);
        }
      }
      std::cout << sym_frob_norm << "\n\n";

      // Compare this Sigma with the Sigma from the block Chol.
      std::cout << "Explicit Sigma:\n";
      for (i = 0; i < coarse_dof; i++)
      {
        for (j = 0; j < coarse_dof; j++)
        {
          std::cout << Sigma[x_fine/x_block][y_fine/y_block][i][j] << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n\n";
      std::cout << "Block Chol Sigma:\n";
      for (i = 0; i < coarse_dof; i++)
      {
        for (j = 0; j < coarse_dof; j++)
        {
          std::cout << block_Sigma[coarse_lat->cm_coord_to_index(x_fine/x_block,y_fine/y_block,i,j)] << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n\n";
    }
  }


  // Clean up.
  deallocate_vector(&block_Sigma);

  delete symmetric_transfer_obj;

  /////////////////////////////////////////////
  // TEST 2: ASYMMETRIC PROLONG AND RESTRICT //
  /////////////////////////////////////////////

  // Generate random vectors.
  for (i = 0; i < coarse_dof; i++)
  {
    gaussian(prolong_vecs[i], fine_size_cv, generator);
    gaussian(restrict_vecs[i], fine_size_cv, generator);
  }

  // Get the block LU from block biortho.
  TransferMG* asymmetric_transfer_obj = new TransferMG(lat, coarse_lat, prolong_vecs, restrict_vecs, true, true);
  complex<double>* block_L = allocate_vector<complex<double>>(coarse_lat->get_size_cm());
  complex<double>* block_U = allocate_vector<complex<double>>(coarse_lat->get_size_cm());
  asymmetric_transfer_obj->copy_LU(block_L, block_U);

  // Do a block Gram Schmidt over each block separately.
  //complex<double> M[x_coarse][y_coarse][coarse_dof][coarse_dof];
  complex<double> L[x_coarse][y_coarse][coarse_dof][coarse_dof];
  complex<double> U[x_coarse][y_coarse][coarse_dof][coarse_dof];

  for (x_fine = 0; x_fine < x_len; x_fine += x_block)
  {
    for (y_fine = 0; y_fine < y_len; y_fine += y_block)
    {
      // Form the outer product on the block.
      for (i = 0; i < coarse_dof; i++) for (j = 0; j < coarse_dof; j++)
      {
        M[x_fine/x_block][y_fine/y_block][i][j] = 0.0;
        for (int x_in = 0; x_in < x_block; x_in++)
        {
          for (int y_in = 0; y_in < y_block; y_in++)
          {
            M[x_fine/x_block][y_fine/y_block][i][j] += conj(restrict_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)])*
                              prolong_vecs[j][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)];
          }
        }
      }

      // Get sigma on each block.
      for (i = 0; i < coarse_dof; i++) for (j = 0; j < coarse_dof; j++)
      {
        L[x_fine/x_block][y_fine/y_block][i][j] = 0.0;
        U[x_fine/x_block][y_fine/y_block][i][j] = 0.0;
      }

      for (i = 0; i < coarse_dof; i++)
      {
        for (j = 0; j < i; j++)
        {
          // Get L, U.
          for (int x_in = 0; x_in < x_block; x_in++)
          {
            for (int y_in = 0; y_in < y_block; y_in++)
            {
              L[x_fine/x_block][y_fine/y_block][i][j] += conj(prolong_vecs[j][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)])*
                                restrict_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)];
              U[x_fine/x_block][y_fine/y_block][j][i] += conj(restrict_vecs[j][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)])*
                                prolong_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)];
            }
          }

          // i -= alpha*j
          for (int x_in = 0; x_in < x_block; x_in++)
          {
            for (int y_in = 0; y_in < y_block; y_in++)
            {
              restrict_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)] -= 
                L[x_fine/x_block][y_fine/y_block][i][j]*restrict_vecs[j][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)];
              prolong_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)] -= 
                U[x_fine/x_block][y_fine/y_block][j][i]*prolong_vecs[j][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)];
            }
          }

          // Now conjugate L.
          L[x_fine/x_block][y_fine/y_block][i][j] = conj(L[x_fine/x_block][y_fine/y_block][i][j]);
        }

        // Normalize R[i]*L[i].
        for (int x_in = 0; x_in < x_block; x_in++)
        {
          for (int y_in = 0; y_in < y_block; y_in++)
          {
            L[x_fine/x_block][y_fine/y_block][i][i] += conj(restrict_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)])*
                              prolong_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)];
          }
        }
        U[x_fine/x_block][y_fine/y_block][i][i] = sqrt(abs(L[x_fine/x_block][y_fine/y_block][i][i]));
        L[x_fine/x_block][y_fine/y_block][i][i] /= U[x_fine/x_block][y_fine/y_block][i][i];

        for (int x_in = 0; x_in < x_block; x_in++)
        {
          for (int y_in = 0; y_in < y_block; y_in++)
          {
            prolong_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)] /= U[x_fine/x_block][y_fine/y_block][i][i];
            restrict_vecs[i][lat->cv_coord_to_index(x_fine+x_in, y_fine+y_in, 0)] /= conj(L[x_fine/x_block][y_fine/y_block][i][i]);
          }
        }
      }

      // Check the explicit decomposition.
      std::cout << "Coarse (" << x_fine/x_block << ", " << y_fine/y_block << ") Frob = ";

      double sym_frob_norm = 0.0;
      for (i = 0; i < coarse_dof; i++)
      {
        for (j = 0; j < coarse_dof; j++)
        {
          complex<double> L_t_U = 0.0;
          for (k = 0; k < coarse_dof; k++)
          {
            L_t_U += L[x_fine/x_block][y_fine/y_block][i][k]*U[x_fine/x_block][y_fine/y_block][k][j];
          }
          sym_frob_norm += abs(M[x_fine/x_block][y_fine/y_block][i][j] - L_t_U)*abs(M[x_fine/x_block][y_fine/y_block][i][j] - L_t_U);
        }
      }
      std::cout << sym_frob_norm << "\n\n";

      // Compare this L with the L from the block LU.
      std::cout << "Explicit L:\n";
      for (i = 0; i < coarse_dof; i++)
      {
        for (j = 0; j < coarse_dof; j++)
        {
          std::cout << L[x_fine/x_block][y_fine/y_block][i][j] << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n\n";
      std::cout << "Block L:\n";
      for (i = 0; i < coarse_dof; i++)
      {
        for (j = 0; j < coarse_dof; j++)
        {
          std::cout << block_L[coarse_lat->cm_coord_to_index(x_fine/x_block,y_fine/y_block,i,j)] << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n\n";

      // Compare this U with the U from the block LU.
      std::cout << "Explicit U:\n";
      for (i = 0; i < coarse_dof; i++)
      {
        for (j = 0; j < coarse_dof; j++)
        {
          std::cout << U[x_fine/x_block][y_fine/y_block][i][j] << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n\n";
      std::cout << "Block U:\n";
      for (i = 0; i < coarse_dof; i++)
      {
        for (j = 0; j < coarse_dof; j++)
        {
          std::cout << block_U[coarse_lat->cm_coord_to_index(x_fine/x_block,y_fine/y_block,i,j)] << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n\n";
    }
  }


  // Clean up.
  deallocate_vector(&block_L);
  deallocate_vector(&block_U);

  ////////////////////////////
  // AND THAT'S ALL, FOLKS! //
  ////////////////////////////

  // Clean up.
  deallocate_vector(&coarse_vec_1);
  deallocate_vector(&coarse_vec_2);
  deallocate_vector(&fine_vec_1);
  deallocate_vector(&fine_vec_2);

  for (i = 0; i < coarse_dof; i++)
  {
    deallocate_vector(&prolong_vecs[i]);
    deallocate_vector(&restrict_vecs[i]);
  }
  delete[] prolong_vecs;
  delete[] restrict_vecs;

  return 0;
}
