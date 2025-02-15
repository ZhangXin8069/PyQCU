// Copyright (c) 2017 Evan S Weinberg
// Test code for a real operator.
// Computes 'N' lowest eigenvalues
// by applying power iterations to 
// 

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <sstream>
#include <random>

#include "blas/generic_vector.h"

using namespace std; 

struct laplace_struct
{
  int length;
  double m_sq;
};

// Square laplacian function.
void square_laplacian(double* lhs, double* rhs, void* extra_data);
void square_laplacian_shift(double* lhs, double* rhs, void* extra_data);


int main(int argc, char** argv)
{  
  // Iterators.
  int i,j,k;

  // Set output precision to be long.
  cout << setprecision(20);

  // Random number generator.
  std::mt19937 generator (1337u);

  // Basic information about the lattice.
  int length = 64;
  int volume = length*length;
  double m_sq = 0.01;
  
  // Structure which gets passed to the function.
  laplace_struct lapstr;
  lapstr.length = length;
  lapstr.m_sq = m_sq;
  
  // Parameters related to eigensolve.
  double tol = 1e-5;
  int max_iter = 5000;
  
  // How many eigenvalues are we trying to find?
  int n_eigen = 10;

  // What's our lock range? (i.e., if we want 10 eigenvalues,
  // do we only search lowest 20 for convergence?)
  int n_lock_interval = 10; 

  // How large of a search space do we have?
  int n_space = 10;

  // How often should we ortho?
  int ortho_freq = 100;
  double inv_freq = 1.0/((double)ortho_freq);

  // How often do we reset the locks?
  int lock_reset_freq = 5;

  // Verbosity settings
  bool print_swap = false; // Print every time we swap an eigenvalue
  bool print_step = true; // Print iteration number
  bool print_progression = false; // Print progression of Rayleigh quotients
  bool print_lock = true; // Print "converged" eigenvalues
  bool print_final = true; // Print eigenvalues at the end
  
  // Generate some vectors.
  double** evecs = new double*[n_space];
  double** new_evecs = new double*[n_space];
  for (i = 0; i < n_space; i++)
  {
    evecs[i] = allocate_vector<double>(volume);
    new_evecs[i] = allocate_vector<double>(volume);
    gaussian(evecs[i], volume, generator);
    normalize(evecs[i], volume);
    copy_vector(new_evecs[i], evecs[i], volume);
  }

  double* tmp_vec = allocate_vector<double>(volume);

  // Generate some storage space for eigenvalues.
  double* evals = new double[n_space];
  double* tmp_evals = new double[n_space];
  for (i = 0; i < n_space; i++)
  {
    evals[i] = 100.0; // something that's definitely not the eigenvalue.
    tmp_evals[i] = 0.0;
  }

  // Generate lock space. Once a vector has locked, we stop updating it.
  bool* lock_space = new bool[n_space];
  for (i = 0; i < n_space; i++)
    lock_space[i] = false;

  // Alrighty, let's roll!
  for (i = 0; i < max_iter; i += ortho_freq)
  {
    int count_done = 0;

    // Do ortho_freq hits of the matrix on each vector, copy back in.
    for (k = 0; k < n_space; k++)
    {
      if (lock_space[k])
        continue;

      for (j = 0; j < ortho_freq; j++)
      {
        zero_vector(tmp_vec, volume);
        square_laplacian_shift(tmp_vec, new_evecs[k], &lapstr);
        copy_vector(new_evecs[k], tmp_vec, volume);
      }
    }

    if (print_step) cout << "Finished iteration " << i+ortho_freq << "\n";

    // Grab the Rayleigh quotient, check convergence.
    for (k = 0; k < n_space; k++)
    {
      if (lock_space[k])
      {
        if (k < n_lock_interval)
        {
          count_done++;
          if (print_lock) cout << k << " " << 8.0 + m_sq - tmp_evals[k] << " "; 
        }

        continue;
      }

      double dotval = dot(new_evecs[k], evecs[k], volume);
      tmp_evals[k] = pow(dotval, inv_freq);
      //caxy(dotval, evecs[k], tmp_vec, volume);
      //if (diffnorm2sq(tmp_vec,new_evecs[k],volume)/norm2sq(new_evecs[k],volume) < tol)
      if (abs(tmp_evals[k]-evals[k])/evals[k] < tol)
      {
        lock_space[k] = true;
        if (k < n_lock_interval)
        {
          count_done++;
          if (print_lock) cout << k << " " << 8.0 + m_sq - tmp_evals[k] << " "; 
        }
      }
      evals[k] = tmp_evals[k];
    }
    if (print_lock) cout << "\n" << count_done << "\n";

    if (count_done >= n_eigen)
      break;

    // Sort by smallest to largest eigenvalue.
    // This means as much as possible gets ortho'd out of the
    // last eigenvector.
    for (j = 0; j < n_space; j++)
    {
      for (k = 0; k < n_space-1; k++)
      {
        if (evals[k] < evals[k+1])
        {
          swap(evals[k], evals[k+1]);
          swap(new_evecs[k], new_evecs[k+1]);
          swap(lock_space[k], lock_space[k+1]);
          if (print_swap) std::cout << j << " " << k << " SWAP!\n";
        }
      }
    }

    if (print_progression)
    {
      cout << i+ortho_freq << " ";
      for (k = 0; k < n_space; k++)
        cout << 8.0+m_sq-evals[k] << " ";
      cout << "\n";
    }

    // every so often kill the lock.
    if (lock_reset_freq > 0 && i % (lock_reset_freq*ortho_freq) == 0)
    {
      for (k = 0; k < n_space; k++)
        lock_space[k] = false;
    }

    // Orthonormalize via modified Gram Schmidt.
    for (k = 0; k < n_space; k++)
    {
      if (lock_space[k]) // only update vectors that haven't locked.
        continue;

      /*for (j = 0; j < k; j++)
      {
        orthogonal(new_evecs[k], new_evecs[j], volume);
      }*/
      for (j = 0; j < n_space; j++)
      {
        // Orthogonalize against previous OR locked eigenvectors.
        if (j != k && (j < k || lock_space[j]))
          orthogonal(new_evecs[k], new_evecs[j], volume);
      }
      normalize(new_evecs[k], volume);
      copy_vector(evecs[k], new_evecs[k], volume);
    }
  }

  // One last orthonormalize.
  for (k = 0; k < n_space; k++)
  {
    for (j = 0; j < k; j++)
    {
      orthogonal(new_evecs[k], new_evecs[j], volume);
    }
    normalize(new_evecs[k], volume);
    copy_vector(evecs[k], new_evecs[k], volume);
    zero_vector(tmp_vec, volume);
    square_laplacian_shift(tmp_vec, evecs[k], &lapstr);
    evals[k] = dot(tmp_vec, evecs[k], volume);
  }

  // Print eigenvalues.
  if (print_final)
  {
    for (k = 0; k < n_space; k++)
      cout << "Eigenvalue " << k << " = " << 8.0 + m_sq - evals[k] << "\n";
  }

  int tot_count = 0;
  for (k = 0; k < n_space; k++)
  {
    if (lock_space[k]) tot_count++;
  }
  cout << "Locked vectors: " << tot_count << "\n";

  /*for (i = 0; i < volume; i++)
  {
    cout << new_evecs[0][i] << "\n";
  }*/

  // Clean up.
  delete[] lock_space; 

  delete[] evals;
  delete[] tmp_evals;

  deallocate_vector(&tmp_vec);

  for (i = 0; i < n_space; i++)
  {
    deallocate_vector(&evecs[i]);
    deallocate_vector(&new_evecs[i]);
  }
  delete[] evecs;
  delete[] new_evecs;
}

// Square lattice.
// Kinetic term for a 2D laplacian w/ period bc. Applies lhs = A*rhs.
// The unit vectors are e_1 = xhat, e_2 = yhat.
// The "extra_data" allows us to generalize these functions later.
// It would become an internal structure in C++ code.
void square_laplacian(double* lhs, double* rhs, void* extra_data)
{
  // Declare variables.
  int i;
  int x,y;
  laplace_struct* lapstr = (laplace_struct*)extra_data;

  int length = lapstr->length;
  int volume = lapstr->length*lapstr->length;
  double m_sq = lapstr->m_sq;
  
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
    lhs[i] = lhs[i]-rhs[y*length+((x+1)%length)];

    // - e1.
    lhs[i] = lhs[i]-rhs[y*length+((x+length-1)%length)];

    // + e2.
    lhs[i] = lhs[i]-rhs[((y+1)%length)*length+x];

    // - e2.
    lhs[i] = lhs[i]-rhs[((y+length-1)%length)*length+x];

    // 0
    // Added mass term here.
    lhs[i] = lhs[i]+(4+m_sq)*rhs[i];
  }

}

// Shifted square laplace function. Looks at 8 + m^2 - Laplace op,
// which transforms the lowest eigenvalues into the highest
// eigenvalues (unless the mass is something stupid huge).
void square_laplacian_shift(double* lhs, double* rhs, void* extra_data)
{
  laplace_struct* lapstr = (laplace_struct*)extra_data;

  int volume = lapstr->length*lapstr->length;
  double m_sq = lapstr->m_sq;

  square_laplacian(lhs, rhs, extra_data);
  caxpby(8.0+m_sq, rhs, -1.0, lhs, volume);
}

