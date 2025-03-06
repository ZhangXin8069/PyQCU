// Copyright (c) Evan S Weinberg 2017
// Code for computing an unpivoted QR decomposition.
// Based heavily on code found online on 2017-04-10 by
// Carlos F. Borges: faculty.nps.edu/borges/Teaching/MA3046/
// as well as Wikipedia. 

// Perform a QR decomposition without pivoting. At the end, produces matrices
// Q and R such that A = Q*R.
// Arg 1: [out] orthonormal matrix Q. Assumed to be pre-allocated of size N_dim*N_dim.
// Arg 2: [out] upper right triangular matrix R. Assumed to be pre-allocated of size N_dim*N_dim.
// Arg 3: [in] square matrix A to be decomposed. Assumed to be pre-allocated of size N_dim*N_dim.
// Arg 4: [in] dimension of matrix. 
void qr_decomposition(std::complex<double>* Q, std::complex<double>* R, std::complex<double>* A, int N_dim)
{
  // Initialize Q and R, pass to internal function.

  for (int i = 0; i < N_dim*N_dim; i++)
  {
    Q[i] = 0.0;
  }
  for (int i = 0; i < N_dim*N_dim; i+=(N_dim+1))
  {
    Q[i] = 1.0;
  }

  for (int i = 0; i < N_dim*N_dim; i++)
  {
    R[i] = A[i];
  }

  std::complex<double> v[N_dim];
  std::complex<double> T[N_dim*N_dim];

  for (int j = 0; j < N_dim-1; j++)
  {
    // Prepare the Householder vector 'v'.
    // The vector will contain the factor of 2 already.
    for (int i = 0; i < j; i++)
    {
      v[i] = 0.0;
    }

    double norm_accumulate = 0.0;
    double norm_scale = std::abs(R[j*N_dim+j]);
    std::complex<double> scale_phase;
    if (norm_scale > 0.0)
    {
      scale_phase = R[j*N_dim+j]/norm_scale;
    }
    else
    {
      scale_phase = 1.0; // avoid issues if the top site is 0.
    }

    for (int i = j; i < N_dim; i++)
    {
      v[i] = R[i*N_dim + j];
      norm_accumulate += std::real(std::conj(R[i*N_dim+j])*R[i*N_dim+j]);
    }

    v[j] -= scale_phase*std::sqrt(norm_accumulate);

    norm_accumulate += (-norm_scale*norm_scale + std::real(std::conj(v[j])*v[j]));

    // Safety if norm is zero.
    if (norm_accumulate > 0.0)
    {
      double sqrt_2_inv_norm = sqrt(2.0/norm_accumulate);
      for (int i = j; i < N_dim; i++)
      {
        v[i] *= sqrt_2_inv_norm;
      }
    }
    // Good, we've got the Householder vector.

    // Update R.
    for (int l = 0; l < N_dim; l++)
    {
      T[l] = 0.0;
      for (int k = j; k < N_dim; k++)
      {
        T[l] += std::conj(v[k])*R[k*N_dim + l];
      }
    }

    for (int i = 0; i < N_dim; i++)
    {
      for (int l = j; l < N_dim; l++)
      {
        R[i*N_dim + l] -= v[i]*T[l];
      }
    }
    
    // Fix portions of R to zero that should be zero.
    for (int i = j+1; i < N_dim; i++)
    {
      R[i*N_dim + j] = 0.0;
    }


    // Update Q.
    for (int i = 0; i < N_dim; i++)
    {
      T[i] = 0.0;
      for (int l = 0; l < N_dim; l++)
      {
        T[i] += Q[i*N_dim + l]*v[l];
      }
    }

    for (int i = 0; i < N_dim; i++)
    {
      for (int k = j; k < N_dim; k++)
      {
        Q[i*N_dim + k] -= T[i]*std::conj(v[k]);
      }
    }

  }
}

// Compute the inverse of a matrix, A^{-1}, given a QR decomposition of A.
// Arg 1: [out] Solution A^{-1}. Assumed to be pre-allocated of size N_dim*N_dim.
// Arg 2: [in] Matrix Q. Assumed to be pre-allocated of size N_dim*N_dim,
//             and the result of the QR decomposition of A.
// Arg 3: [in] Matrix R. Assumed to be pre-allocated of size N_dim*N_dim,
//             and the result of the QR decomposition of A.
// Arg 4: [in] Dimension of the matrix.
void matrix_invert_qr(std::complex<double>* Ainv, std::complex<double>* Q, std::complex<double>* R, int N_dim)
{
  int i, j, n;
  std::complex<double> Qdagger[N_dim*N_dim];

  // We need to compute R^{-1} Q^\dagger

  // Step 1: Populate Qdagger with... Q^\dagger
  for (i = 0; i < N_dim; i++)
    for (j = 0; j < N_dim; j++)
      Qdagger[j*N_dim+i] = std::conj(Q[i*N_dim+j]);

  // Step 2: Back-substitute R.
  // We want to reuse R as much as possible.
  for (n = N_dim-1; n >= 0; n--)
  {
    // All columns of Ainv
    for (i = 0; i < N_dim; i++)
    {
      Ainv[n*N_dim+i] = Qdagger[n*N_dim+i];
      for (j = n+1; j < N_dim; j++)
      {
        Ainv[n*N_dim+i] -= R[n*N_dim+j]*Ainv[j*N_dim+i];
      }
      Ainv[n*N_dim+i] /= R[n*N_dim+n];
    }
  }
}

std::complex<double> matrix_det_qr(std::complex<double>* Q, std::complex<double>* R, int N_dim)
{
  std::complex<double> det = 1.0;
  int i;
  for (i = 0; i < N_dim; i++)
    det *= R[i*(N_dim+1)];

  return det;
}
