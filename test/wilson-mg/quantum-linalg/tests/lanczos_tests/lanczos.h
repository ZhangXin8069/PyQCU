// Copyright (c) 2018 Evan S Weinberg
// Simple Lanczos (no restarting, deflating, etc)
// for complex values. Easy to generalize to real,
// my trait-foo just isn't strong at the moment.
// Based on arXiv:1512.08135.

#ifndef QLINALG_LANCZOS
#define QLINALG_LANCZOS

#include <complex>
#include <random>
#include <Eigen/Dense>

#include "blas/generic_vector.h"
#include "operator.h"

template <typename T>
class SimpleComplexLanczos
{
private:

  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> realMatrix;

  Operator<std::complex<T>>* op; // Linear operator
  int m; // size of Krylov space
  std::mt19937& generator; // rng

  // length of vector
  int length;

  // Tridiagonal matrix T_m
  realMatrix Tm;

  // Internal eigensolver for T_m
  Eigen::SelfAdjointEigenSolver<realMatrix> eigsolve_Tm;

  // Did we compute eigenvalues?
  bool done_compute;

  std::complex<T>* w; // temporary vector
  std::complex<T>** Q; // Built up storage space

public:
  SimpleComplexLanczos(Operator<std::complex<T>>* op, int m, std::mt19937& generator)
   : op(op), m(m), generator(generator),
     length(op->get_length()), Tm(realMatrix::Zero(m,m)), eigsolve_Tm(m),
     done_compute(false)
  {
    w = allocate_vector<std::complex<T>>(length);

    Q = new std::complex<T>*[m];
    for (int i = 0; i < m; i++) {
      Q[i] = allocate_vector<std::complex<T>>(length);
    }
  }

  ~SimpleComplexLanczos()
  {
    deallocate_vector(&w);
    for (int i = 0; i < m; i++) {
      deallocate_vector(&Q[i]);
    }
    delete[] Q;
  }

  // Compute Ritz values
  void compute()
  {
    // If we're doing a recalculation, zero out Tm.
    if (done_compute) {
      Tm = realMatrix::Zero(m,m);
    }

    // Prepare a random starting vector.
    // We could let the user pass this in---
    // it's a way to pass in an initial guess for
    // _some_ eigenvector.
    gaussian(Q[0], length, generator);
    normalize(Q[0], length);

    // Let's goooo
    for (int i = 0; i < m; i++) {
      zero_vector(w, length);

      (*op)(w, Q[i]); // w = B q_i

      if (i > 0) {
        caxpy(-Tm(i-1,i), Q[i-1], w, length); // w -= beta_i q_{i-1}
      }

      // Compute alpha_i
      Tm(i,i) = re_dot(Q[i], w, length);

      // Break here if we're on the last step.
      if (i == (m-1)) { break; }

      // w -= alpha_i q_i
      caxpy(-Tm(i,i), Q[i], w, length);

      // Reorthogonalize
      for (int j = 0; j < i; j++) {
        orthogonal(w, Q[j], length);
      }

      // Compute beta_{i+1}
      Tm(i+1,i) = Tm(i,i+1) = sqrt(norm2sq(w, length));

      // Check for breakdown
      if (fabs(Tm(i,i+1)) < 1e-10) {
        gaussian(Q[i+1], length, generator);
        normalize(Q[i+1], length);
      } else {
        caxy(1.0/Tm(i+1,i), w, Q[i+1], length);
      }
    }

    // Compute the Ritz values
    eigsolve_Tm.compute(Tm);

    // We've finished the calculation!
    done_compute = true;
  }

  // Get the Ritz values. Assumes eigs has been allocated.
  bool ritzvalues(T* eigs)
  {
    if (!done_compute) { return false; }

    for (int i = 0; i < m; i++) {
      eigs[i] = eigsolve_Tm.eigenvalues()(i);
    }

    return true;
  }

  // Get a ritzvector. Requires O(length*m) compute.
  bool ritzvector(int eig_num, std::complex<T>* evec)
  {
    if (!done_compute) { return false; }
    if (eig_num < 0 || eig_num >= m) { return false; }

    for (int i = 0; i < m; i++)
    {
      // eigenvectors live in columns of eigenvector matrix.
      // Access a single one with eigsolve_Tm.eigenvectors().col(eig_num)
      caxpy(eigsolve_Tm.eigenvectors()(i,eig_num), Q[i], evec, length);
    }

    return true;
  }

  // Get the eigenvectors. Requires O(length*m^2) compute.
  // Assumes eigenvectors has been allocated. Fast index -> element,
  // slow index -> which eigenvector.
  bool ritzvectors(std::complex<T>** evec)
  {
    if (!done_compute) { return false; }

    // There's plenty of memory reuse to exploit
    // with block-BLAS routines because each Ritz vector
    // is a linear combination of all Q[i]s. 
    for (int i = 0; i < m; i++) {
      ritzvector(i, evec[i]);
    }

    return true;
  }
};

#endif // QLINALG_LANCZOS
