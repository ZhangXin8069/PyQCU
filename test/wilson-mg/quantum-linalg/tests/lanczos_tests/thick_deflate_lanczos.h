// Copyright (c) 2018 Evan S Weinberg
// Simple Lanczos with thick restarts
// and deflation for Hermitian complex matrices.
// Based on arXiv:1512.08135.

#ifndef QLINALG_LANCZOS_TR
#define QLINALG_LANCZOS_TR

#include <complex>
#include <random>
#include <map>
#include <functional>
#include <algorithm>
#include <Eigen/Dense>

#include "blas/generic_vector.h"
#include "operator.h"

struct TRCLStruct
{
  int n_ev; // desired number of eigenvalues

  int m; // Subspace size at restart
  double tol; // Tolerance at which an eigenvalue is locked

  int max_restarts; // maximum number of restarts
  int preserved_space; // size of space to preserve after a restart
                       // set to -1 to default to m/4+1

  bool deflate; // add deflation of locked eigenvalues, only 
                // really good with polynomial acceleration

  std::mt19937* generator; // generator

  bool verbose = false; // verbosity
};

template <typename T, typename Comparitor>
class ThickRestartComplexLanczos
{
protected:


  typedef Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> cplxMatrix;

  Operator<std::complex<T>>* op; // Linear operator

  int n_ev; // desired number of eigenvalues
  int m; // subspace size
  double tol; // lock tolerance
  int max_restarts; // max number of restarts
  int preserved_space; // size of space to preserve after a restart
                       // set to -1 to default to m/4+1
  bool deflate; // add deflation of locked eigenvalues, only 
                // really good with polynomial acceleration

  std::mt19937& generator; // rng

  bool verbose;

  // length of vector
  int length;

  // Tridiagonal matrix T_m
  cplxMatrix Tm;

  // Internal eigensolver for T_m
  Eigen::SelfAdjointEigenSolver<cplxMatrix> eigsolve_Tm;

  // Did we compute eigenvalues?
  bool done_compute;

  std::complex<T>* w; // temporary vector
  std::complex<T>** Q; // Built up storage space

  // Locked eigenvalues and eigenvectors
  std::map<T,complex<T>*,Comparitor> locked_eigs;

public:

  ThickRestartComplexLanczos(Operator<std::complex<T>>* op, TRCLStruct& tr_struct)
   : op(op),
     n_ev(tr_struct.n_ev),
     m(tr_struct.m), tol(tr_struct.tol), max_restarts(tr_struct.max_restarts),
     preserved_space(tr_struct.preserved_space <= 0 ? m/4+1 : tr_struct.preserved_space),
     deflate(tr_struct.deflate),
     generator(*tr_struct.generator), verbose(tr_struct.verbose), 
     length(op->get_length()), Tm(cplxMatrix::Zero(m,m)), eigsolve_Tm(m),
     done_compute(false),
     w(nullptr), Q(nullptr)
  {
    w = allocate_vector<std::complex<T>>(length);

    Q = new std::complex<T>*[m+1]; // +1 for the restart
    for (int i = 0; i < m+1; i++) {
      Q[i] = allocate_vector<std::complex<T>>(length);
    }
  }

  ~ThickRestartComplexLanczos()
  {
    deallocate_vector(&w);
    for (int i = 0; i < m+1; i++) {
      deallocate_vector(&Q[i]);
    }
    delete[] Q;

    // deallocate all vectors
    std::for_each(locked_eigs.begin(), locked_eigs.end(),
      [](const std::pair<T,std::complex<T>*>& eig_struct) { if (eig_struct.second != nullptr) deallocate_vector(&eig_struct.second); } );
    //for (auto it = locked_eigs.begin(); it != locked_eigs.end(); ++it) {
    //  deallocate_vector(&it->second);
    //}
  }

  // Compute Ritz values
  void compute()
  {
    // If we're doing a recalculation, zero out Tm,
    // zero out lambda, zero out U
    if (done_compute) {
      Tm = cplxMatrix::Zero(m,m);

      // deallocate all vectors
      for (auto it = locked_eigs.begin(); it != locked_eigs.end(); ++it) {
        deallocate_vector(&it->second);
      }
      locked_eigs.clear();
    }

    // Prepare a random starting vector.
    // We could let the user pass this in---
    // it's a way to pass in an initial guess for
    // _some_ eigenvector.
    gaussian(Q[0], length, generator);
    normalize(Q[0], length);

    // number of vectors carried over
    int l = 0;

    // Extra beta for restarts
    double beta_restart = 0.0;

    // Let's goooo
    int iters = 0;
    bool converged = false;
    while (iters < max_restarts && !converged) {

      if (verbose) std::cout << "Iteration " << iters << (verbose ? "\n" : ", ") << std::flush;

      // last bit of thick restart. put s in correct place.
      if (l > 0) {
        for (int i = 0; i < l; i++) {
          Tm(l,i) = Tm(m-1,i);
          Tm(m-1,i) = 0.0;
          Tm(i,l) = conj(Tm(l,i));
        }
      }

      // If we're not on the first iter, T_m has values filled in.
      // Fill out the rest of T_m. Standard lanczos!

      for (int i = l; i < m; i++) {

        //std::cout << "Original Lanczos i = " << i << "\n" << std::flush;
        zero_vector(w, length);

        // poly acceleration would go here
        (*op)(w, Q[i]); // w = B q_i

        // Optionally deflate
        if (deflate) {
          std::for_each(locked_eigs.begin(), locked_eigs.end(), 
            [this](const std::pair<T,std::complex<T>*>& eig_struct) {
              orthogonal(w, eig_struct.second, length);
            });
          /*for (auto it = locked_eigs.begin(); it != locked_eigs.end(); ++it) {
            complex<T> dotprod = dot(it->second, w, length);
            caxpy(-dotprod, it->second, w, length);
          }*/
        }

        if (i > l) {
          caxpy(-Tm(i-1,i), Q[i-1], w, length); // w -= beta_i q_{i-1}
        }

        // Compute alpha_i
        Tm(i,i) = re_dot(Q[i], w, length);

        // w -= alpha_i q_i
        caxpy(-Tm(i,i), Q[i], w, length);

        // Reorthogonalize. Should be replaced with a partial reortho
        for (int j = 0; j < i; j++) {
          orthogonal(w, Q[j], length);
        }
        for (auto it = locked_eigs.begin(); it != locked_eigs.end(); ++it) {
          orthogonal(w, it->second, length);
        }

        // Compute beta_{i+1}
        double beta = sqrt(norm2sq(w, length));

        if (i == m-1) {
          beta_restart = beta;
        } else {
          Tm(i+1,i) = Tm(i,i+1) = beta;
        }

        // Check for breakdown
        if (beta < 1e-10) {
          gaussian(Q[i+1], length, generator);
          normalize(Q[i+1], length);
        } else {
          caxy(1.0/beta, w, Q[i+1], length);
        }
      }

      iters++;

      // Calculate the Ritz values
      eigsolve_Tm.compute(Tm);

      // debug...
      if (verbose) {
        std::cout << "The Ritz values are:\n" << eigsolve_Tm.eigenvalues().transpose() << "\n";
        std::cout << std::flush;
      }

      l = 0;

      // Custom eigenstructure.
      // s_value is the `s` value needed for the restart
      // u_vector is the Ritz vector
      struct Eigenset {
        std::complex<T> s_value;
        std::complex<T>* u_vector;
      };

      // first element: Ritz value
      // second element: s value (for restart), ritz vector, prev Ritz value
      std::map<T,Eigenset> candidate; 

      // HACK for now. Figure out if the comparitor is a less
      // or greater.
      Comparitor test;
      bool less_than = true;
      if (test(3,2)) { less_than = false; }

      // For the safety check (looking for missing eigenvalues)
      T safe_eig = 0.0;
      T safe_found = false;

      int local_preserved_space = preserved_space;
      for (int i = 0; i < local_preserved_space; i++) {
        // Ritzvalues are sorted smallest to largest, use this
        // to reverse the order
        int i_idx = less_than ? i : (m - i - 1);

        // Query the eigenvalue
        T eval = eigsolve_Tm.eigenvalues()[i_idx];

        // Figure out if the Ritz value is a good candidate.
        // Add it as a candidate unless its magnitude is
        // less than the tolerance when we're deflating
        //  (need a better condition)
        if (!deflate || fabs(eval) > tol) {

          // Try to catch cases where there's
          // an eigenvalue in between found Ritz values
          if (!safe_found) { safe_found = true; safe_eig = eval; }

          Eigenset eset;
          eset.s_value = beta_restart*eigsolve_Tm.eigenvectors().col(i_idx)(m-1);
          eset.u_vector = allocate_vector<complex<T>>(length);
          zero_vector(eset.u_vector, length);
          for (int j = 0; j < m; j++) {
            caxpy(eigsolve_Tm.eigenvectors().col(i_idx)(j), Q[j], eset.u_vector, length);
          }

          candidate.insert(
            std::make_pair(eval,
              eset));
        } else {
          if (local_preserved_space < m) { local_preserved_space++; continue; } // give some more room to look
        }
        
      }

      // Zero out all Q except the last one
      for (int i = 0; i < m; i++) {
        zero_vector(Q[i], length);
      }

      // Zero out Tm, start rebuilding
      Tm = cplxMatrix::Zero(m,m);

      for (auto it = candidate.begin(); it != candidate.end(); ++it) {


        // One way to check convergence: explicitly check |A v - lambda v|/lambda
        /*
        zero_vector(w, length);
        (*op)(w, it->second.u_vector); // w = B q_i
        double rayleigh = re_dot(it->second.u_vector, w, length);

        // check convergence. second condition sort of b.s., but it's a last ditch type thing
        cax(1.0/it->first, w, length);
        if (sqrt(diffnorm2sq(w, it->second.u_vector,length)) < tol ||*/
        // otherwise
        if (verbose) { std::cout << fabs(it->second.s_value/it->first) << "\n"; }
        if (fabs(it->second.s_value/it->first) < tol) {
          // We've got a lock!
          complex<double>* tmp = allocate_vector<complex<T>>(length);
          copy_vector(tmp, it->second.u_vector, length);
          normalize(tmp, length);
          locked_eigs.insert(std::make_pair(it->first, tmp)); // Rayleigh instead?

        } else {
          // Add to the thick restart set
          copy_vector(Q[l], it->second.u_vector, length);
          Tm(l,l) = it->first;

          // Store s somewhere safe.
          Tm(m-1,l) = it->second.s_value;

          l++;
        }
      }
      if (verbose) { std::cout << "Length: " << locked_eigs.size() << "\n\n"; }

      // Clean up candidates
      for (auto it = candidate.begin(); it != candidate.end(); ++it) {
        deallocate_vector(&it->second.u_vector);
      }

      // Put the last Q in place for the restart
      copy_vector(Q[l], Q[m], length);

      // See if we can exit.
      if (locked_eigs.size() >= static_cast<unsigned int>(n_ev)) {
        // If there's a Ritz value less than the largest locked vector/
        //                         greater than the smallest locked vector,
        // that means there's some eigenvalue between converged eigenvalues.
        // Delete the largest/smallest locked vector and continue
        if (safe_found) {
          Comparitor c;
          bool continue_flag = false;
          while (locked_eigs.size() >= static_cast<unsigned int>(n_ev) && c(safe_eig,locked_eigs.rbegin()->first)) {
            auto it = locked_eigs.end();
            it--;
            deallocate_vector(&it->second);
            locked_eigs.erase(it);
            continue_flag = true;
          }
          if (continue_flag && locked_eigs.size() == static_cast<unsigned int>(n_ev)) {
            continue_flag = false;
          }
          if (continue_flag) {
            if (verbose) { std::cout << " converged " << locked_eigs.size() << " eigenvalues.\n"; }
            continue;
          }
        }

        auto it = locked_eigs.begin();
        for (int i = 0; i < n_ev; i++) { it++; }
        auto it2 = it;
        for ( ; it != locked_eigs.end(); it++)
        {
          deallocate_vector(&it->second);
        }
        locked_eigs.erase(it2,locked_eigs.end());
        converged = true;
      }

      if (verbose) std::cout << " converged " << locked_eigs.size() << " eigenvalues.\n";

    }

    if (iters == max_restarts && verbose) {
      std::cout << "Hit maximum restarts.\n\n";
    }

    if (verbose) {
      // Uh... I guess we have eigenvalues.
      std::cout << "Computed eigenvalues.\n";
      for (auto it = locked_eigs.begin(); it != locked_eigs.end(); ++it) {
        std::cout << it->first << "\n";
      }
      std::cout << "\n";
    }

    std::cout << "Number of iterations: " << iters << "\n";

    // We've finished the calculation!
    done_compute = true;
  }

  // Get number of converged eigenvalues
  int num_converged()
  {
    if (!done_compute) { return -1; }

    return locked_eigs.size();
  }

  // Get the Ritz values. Assumes eigs has been allocated.
  bool ritzvalues(T* eigs)
  {
    if (!done_compute) { return false; }

    int i = 0;
    for (auto it = locked_eigs.begin(); it != locked_eigs.end(); ++it) {
      eigs[i++] = it->first;
    }

    return true;
  }

  // Get a ritzvector. 
  bool ritzvector(int eig_num, std::complex<T>* evec)
  {
    if (!done_compute) { return false; }
    int n_locked = locked_eigs.size();
    if (eig_num < 0 || eig_num >= n_locked) { return false; }

    auto it = locked_eigs.begin();
    for (int i = 0; i < eig_num-1; i++, it++) { ; }

    copy_vector(evec, it->second, length);

    return true;
  }

  // Get the eigenvectors. Requires O(length*m^2) compute.
  // Assumes eigenvectors has been allocated. Fast index -> element,
  // slow index -> which eigenvector.
  bool ritzvectors(std::complex<T>** evec)
  {
    if (!done_compute) { return false; }

    int i = 0;
    for (auto it = locked_eigs.begin(); it != locked_eigs.end(); ++it) {
      copy_vector(evec[i++], it->second, length);
    }

    return true;
  }
};

#endif // QLINALG_LANCZOS_TR
