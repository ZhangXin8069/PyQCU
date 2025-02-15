// Copyright (c) 2018 Evan S Weinberg
// Thick restarted and deflated Lanczos.
// Points out where some type of polynomial
// acceleration could go in.
// Based on arXiv:1512:08135

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <sstream>
#include <complex>
#include <random>
#include <vector>

// Borrow dense matrix eigenvalue routines.
#include <Eigen/Dense>

#include "blas/generic_vector.h"

#include "square_laplace.h"

using namespace std; 
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, ColMajor> dMatrix;
typedef Matrix<std::complex<double>, Dynamic, Dynamic, ColMajor> cMatrix;

#ifndef QLINALG_FCN_POINTER
#define QLINALG_FCN_POINTER
typedef void (*matrix_op_real)(double*,double*,void*);
typedef void (*matrix_op_cplx)(complex<double>*,complex<double>*,void*);
#endif

// Operator
template <typename T>
class Operator {
protected:
  int length;

public:
  Operator(int length)
   : length(length)
  {;}

  virtual void operator()(T* out, T* in) = 0;

  int get_length()
  {
    return length;
  }

  static void operator_apply(T* out, T* in, void* data)
  {
    Operator* op = dynamic_cast<Operator*>(data);
    (*op)(out,in);
  }
};

// Function pointer wrapper
template <typename T>
class FunctionWrapper : public Operator<T> {

protected:
  matrix_op_cplx fcn;
  void* data;

public:
  FunctionWrapper(matrix_op_cplx fcn, void* data, int length)
   : Operator<T>(length), fcn(fcn), data(data)
  { ; }

  void operator()(T* out, T* in)
  {
    // Default: identity
    (*fcn)(out, in, data);
  }
};

// Linear map from [a,b] to [-1,1]
template <typename T, typename U>
class LinearMapToUnit : public Operator<T> {
protected:
  using Operator<T>::length;

  Operator<T>* op;
  const U a,b;
  T* tmp;

  U slope, shift;

public:
  LinearMapToUnit(Operator<T>* op, const U a, const U b)
   : Operator<T>(op->get_length()), op(op), a(a), b(b)
  {
    tmp = allocate_vector<T>(length);
    U c = 0.5*(a+b);
    U d = 0.5*(b-a);

    slope = 1.0/d;
    shift = -c*slope;
  }

  ~LinearMapToUnit()
  {
    deallocate_vector(&tmp);
  }

  void operator()(T* out, T* in)
  {
    zero_vector(tmp, length);
    op(tmp, in);
    caxpbyz(slope, tmp, shift, in, out, length);
  }
};

// Apply the series of 1/(1+x)
template <typename T>
class OneOverOnePlusX : public Operator<T> {
protected:
  using Operator<T>::length;

  Operator<T>* op;
  int order;
  T* tmp;
  T* tmp2;

public:
  OneOverOnePlusX(Operator<T>* op, int order)
   : Operator<T>(op->get_length()), op(op), order(order)
  {
    tmp = allocate_vector<T>(length);
    tmp2 = allocate_vector<T>(length);
  }

  ~OneOverOnePlusX()
  {
    deallocate_vector(&tmp);
    deallocate_vector(&tmp2);
  }

  void operator()(T* out, T* in)
  {
    // Apply the series for 1/(1+x) to whatever order
    copy_vector(out, in, length);
    copy_vector(tmp, in, length);

    for (int i = 0; i < order; i++)
    {
      // Apply the operator
      (*op)(tmp2, tmp);

      // accumulate
      caxpy((i%2 == 0) ? -1.0 : 1.0, tmp2, out, length);

      std::cout << "Step " << i << " tmp: " << tmp[0] << " tmp2: " << tmp2[0] << " Val: " << out[0] << "\n";

      // pointer swap
      std::swap(tmp, tmp2);


    }

  }
};

// Simple test
void identity_function(complex<double>* out, complex<double>* in, void* data)
{
  out[0] = in[0];
}

void one_half_function(complex<double>* out, complex<double>* in, void* data)
{
  out[0] = 0.5*in[0];
}

int main(int argc, char** argv)
{  
  complex<double> *rhs_cplx;

  // Set output precision to be long.
  cout << setprecision(10);

  // RNG related things.
  std::mt19937 generator (1337u); // RNG, 1337u is the seed. 

  ///////////////////////////////////
  // Properties of Linear operator //
  ///////////////////////////////////
  double inv_variance = 6.0; // inverse of variance for gaussian non-compact U(1) links.

  // Basic information about the lattice.
  int length = 8;
  double m_sq = 0.001;
  
  // Some start-up.
  int volume = length*length;
  
  // Create a random compact U(1) link.
  complex<double>* gauge_links = allocate_vector<complex<double>>(2*length*length);
  gaussian_real(gauge_links, 2*length*length, generator, 1.0/inv_variance);
  polar(gauge_links, 2*length*length);
  
  // Vectors.
  rhs_cplx = allocate_vector<complex<double>>(volume);

  // Zero out the vector.
  zero_vector(rhs_cplx, length*length);

  // Structure which gets passed to the function.
  laplace_gauged_struct lapstr_gauged;
  lapstr_gauged.length = length;
  lapstr_gauged.m_sq = m_sq;
  lapstr_gauged.gauge_links = gauge_links; 

  // Uncomment this to get the free field.
  //constant_vector(gauge_links, 1.0, 2*volume);


  FunctionWrapper<complex<double>> fcn(identity_function, nullptr, 1);
  complex<double> in = 2.0;
  complex<double> out = 0.0;
  fcn(&out, &in);

  std::cout << "Val: " << in << " Fcn: " << out << "\n";

  FunctionWrapper<complex<double>> ohalf(one_half_function, nullptr, 1);
  OneOverOnePlusX<complex<double>> rational(&ohalf, 2); 
  complex<double> in1 = 1.0;
  complex<double> out1 = 0.0;
  rational(&out1, &in1);
  std::cout << "Val: " << in1 << " Series: " << out1 << " Exact: " << 1.0/(1.0+0.5) << "\n";
  return 0;

  //////////////////////////////////////
  // Properties for Lanczos algorithm //
  //////////////////////////////////////

  int max_iterations = 20; // maximum number of restarts
  int max_no_TR_candidates = 2; // maximum number of times we press on when there's
                                // no TR candidate
  int max_no_TM_eig_candidates = 2; // maximum number of times we press on when there's
                                    // no candidates in TM
  int m = 30; // Max subspace size before restart
  double tol = 1e-8; // Tolerance of eigenvalue computation

  // We're looking to grab all eigenvalues in the interval below
  // If we switched to a polynomial acceleration scheme, we'd look
  // for all eigenvalues above some value.
  // This is a choice! I'll make it clear below where to modify
  // this stopping condition.
  double eig_ceil = 2.0;
  double eig_floor = 1e-6; // Avoid deflated modes


  //////////////////////////////////////////
  // Prepare memory for Lanczos algorithm //
  //////////////////////////////////////////

  // Allocate Lanczos space. Extra vector needed for restarts.
  complex<double>** Q = new complex<double>*[m+1];
  for (int i = 0; i < m+1; i++) {
    Q[i] = allocate_vector<complex<double>>(volume);
  }

  // Extra beta for restarts
  double beta_restart = 0.0;

  // Storage for locked vectors!
  std::vector<complex<double>> lambda;
  std::vector<complex<double>*> U;

  // Intermediate vectors
  complex<double>* w = allocate_vector<complex<double>>(volume);
  complex<double>* lambda_x_u = allocate_vector<complex<double>>(volume);

  // Allocate T_m matrix (tridiagonal in un-restarted case).
  // Can generally be complex because of the restart
  cMatrix Tm = cMatrix::Zero(m,m);

  // Prepare an eigensolver
  SelfAdjointEigenSolver<cMatrix> eigsolve_Tm(m);

  ////////////////////////
  // Begin the Lanczos! //
  ////////////////////////

  // Prepare a random unit vector for Q[0]!
  gaussian(Q[0], volume, generator);
  normalize(Q[0], volume);

  int iters = 0;
  int l = 0;
  int locked = 0;
  int num_no_TR_candidates = 0;
  int num_no_TM_eig_candidates = 0;

  // Let's gooooo
  while (iters < max_iterations) {
    // last bit of thick restart. put s in correct place.
    if (l > 0) {
      for (int i = 0; i < l; i++) {
        Tm(l,i) = Tm(m-1,i);
        Tm(m-1,i) = 0.0;
        Tm(i,l) = conj(Tm(l,i));
      }
    }

    // If we're not on the first iter, T_m has values filled in.
    // Fill out rest of T_m. Standard lanczos!
    for (int i = l; i < m; i++) {
      zero_vector(w, volume);

      // PUT IN POLY ACCELERATION HERE
      square_laplacian_gauged(w, Q[i], &lapstr_gauged); // w = B q_i
      // deflate
      for (int j = 0; j < locked; j++) {
        complex<double> dotprod = dot(U[j], w, volume);
        caxpy(-dotprod, U[j], w, volume);
      }

      if (i > l) {
        caxpy(-Tm(i-1,i), Q[i-1], w, volume); // w -= beta_i q_{i-1}
      }

      // Compute alpha_i
      Tm(i,i) = re_dot(Q[i], w, volume);

      // w -= alpha_i q_i
      caxpy(-Tm(i,i), Q[i], w, volume);

      // Reorthogonalize
      for (int j = 0; j < i; j++) {
        orthogonal(w, Q[j], volume);
      }

      // Compute beta_{i+1}
      double beta = sqrt(norm2sq(w,volume));

      if (i == m-1) {
        beta_restart = beta;
      } else {
        Tm(i+1,i) = Tm(i,i+1) = beta;
      }

      // Check for breakdown
      if (beta < 1e-10) {
        gaussian(Q[i+1], volume, generator);
        normalize(Q[i+1], volume);
      } else {
        caxy(1.0/beta, w, Q[i+1], volume);
      }   
    }

    iters++;

    // Calculate the Ritz values
    eigsolve_Tm.compute(Tm);
    std::cout << "The Ritz values are:\n" << eigsolve_Tm.eigenvalues().transpose() << "\n";
    std::cout << std::flush;

    // Compute candidate pairs---evals between the min and max.
    // There's lots of potential for memory reuse here...
    std::vector<complex<double>> theta_candidate;
    std::vector<complex<double>> s_candidate; // used for restart
    std::vector<complex<double>*> u_candidate;

    for (int i = 0; i < m; i++) {
      double eval = eigsolve_Tm.eigenvalues()(i);
      if (eval > eig_floor && eval < eig_ceil) {
        // Add it as a candidate pair
        theta_candidate.push_back(eval);
        s_candidate.push_back(beta_restart*eigsolve_Tm.eigenvectors().col(i)(m-1)); 

        // Build up the Ritz vector
        u_candidate.push_back(allocate_vector<complex<double>>(volume));
        zero_vector(u_candidate[u_candidate.size()-1], volume);
        for (int j = 0; j < m; j++) {
          caxpy(eigsolve_Tm.eigenvectors().col(i)(j), Q[j], u_candidate[u_candidate.size()-1], volume);
        }
      }
    }

    // Zero out all Q except the last one!
    for (int i = 0; i < m; i++) {
      zero_vector(Q[i], volume);
    }

    int num_candidate = theta_candidate.size();

    // if no candidates were found, quit.
    // Might be a sign we've found all the eigenvalues... or not.
    // Needs a smarter heuristic. Again, should be safer
    // with poly accel.
    if (num_candidate == 0) {

      num_no_TM_eig_candidates++;
      std::cout << "No TM eigenvalue candidates: " << num_no_TM_eig_candidates << ".\n";
      if (num_no_TM_eig_candidates == max_no_TM_eig_candidates) {
        std::cout << "Max number of times with no candidate TM eigenvalues reached.\n";
        break;
      } else {
        l = 0;
        Tm = cMatrix::Zero(m,m);
        copy_vector(Q[0], Q[m], volume);
        continue;
      }
    }

    // Reset l
    l = 0;


    // Zero out Tm, start rebuilding
    Tm = cMatrix::Zero(m,m);

    for (int i = 0; i < num_candidate; i++) {
      // Compute Rayleigh value. Technically wasteful
      // because we aren't doing poly accel.
      // Reuse w.
      zero_vector(w, volume);
      square_laplacian_gauged(w, u_candidate[i], &lapstr_gauged);

      double rayleigh = re_dot(u_candidate[i], w, volume);

      // Make sure we're in the target range
      if (rayleigh > eig_ceil || rayleigh < eig_floor) { continue; }

      // check convergence
      caxy(rayleigh, u_candidate[i], lambda_x_u, volume);
      if (sqrt(diffnorm2sq(w, lambda_x_u, volume))/rayleigh < tol) {
        // We've got a lock!
        lambda.push_back(rayleigh);
        U.push_back(allocate_vector<complex<double>>(volume));
        copy_vector(U[locked], u_candidate[i], volume);
        normalize(U[locked], volume);
        locked++;
      } else {
        // Add it to the thick restart set
        copy_vector(Q[l], u_candidate[i], volume);
        Tm(l,l) = eigsolve_Tm.eigenvalues()(i);

        // Store s somewhere safe
        Tm(m-1,l) = s_candidate[i];

        l++;
      }
    }

    // Clean up candidates.
    for (int i = 0; i < num_candidate; i++) {
      deallocate_vector(&u_candidate[i]);
    }

    // If there are no vectors in the thick restart set, quit.
    if (l == 0) {
      num_no_TR_candidates++;
      std::cout << "No thick restart vectors: " << num_no_TR_candidates << ".\n";
      if (num_no_TR_candidates == max_no_TR_candidates) {
        std::cout << "Max number of times with no thick restart vectors reached.\n";
        break;
      }
    }

    // Put last Q in place.
    copy_vector(Q[l], Q[m], volume);

  }

  if (iters == max_iterations) {
    std::cout << "Hit maximum iterations.\n" << "\n";
  }

  // Uh... I guess we have eigenvalues.
  std::cout << "Computed eigenvalues.\n";
  for (unsigned int i = 0; i < lambda.size(); i++) {
    std::cout << lambda[i] << "\n";
  }
  std::cout << "\n";



  /**/

  if (volume <= 256) {

    // Let's get the eigenvalues of the full operator!

    // Allocate a sufficiently gigantic matrix.
    cMatrix mat_cplx = cMatrix::Zero(volume, volume);

    // Form matrix elements. This is where it's important that
    // dMatrix and cMatrix are column major.
    // I should probably make this safer by using a "Map".
    for (int i = 0; i < volume; i++)
    {
      // Set a point on the rhs for a matrix element.
      zero_vector(rhs_cplx, volume);
      rhs_cplx[i] = 1.0;

      // Where we put the result of the matrix element.
      complex<double>* mptr = &(mat_cplx(i*volume));

      square_laplacian_gauged(mptr, rhs_cplx, &lapstr_gauged);
    }

    // Get the eigenvalues.
    SelfAdjointEigenSolver<cMatrix> eigsolve_cplx(volume);
    eigsolve_cplx.compute(mat_cplx);

    std::cout << "The eigenvalues are:\n" << eigsolve_cplx.eigenvalues() << "\n";

    ////////////////////////////////
    // COMPARE LOWEST EIGENVECTOR //
    ////////////////////////////////

    std::cout << "Compare results:\n\n";
    std::cout << "Lanczos Exact Ratio\n";
    for (int i = 0; i < volume; i++) {
      std::cout << U[0][i] << " " << eigsolve_cplx.eigenvectors().col(0)(i)
                << " " << U[0][i]/eigsolve_cplx.eigenvectors().col(0)(i) << "\n";
    }
  }

  //////////////
  // CLEAN UP //
  //////////////

  deallocate_vector(&w);
  deallocate_vector(&lambda_x_u);

  for (int i = 0; i <= m; i++) {
    deallocate_vector(&Q[i]);
  }
  delete[] Q;

  deallocate_vector(&rhs_cplx);
  deallocate_vector(&gauge_links);
  return 0;
}


