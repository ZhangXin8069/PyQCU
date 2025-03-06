// Copyright (c) 2017 Evan S Weinberg
// Interface for finding the eigenvalues of a generic
// unstructured complex matrix using ARPACK.
// Based on a reference interface from Alexei Strelchenko
// Hoping to extend this to different precisions,
// real matrices as well as complex matrices, unstructured
// matrices.
// Reference to ARPACK functions:
// ZNAUPD: https://github.com/pv/arpack-ng/blob/master/SRC/znaupd.f
// ZNEUPD: https://github.com/pv/arpack-ng/blob/master/SRC/zneupd.f

#ifndef QLINALG_INTERFACE_ARPACK
#define QLINALG_INTERFACE_ARPACK

#include <iostream>
#include <complex>
#include <cmath>
#include <string.h>

#include "../../blas/generic_vector.h"

extern "C" {

  // Convenient define to map to FORTRAN calling conventions
  #define ARPACK(s) s ##   _

  // Iterative callback functions. These functions chug along
  // for a while, and return when you need to apply
  // a matrix-vector operation. These will get wrapped
  // in an interface quickly. 

  // double precision complex
  extern int ARPACK(znaupd) (int *ido, char *bmat, int *n, char *which,
                        int *nev, double *tol, complex<double> *resid,
                        int *ncv, complex<double> *v, int *ldv, 
                        int *iparam, int *ipntr, complex<double> *workd,
                        complex<double> *workl, int *lworkl, double *rwork,
                        int *info);

  // double precision complex
  extern int ARPACK(zneupd) (int *comp_evecs, char *howmany, int *select, complex<double> *evals, 
                        complex<double> *v, int *ldv, complex<double> *sigma, complex<double> *workev, 
                        char *bmat, int *n, char *which, int *nev, double *tol, complex<double> *resid, 
                        int *ncv, complex<double> *v1, int *ldv1, int *iparam, int *ipntr, 
                        complex<double> *workd, complex<double> *workl, int *lworkl, double *rwork, int *info);



} // extern "C"

// Struct that contains information about the
// ARPACK eigensolve. Tries to match
// the "inversion_info" struct in
// /inverters/inversion_struct.h" as best
// as possible.
struct arpack_info
{
    bool success; // is non-zero if there's an error.
    int iter; // number of Arnoldi terations (IPARAM(3))
    int n_conv; // number of converged eigenvalues (IPARAM(5))
    int ops_count; // number of OP*x applications (IPARAM(9))
    int znaupd_code; // znaupd error code.

   // Default Constructor
  arpack_info() : success(false), iter(0), n_conv(0), ops_count(0), znaupd_code(0)
  { ; }

  // Copy constructor.
  arpack_info(const arpack_info &obj)
  {
    success = obj.success;
    iter = obj.iter;
    n_conv = obj.n_conv;
    ops_count = obj.ops_count;
    znaupd_code = obj.znaupd_code; 
  }

  // Assignment operator.
  arpack_info& operator=(const arpack_info &obj)
  {
    success = obj.success;
    iter = obj.iter;
    n_conv = obj.n_conv;
    ops_count = obj.ops_count;
    znaupd_code = obj.znaupd_code;
    return *this;
  }

  // For C++11, may need to add move constructors?
  // http://stackoverflow.com/questions/255612/dynamically-allocating-an-array-of-objects
};


/******************************
* Double Complex Unstructured *
******************************/

// double precision complex general struct.
class arpack_dcn
{
public:

  // Enum to specify what part of spectrum to get.
  enum arpack_spectrum_piece
  {
    ARPACK_NONE,
    ARPACK_LARGEST_MAGNITUDE,
    ARPACK_SMALLEST_MAGNITUDE,
    ARPACK_LARGEST_REAL,
    ARPACK_SMALLEST_REAL,
    ARPACK_LARGEST_IMAGINARY,
    ARPACK_SMALLEST_IMAGINARY,
  };

private:
  // Get rid of copy, assignment operator.
  arpack_dcn(arpack_dcn const &);
  arpack_dcn& operator=(arpack_dcn const &);

   // Scalar types which set memory allocation.
  int n; // imum possible dimension of the matrix.
           // This doesn't have to equal N, but you might for
           // alignment.
           
  int max_nev; // maximum number of eigenvalues you'll want.

  int max_ncv; // maximum number of basis vectors within IRAP.

  int ldv; // leading dimension of the eigenvectors in V.
            // This doesn't have to be the dimension of the
            // matrix, but it can be.

  /**************************************************************
  // Vectors and variables that get passed to ARPACK functions. *
  **************************************************************/
  std::complex<double>* v; // 2d array of size [ldv][ncv]. Holds eigenvectors.
  std::complex<double>* d; // 1d array of size [ncv]. Holds eigenvalues.
  std::complex<double>* workl; // 1d array of size 3*ncv**2
                                // + 5*ncv.
                                // Workspace. Size suggested by znsimp.f.
  std::complex<double>* workd; // 1d array of size 3*n.
                                // This space gets used in reverse feedback to
                                // hold rhs = A*lhs.
  std::complex<double>* resid; // 1d array of size n.
                                // This holds internal residual vectors.       
                                // This is normally zero at the start, but it
                                // can be set to non-zero to choose an initial
                                // guess vector.
  std::complex<double>* workev; // Working space. 1d array of size 2*ncv.
  double* rwork; // Working space. 1d array of size ncv.
  double* rd; // Working space. 2d array of size [ncv][3];
  int* select; // Should be a logical. 

  int iparam[11]; // Various parameters which get passed in.
  int ipntr[14]; // I believe ARPACK uses this to maintain its own state?

  char which[3]; // used to specify portion of spectrum desired.
  char bmat; // Used to specify that this is a regular, not generalized, eigenvalue problem.
  char hwmny; // ESW addition: what type of eigenvalue to get.
  int ido, lworkl, info, ierr, ishfts, mode1; 


  int rvec; // 

  /***************************************************************/

  // Are we allocated? 0 no, 1 yes.
  int is_allocated; 

  // Are we preparing for a solve
  // of the entire spectrum?
  bool entire_spectrum; 

  // Are the structures in a state to pull
  // eigenvalues, eigenvectors?
  // Holds the state of the prepared eigenvalues. 
  arpack_info info_solve; // only meaningful when access_ready == true.
  bool access_ready;
  int maxitr; // maximum number of iterations.
  double tol; // tolerance of solve.
  arpack_spectrum_piece spec_piece;  
  int ev;
  int cv; 

  // Matrix pointer and extra data.
  void (*matrix_vector)(complex<double>*,complex<double>*,void*);
  void* extra_info;

public:

  // Allocate memory for a certain matrix size, number of eigenvalues,
  // number of internal values. 
  arpack_dcn(int n, int maxitr, double tol, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, int nev, int ncv)
   : n(n), max_nev(nev), max_ncv(ncv), ldv(n),
     v(0), d(0), workl(0), workd(0), resid(0), workev(0),
     rwork(0), rd(0), select(0), ierr(0),
     is_allocated(false), entire_spectrum(false), 
     maxitr(maxitr), tol(tol), ev(0), cv(0), 
     matrix_vector(matrix_vector), extra_info(extra_info)
  {
    if (nev > n/2)
    {
      std::cout << "[ARPACK_ERROR]: \"arpack_dcn\" does not support allocating space for more than dim/2 eigenvalues (use special routine for entire spectrum).\n";
      return;
    }

    if (ncv > n)
    {
      std::cout << "[ARPACK_ERROR]: \"arpack_dcn\" does not support more internal vectors than the dimension of the matrix.\n";
      return;
    }

    // Allocate fixed memory arrays.
    v = allocate_vector<std::complex<double> >(ldv*ncv);
    d = allocate_vector<std::complex<double> >(ncv);
    workl = allocate_vector<std::complex<double> >(3*ncv*ncv+5*ncv);
    workd = allocate_vector<std::complex<double> >(3*n);
    workev = allocate_vector<std::complex<double> >(2*ncv);
    resid = allocate_vector<std::complex<double> >(n);
    rwork = allocate_vector<double>(ncv);
    rd = allocate_vector<double>(3*ncv);
    select = allocate_vector<int>(n);

    is_allocated = true;
    if (nev == n/2 && ncv == n)
      entire_spectrum = true; // whether or not the user ever wants it, we're ready.
  }

  // Allocate memory for a solve with specific nev, "heuristic" ncv.
  arpack_dcn(int n, int maxitr, double tol, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info, int nev)
   : arpack_dcn(n, maxitr, tol, matrix_vector, extra_info, nev, std::min(n, 2*nev + nev/2)) { ; }

  // Allocate memory for a solve of the entire spectrum.
  arpack_dcn(int n, int maxitr, double tol, void (*matrix_vector)(complex<double>*,complex<double>*,void*), void* extra_info)
   : arpack_dcn(n, maxitr, tol, matrix_vector, extra_info, n/2, n)
  {
    entire_spectrum = true;
  }



  // Free memory.
  ~arpack_dcn()
  {
    if (is_allocated)
    {
      if (v != 0) { deallocate_vector(&v); v = 0; }
      if (d != 0) { deallocate_vector(&d); d = 0; }
      if (workl != 0) { deallocate_vector(&workl); workl = 0; }
      if (workd != 0) { deallocate_vector(&workd); workd = 0; }
      if (workev != 0) { deallocate_vector(&workev); workev = 0; }
      if (resid != 0) { deallocate_vector(&resid); resid = 0; }
      if (rwork != 0) { deallocate_vector(&rwork); rwork = 0; }
      if (rd != 0) { deallocate_vector(&rd); rd = 0; }
      if (select != 0) { deallocate_vector(&select); select = 0; }
    }
  }

private:

  // Internal function which prepares 
  // the various private arrays in this class
  // to return eigenvalues, eigenvectors.
  bool internal_prepare(arpack_spectrum_piece in_spec_piece, int in_ev, int in_cv)
  {
    if (!is_allocated)
    {
      std::cout << "[ARPACK_ERROR]: \"arpack_dcn->internal_prepare\" cannot be called as internal structures are not allocated.\n";
      info_solve.success = false; 
      return false;
    }

    if (in_ev > max_nev)
    {
      std::cout << "[ARPACK_ERROR]: \"arpack_dcn->internal_prepare\" cannot be used to obtain more than nev eigenvalues.\n";
    }

    if (in_ev == 0)
    {
      std::cout << "[ARPACK_ERROR]: \"arpack_dcn->internal_prepare\" cannot be used to obtain zero eigenvalues.\n";
    }

    if (in_cv > max_ncv)
    {
      std::cout << "[ARPACK_ERROR]: \"arpack_dcn->internal_prepare\" cannot be used to obtain more than nev eigenvalues.\n";
    }

    if (in_cv < (in_ev) + 2)
    {
      std::cout << "[ARPACK_ERROR]: \"arpack_dcn->internal_prepare\" cannot be used when ncv < nev+2.\n";
    }

    info_solve.iter = 0;
    info_solve.ops_count = 0;
    info_solve.n_conv = 0;
    info_solve.znaupd_code = 0;
    info_solve.success = false; // bad unless proven otherwise.

    // Begin!
    bmat = 'I'; // This is a standard problem, as opposed
               // to a generalized eigenvalue problem.
   
    // Stopping rules + initial conditions before calling DSAUPD
    lworkl = 3*in_cv*in_cv+5*in_cv; // Trusting arpack here.
    ido = 0; // This is the reverse communication parameter 
            // from DSAUPD. Each call changes the value of this
            // parameter, and based on its value something must
            // be done. Has to be set to 0 before first call.
    info = 0; // On first use, specifies starting vector. Setting it
             // to 0 means use a random initial vector for arnoldi
             // iterations. Non-zero: pass in starting vector to
             // the array "resid".
             
    // Specify the algorithm mode. 
    ishfts = 1; // use an exact shift strategy. check DSAUPD
               // documentation for what this means. There are
               // options here to shift the matrix. (PARAM(1))
    mode1 = 1; // Use mode 1 of DSAUPD. Check documentation! (PARAM(7))

    // Set up iparam.
    iparam[0] = ishfts; // Used for a shift-invert strategy. 
    iparam[2] = maxitr; // On return, gives actual number of iters.
    iparam[3] = 1; // Blocking strategy. arpack-ng only supports 1.
    iparam[6] = mode1; 
   
    switch (in_spec_piece)
    {
      case ARPACK_LARGEST_MAGNITUDE:
        strcpy(which, "LM");
        break;
      case ARPACK_SMALLEST_MAGNITUDE:
        strcpy(which, "SM");
        break;
      case ARPACK_LARGEST_REAL:
        strcpy(which, "LR");
        break;
      case ARPACK_SMALLEST_REAL:
        strcpy(which, "SR");
        break;
      case ARPACK_LARGEST_IMAGINARY:
        strcpy(which, "LI");
        break;
      case ARPACK_SMALLEST_IMAGINARY:
        strcpy(which, "SI");
        break;
      case ARPACK_NONE:
      default:
        std::cout << "[ARPACK_ERROR]: \"arpack_dcn->internal_prepare\" given an inappropriate eigenvalue type.\n";
        return false; 
    }
   
    // Main reverse communication loop!
    while (ido != 99) // 99 means we're complete!
    {
      // Call dsaupd!

      ARPACK(znaupd)(&ido, &bmat, &n, which, &in_ev, &tol,
                   resid, &in_cv, v, &ldv,
                   (int*)iparam, (int*)ipntr, workd,
                   workl, &lworkl, rwork, &info);

      // Check for errors!                   
      if (info != 0)
      {
        info_solve.iter = iparam[2];
        info_solve.ops_count = iparam[8];
        info_solve.n_conv = iparam[4];
        info_solve.znaupd_code = info;
        return false; 
      }

      // See if we need to iterate more!
      if (ido == -1 || ido == 1)
      {
        // Perform the y = A*x, where 'x' starts at workd[ipntr[0]-1]
        // and y starts at workd[ipntr[1]-1], where the -1 is
        // because C, not FORTRAN.
        (*matrix_vector)(&(workd[ipntr[1]-1]), &(workd[ipntr[0]-1]), extra_info);
      }


    } // Loop back.

    // Done with loop!

    // May still be meaningful even if there was a failure.
    info_solve.iter = iparam[2];
    info_solve.ops_count = iparam[8];
    info_solve.n_conv = iparam[4];
    info_solve.success = true;

    ev = in_ev;
    cv = in_cv;
    spec_piece = in_spec_piece; 

    return true;
  }

private: 

  // Private merge sort function which sorts by eigenvalue, also rearranged eigenvectors.
  bool merge_sort(complex<double>* sort_evals, complex<double>** sort_evecs, complex<double>* temp_evals, complex<double>** temp_evecs, int size, arpack_spectrum_piece spectrum_sort, bool has_evecs)
  {
    if (spectrum_sort == ARPACK_NONE)
      return false;

    if (size == 1) // trivial
    {
      return true;
    }
    if (size == 2) // trivial
    {
      switch (spectrum_sort)
      {
        case ARPACK_NONE:
          return false;
        case ARPACK_SMALLEST_MAGNITUDE:
          if (std::abs(sort_evals[1]) < std::abs(sort_evals[0]))
          {
            std::swap(sort_evals[1], sort_evals[0]);
            if (has_evecs) std::swap(sort_evecs[1], sort_evecs[0]);
          }
          break;
        case ARPACK_LARGEST_MAGNITUDE:
          if (std::abs(sort_evals[1]) > std::abs(sort_evals[0]))
          {
            std::swap(sort_evals[1], sort_evals[0]);
            if (has_evecs) std::swap(sort_evecs[1], sort_evecs[0]);
          }
          break;
        case ARPACK_SMALLEST_REAL:
          if (std::real(sort_evals[1]) < std::real(sort_evals[0]))
          {
            std::swap(sort_evals[1], sort_evals[0]);
            if (has_evecs) std::swap(sort_evecs[1], sort_evecs[0]);
          }
          break;
        case ARPACK_LARGEST_REAL:
          if (std::real(sort_evals[1]) > std::real(sort_evals[0]))
          {
            std::swap(sort_evals[1], sort_evals[0]);
            if (has_evecs) std::swap(sort_evecs[1], sort_evecs[0]);
          }
          break;
        case ARPACK_SMALLEST_IMAGINARY:
          if (std::imag(sort_evals[1]) < std::imag(sort_evals[0]))
          {
            std::swap(sort_evals[1], sort_evals[0]);
            if (has_evecs) std::swap(sort_evecs[1], sort_evecs[0]);
          }
          break;
        case ARPACK_LARGEST_IMAGINARY:
          if (std::imag(sort_evals[1]) > std::imag(sort_evals[0]))
          {
            std::swap(sort_evals[1], sort_evals[0]);
            if (has_evecs) std::swap(sort_evecs[1], sort_evecs[0]);
          }
          break;
        default:
          return false;                                     
          break;
      }
      return true;
    }

    // recurse.
    if (!merge_sort(sort_evals, sort_evecs, temp_evals, temp_evecs, size/2, spectrum_sort, has_evecs))
      return false;

    if (!merge_sort(sort_evals + size/2, sort_evecs + size/2, temp_evals + size/2, temp_evecs + size/2, size - size/2, spectrum_sort, has_evecs))
      return false;

    // merge
    int curr1 = 0;
    int curr2 = size/2;
    int currtmp = 0;

    while (curr1 < size/2 && curr2 < size)
    {
      switch (spectrum_sort)
      {
        case ARPACK_NONE:
          return false;
        case ARPACK_SMALLEST_MAGNITUDE:
          if (std::abs(sort_evals[curr1]) < std::abs(sort_evals[curr2]))
          {
            temp_evals[currtmp++] = sort_evals[curr1++];
            if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr1-1];
          }
          else
          {
            temp_evals[currtmp++] = sort_evals[curr2++];
            if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr2-1]; 
          }
          break;
        case ARPACK_LARGEST_MAGNITUDE:
          if (std::abs(sort_evals[curr1]) > std::abs(sort_evals[curr2]))
          {
            temp_evals[currtmp++] = sort_evals[curr1++];
            if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr1-1];
          }
          else
          {
            temp_evals[currtmp++] = sort_evals[curr2++];
            if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr2-1]; 
          }
          break;
        case ARPACK_SMALLEST_REAL:
          if (std::real(sort_evals[curr1]) < std::real(sort_evals[curr2]))
          {
            temp_evals[currtmp++] = sort_evals[curr1++];
            if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr1-1];
          }
          else
          {
            temp_evals[currtmp++] = sort_evals[curr2++];
            if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr2-1]; 
          }
          break;
        case ARPACK_LARGEST_REAL:
          if (std::real(sort_evals[curr1]) > std::real(sort_evals[curr2]))
          {
            temp_evals[currtmp++] = sort_evals[curr1++];
            if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr1-1];
          }
          else
          {
            temp_evals[currtmp++] = sort_evals[curr2++];
            if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr2-1]; 
          }
          break;
        case ARPACK_SMALLEST_IMAGINARY:
          if (std::imag(sort_evals[curr1]) < std::imag(sort_evals[curr2]))
          {
            temp_evals[currtmp++] = sort_evals[curr1++];
            if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr1-1];
          }
          else
          {
            temp_evals[currtmp++] = sort_evals[curr2++];
            if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr2-1]; 
          }
          break;
        case ARPACK_LARGEST_IMAGINARY:
          if (std::imag(sort_evals[curr1]) > std::imag(sort_evals[curr2]))
          {
            temp_evals[currtmp++] = sort_evals[curr1++];
            if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr1-1];
          }
          else
          {
            temp_evals[currtmp++] = sort_evals[curr2++];
            if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr2-1]; 
          }
          break;
        default:
          return false;                                     
          break;
      }
    }

    while (curr1 < size/2)
    {
      temp_evals[currtmp++] = sort_evals[curr1++];
      if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr1-1];
    }

    while (curr2 < size)
    {
      temp_evals[currtmp++] = sort_evals[curr2++];
      if (has_evecs) temp_evecs[currtmp-1] = sort_evecs[curr2-1];
    }

    for (int i = 0; i < size; i++)
    {
      sort_evals[i] = temp_evals[i];
      if (has_evecs) sort_evecs[i] = temp_evecs[i];
    }

    return true;
  }

  public:

  // Prepare the eigensystem with some specific properties, or the defaults. 
  bool prepare_eigensystem(arpack_spectrum_piece in_spec_piece = ARPACK_NONE, int in_ev = 0, int in_cv = 0)
  {
    if (in_ev == 0) { in_ev = ev; }
    if (in_cv == 0)
    {
      if (cv == 0)
        in_cv = cv = std::min(2*in_ev+in_ev/2, max_ncv);
      else
        in_cv = cv;
    }
    if (in_ev > max_nev)
    {
      std::cout << "[ARPACK]: In \"prepare_eigensystem\", cannot ask for more eigenvalues than max_nev.\n";
      return false;
    }
    if (in_cv > max_ncv)
    {
      std::cout << "[ARPACK]: In \"prepare_eigensystem\", cannot ask for more cv than max_ncv.\n";
      return false; 
    }

    // Call the internal prep function.
    access_ready = internal_prepare(in_spec_piece, in_ev, in_cv);
    return access_ready;
  }

  // Check if the system is ready to access.
  bool is_ready()
  {
    return access_ready;
  }

  // Get the solve state. 
  arpack_info get_solve_info()
  {
    return info_solve;
  }

  // Get number of computed ev's (or update it, invalidating the state.)
  int check_ev(int new_ev = 0)
  {
    if (new_ev != 0)
    {
      ev = new_ev;
      access_ready = false;
    }
    if (new_ev > max_nev || new_ev == 0)
    {
      std::cout << "[ARPACK]: Cannot set 'ev' to greater than 'max_nev'.\n";
    }
    return ev;
  }

  /// Get number of cv's (or update it, invalidating the state.)
  int check_cv(int new_cv = 0)
  {
    if (new_cv != 0)
    {
      cv = new_cv;
      access_ready = false;
    }
    if (new_cv > max_ncv || new_cv == 0)
    {
      std::cout << "[ARPACK]: Cannot set 'cv' to greater than 'max_ncv'.\n";
    }
    return cv;
  }

  /// Get part of spectrum computed (or update it, invalidating the state.)
  arpack_spectrum_piece check_cv(arpack_spectrum_piece new_spec_piece = ARPACK_NONE)
  {
    if (new_spec_piece != ARPACK_NONE)
    {
      spec_piece = new_spec_piece;
      access_ready = false;
    }
    else
    {
      std::cout << "[ARPACK]: Cannot set spectrum piece to ARPACK_NONE.\n";
    }
    return spec_piece;
  }

  void* check_extra_info(void* new_extra_info = 0)
  {
    if (new_extra_info != 0)
    {
      extra_info = new_extra_info;
      access_ready = false;
    }
    else
    {
      std::cout << "[ARPACK]: Cannot set 'extra_info' to zero.\n";
    }
    return extra_info; 
  }

  // There's a special level of hell for function pointers...
  // I should use a typedef.
  void (*check_matrix_vector())(complex<double>*,complex<double>*,void*)
  {
    return matrix_vector;
  }

  // Force an invalidation of state.
  void force_invalidate()
  {
    access_ready = false;
  }
  

  // fcn to grab eigenvalues (and optionally eigenvectors) if system is prepared.
  // Sorts eigenvalues (and eigenvectors) via enum, optionally.
  // If sort is, for ex, ARPACK_SMALLEST_MAGNITUDE, the smallest mag eigenvalue comes first.
  bool get_eigensystem(complex<double>* eigvals, complex<double>** eigvecs, arpack_spectrum_piece spectrum_sort)
  {
    if (!access_ready)
    {
      return false;
    }

    // Specify a few things...
    hwmny = 'A'; // Get Ritz vectors (assuming we want eigenvectors)
    rvec = (eigvecs == 0) ? 0 : 1;
    complex<double> sigma = 0; // does not matter b/c we're not doing a shift-invert.

    ARPACK(zneupd)(&rvec, &hwmny, select, d,
                      v,  &ldv, &sigma, (complex<double>*)workev,
                      &bmat, &n, which,
                      &ev, &tol, resid, &cv, v,
                      &ldv, (int*)iparam, (int*)ipntr, workd,
                      workl, &lworkl, rwork, &ierr);

    // Check for errors
    if (ierr != 0)
    {
      std::cout << "[ARPACK]: Error in get_eigensystem.\n";
      return false;
    }

    // Copy them over!
    for (int i = 0; i < ev; i++)
    {
      eigvals[i] = d[i];
    }
    if (eigvecs != 0)
    {
      for (int i = 0; i < ev; i++)
        copy_vector(eigvecs[i], (v+i*ldv), n);
    }

    // Sort if we're supposed to.
    if (spectrum_sort != ARPACK_NONE)
    {
      // temp memory.
      complex<double>* temp_evals = new complex<double>[ev];
      complex<double>** temp_evecs = new complex<double>*[ev];
      if (!merge_sort(eigvals, eigvecs, temp_evals, temp_evecs, ev, spectrum_sort, (bool)(eigvecs != 0)))
      {
        delete[] temp_evals;
        delete[] temp_evecs;
        return false;
      }
      delete[] temp_evals;
      delete[] temp_evecs;
    }
    return true; 

  }

  bool get_eigensystem(complex<double>* eigvals, arpack_spectrum_piece spectrum_sort)
  {
    return get_eigensystem(eigvals, 0, spectrum_sort);
  }

  bool get_eigensystem(complex<double>* eigvals, complex<double>** eigvecs)
  {
    return get_eigensystem(eigvals, eigvecs, ARPACK_NONE);
  }

  bool get_eigensystem(complex<double>* eigvals)
  {
    return get_eigensystem(eigvals, 0, ARPACK_NONE);
  }

  // SPECIAL function to get entire spectrum. Does not leave
  // system in a prepared state.
  bool get_entire_eigensystem(complex<double>* eigvals, complex<double>** eigvecs, arpack_spectrum_piece spectrum_sort)
  {
    // Needs max_nev == n/2, max_ncv == n.
    if (!entire_spectrum)
      return false;

    // Grab the relevant parts of the spectrum so they're pre-sorted.
    arpack_spectrum_piece first_part = ARPACK_SMALLEST_MAGNITUDE, second_part = ARPACK_LARGEST_MAGNITUDE;

    switch (spectrum_sort)
    {
      case ARPACK_NONE:
      case ARPACK_SMALLEST_MAGNITUDE:
        first_part = ARPACK_SMALLEST_MAGNITUDE;
        second_part = ARPACK_LARGEST_MAGNITUDE;
        break;
      case ARPACK_LARGEST_MAGNITUDE:
        first_part = ARPACK_LARGEST_MAGNITUDE;
        second_part = ARPACK_SMALLEST_MAGNITUDE;
        break;
       case ARPACK_SMALLEST_REAL:
        first_part = ARPACK_SMALLEST_REAL;
        second_part = ARPACK_LARGEST_REAL;
        break;
      case ARPACK_LARGEST_REAL:
        first_part = ARPACK_LARGEST_REAL;
        second_part = ARPACK_SMALLEST_REAL;
        break;
      case ARPACK_SMALLEST_IMAGINARY:
        first_part = ARPACK_SMALLEST_IMAGINARY;
        second_part = ARPACK_LARGEST_IMAGINARY;
        break;
      case ARPACK_LARGEST_IMAGINARY:
        first_part = ARPACK_LARGEST_IMAGINARY;
        second_part = ARPACK_SMALLEST_IMAGINARY;
        break;
      
    }

    if (!prepare_eigensystem(first_part, n/2, n))
      return false;

    if (!get_eigensystem(eigvals, eigvecs, spectrum_sort))
      return false;

    if (!prepare_eigensystem(second_part, n/2, n))
      return false;

    if (!get_eigensystem(eigvals + n/2, (eigvecs == 0) ? 0 : (eigvecs + n/2), spectrum_sort))
      return false;

    access_ready = false;
    return true;
  }

  bool get_entire_eigensystem(complex<double>* eigvals, arpack_spectrum_piece spectrum_sort)
  {
    return get_entire_eigensystem(eigvals, 0, spectrum_sort);
  }

  bool get_entire_eigensystem(complex<double>* eigvals, complex<double>** eigvecs)
  {
    return get_entire_eigensystem(eigvals, eigvecs, ARPACK_NONE);
  }

  bool get_entire_eigensystem(complex<double>* eigvals)
  {
    return get_entire_eigensystem(eigvals, 0, ARPACK_NONE);
  }
   
};

#endif // QLINALG_INTERFACE_ARPACK