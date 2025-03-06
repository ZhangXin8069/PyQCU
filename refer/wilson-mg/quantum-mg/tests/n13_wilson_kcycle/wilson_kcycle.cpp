// Copyright (c) 2017 Evan S Weinberg
// A test of a K-cycle for the interacting
// Wilson operator. Generates null vectors using
// BiCGstab-L.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"
#include "inverters/generic_gcr.h"
#include "inverters/generic_gcr_var_precond.h"
#include "inverters/generic_bicgstab_l.h"
#include "interfaces/arpack/generic_arpack.h"

// QMG
#include "lattice/lattice.h"
#include "transfer/transfer.h"
#include "stencil/stencil_2d.h"
#include "multigrid/stateful_multigrid.h"

// Grab Wilson operator (we just use unit gauge fields -> free laplace)
#include "operators/wilson.h"
#include "u1/u1_utils.h"


int main(int argc, char** argv)
{
  // Iterators.
  int i,j,k;

  if (argc != 5)
  {
    std::cout << "Error: ./wilson_kcycle expects four arguments, L, mass, beta, n_refine. Try mass = -0.075 for beta 6.0.\n";
    return -1;
  }

  // Are we testing the free (two exact null vectors) or
  // interacting (four algebraic null vectors) case?
  const bool do_free = false;

  // Check the spectrum?
  const bool do_spectrum = false;

  // Are we performing various colinearity checks?
  bool do_colinear = false;

  // Do we use eigenvectors as null vectors?
  bool nulls_are_eigenvectors = false;

  // Do we grab just positive eigenvectors for null vectors, or all?
  bool nulls_positive_evec_only = false;

  // Set output precision to be long.
  cout << setprecision(20);

  // Random number generator.
  std::mt19937 generator (1337u);

  // Basic information for fine level.
  const int x_len = stoi(argv[1]);
  const int y_len = stoi(argv[1]);
  const double beta = stod(argv[3]); 
  const int dof = Wilson2D::get_dof();

  // Information on the Wilson operator.
  double mass;
  if (do_free)
  {
    mass = 0.1;
  }
  else
  {
    // Staggered specific information.
    // For 64^2, beta = 6.0, eigenvalues go negative around -0.075.
    mass = stod(argv[2]);
  }

  // Blocking size.
  const int x_block = 4;
  const int y_block = 4;

  // Number of null vectors.
  int coarse_dof;
  if (do_free)
  {
    coarse_dof = 2;
  }
  else
  {
    coarse_dof = 8;
  }

  // How many times to refine. 
  const int n_refine = stoi(argv[4]); // (64 -> 16 -> 4 -> 1)

  // Information about the outermost solve.
  const double tol = 1e-10; 
  const int max_iter = 1000;
  const int restart_freq = 32;

  // Information about intermediate solves.
  const double inner_tol = 0.2;
  const int inner_max_iter = 1000;
  const int inner_restart_freq = 32;

  // Information about pre- and post-smooths. 
  const int n_pre_smooth = 2;
  const double pre_smooth_tol = 1e-15; // never
  const int n_post_smooth = 2;
  const double post_smooth_tol = 1e-15; // never

  // Information about the coarsest solve.
  const double coarsest_tol = 0.2;
  const int coarsest_max_iter = 1000;
  const int coarsest_restart_freq = 32;

  // Somewhere to solve inversion info.
  inversion_info invif;

  // Verbosity.
  inversion_verbose_struct verb;
  verb.verbosity = VERB_DETAIL;
  verb.verb_prefix = "Level 0: ";
  verb.precond_verbosity = VERB_DETAIL;
  verb.precond_verb_prefix = "Prec ";

  // Create a lattice object for the fine lattice.
  Lattice2D** lats = new Lattice2D*[n_refine+1];
  lats[0] = new Lattice2D(x_len, y_len, dof);

  // Prepare the gauge field.
  Lattice2D* lat_gauge = new Lattice2D(x_len, y_len, 1); // hack...
  complex<double>* gauge_field = allocate_vector<complex<double>>(lat_gauge->get_size_gauge());
  if (do_free)
  {
    unit_gauge_u1(gauge_field, lat_gauge);
  }
  else
  {
    bool need_heatbath = false;
    if (beta == 6.0)
    {
      switch (x_len)
      {
        case 32:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l32t32b60_heatbath.dat");
          break;
        case 64:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l64t64b60_heatbath.dat");
          break;
        case 128:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l128t128b60_heatbath.dat");
          break;
        case 192:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l192t192b60_heatbath.dat");
          break;
        case 256:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l256t256b60_heatbath.dat");
          break;
        default:
          need_heatbath = true;
          break;
      }
    }
    else if (beta == 10.0)
    {
      switch (x_len)
      {
        case 32:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l32t32b100_heatbath.dat");
          break;
        case 64:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l64t64b100_heatbath.dat");
          break;
        case 128:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l128t128b100_heatbath.dat");
          break;
        case 192:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l192t192b100_heatbath.dat");
          break;
        default:
          need_heatbath = true;
          break;
      }
    }
    else
      need_heatbath = true;

    if (need_heatbath)
    {
      std::cout << "[QMG-NOTE]: L = " << x_len << " beta = " << beta << " requires heatbath generation.\n";

      int n_therm = 4000; // how many heatbath steps to perform.
      int n_meas = 100; // how often to measure the plaquette, topo.
      double* phases = allocate_vector<double>(lat_gauge->get_size_gauge());
      zero_vector(phases, lat_gauge->get_size_gauge());
      double plaq = 0.0; // track along the way
      double topo = 0.0;
      for (int i = 0; i < n_therm; i += n_meas)
      {
        // Perform non-compact update.
        heatbath_noncompact_update(phases, lat_gauge, beta, n_therm/n_meas, generator);

        // Get compact links.
        polar_vector(phases, gauge_field, lat_gauge->get_size_gauge());

        plaq = std::real(get_plaquette_u1(gauge_field, lat_gauge));
        topo = std::real(get_topo_u1(gauge_field, lat_gauge));
        std::cout << "[QMG-HEATBATH]: Update " << i << " Plaq " << plaq << " Topo " << topo << "\n";
      }

      // Acquire final gauge field.
      polar_vector(phases, gauge_field, lat_gauge->get_size_gauge());

      // Clean up.
      deallocate_vector(&phases);
    }
  }
  delete lat_gauge;


  // Create a Wilson stencil.
  Wilson2D* wilson_op = new Wilson2D(lats[0], mass, gauge_field);

  // Prepare level solve objects for the top level.
  StatefulMultigridMG::LevelSolveMG** level_solve_objs = new StatefulMultigridMG::LevelSolveMG*[n_refine];

  // Prepare coarsest solve object for the coarsest level.
  StatefulMultigridMG::CoarsestSolveMG* coarsest_solve_obj = new StatefulMultigridMG::CoarsestSolveMG;
  coarsest_solve_obj->coarsest_stencil_app = QMG_MATVEC_ORIGINAL;
  coarsest_solve_obj->coarsest_tol = coarsest_tol;
  coarsest_solve_obj->coarsest_iters = coarsest_max_iter;
  coarsest_solve_obj->coarsest_restart_freq = coarsest_restart_freq;

  // Create a MultigridMG object, push top level onto it!
  StatefulMultigridMG* mg_object = new StatefulMultigridMG(lats[0], wilson_op, coarsest_solve_obj);

  // Create coarse lattices, unit null vectors, transfer objects.
  // Push into MultigridMG object. 
  int curr_x_len = x_len;
  int curr_y_len = y_len;
  TransferMG** transfer_objs = new TransferMG*[n_refine];
  for (i = 1; i <= n_refine; i++)
  {
    const int fine_idx = i-1;

    // Update to the new lattice size.
    curr_x_len /= x_block;
    curr_y_len /= y_block;

    // Create a new lattice object.
    lats[i] = new Lattice2D(curr_x_len, curr_y_len, coarse_dof);

    // Create a new null vectors. These are copied into local memory in the
    // transfer object, so we can create and destroy these in this loop.
    complex<double>** null_vectors = new complex<double>*[coarse_dof];
    if (do_free)
    {
      // Two null vectors, one for each parity component.

      // Top.
      null_vectors[0] = allocate_vector<complex<double>>(lats[i-1]->get_size_cv());
      zero_vector(null_vectors[0], lats[i-1]->get_size_cv());
      constant_vector_blas(null_vectors[0], 2, 1.0, lats[i-1]->get_size_cv()/2);

      // Bottom.
      null_vectors[1] = allocate_vector<complex<double>>(lats[i-1]->get_size_cv());
      zero_vector(null_vectors[1], lats[i-1]->get_size_cv());
      constant_vector_blas(null_vectors[1]+1, 2, 1.0, lats[i-1]->get_size_cv()/2);
    }
    else
    {
      for (j = 0; j < coarse_dof; j++)
      {
        null_vectors[j] = allocate_vector<complex<double> >(lats[i-1]->get_size_cv());
        zero_vector(null_vectors[j], lats[i-1]->get_size_cv());
      }

      if (nulls_are_eigenvectors)
      {

        arpack_dcn* arpack;
        complex<double>* coarsest_evals_right = new complex<double>[coarse_dof];
        complex<double>** coarsest_evecs_right = new complex<double>*[coarse_dof];
        for (j = 0; j < coarse_dof; j++)
        {
          coarsest_evecs_right[j] = allocate_vector<complex<double>>(lats[fine_idx]->get_size_cv());
        }
        
        // Grab lowest coarse_dof eigenvectors of D.
        arpack = new arpack_dcn(lats[fine_idx]->get_size_cv(), 4000, 1e-7,
                      apply_stencil_2D_M, mg_object->get_stencil(fine_idx),
                      coarse_dof, 3*coarse_dof);

        arpack->prepare_eigensystem(arpack_dcn::ARPACK_SMALLEST_MAGNITUDE, coarse_dof, 3*coarse_dof);
        arpack->get_eigensystem(coarsest_evals_right, coarsest_evecs_right, arpack_dcn::ARPACK_SMALLEST_MAGNITUDE);
        delete arpack;
        for (j = 0; j < coarse_dof; j++)
        {
          std::cout << "Right eval " << j << " " << coarsest_evals_right[j] << "\n";
          normalize(coarsest_evecs_right[j], lats[fine_idx]->get_size_cv());
        }
        for (j = 0; j < coarse_dof/2; j++)
        {
          if (nulls_positive_evec_only) // grab only positive imag
          {
            if (coarsest_evals_right[2*j].imag() > coarsest_evals_right[2*j+1].imag())
            {
              copy_vector(null_vectors[j], coarsest_evecs_right[2*j], lats[fine_idx]->get_size_cv());
            }
            else
            {
              copy_vector(null_vectors[j], coarsest_evecs_right[2*j+1], lats[fine_idx]->get_size_cv());
            }
          }
          else
          {
            copy_vector(null_vectors[j], coarsest_evecs_right[j], lats[fine_idx]->get_size_cv());
          }
        }
        for (j = 0; j < coarse_dof; j++)
        {
          deallocate_vector(&coarsest_evecs_right[j]);
        }
        delete[] coarsest_evecs_right;
        delete[] coarsest_evals_right;
      }
      else
      {
        // Create coarse_dof null vectors.
        for (j = 0; j < coarse_dof/2; j++)
        {
          // Check out vector.
          complex<double>* rand_guess = mg_object->get_storage(i-1)->check_out();

          // Fill with random numbers.
          gaussian(rand_guess, lats[i-1]->get_size_cv(), generator);

          // Make orthogonal to previous vectors.
          for (k = 0; k < j; k++)
            orthogonal(rand_guess, null_vectors[k], lats[i-1]->get_size_cv());

          // Check out vector for residual equation.
          complex<double>* Arand_guess = mg_object->get_storage(i-1)->check_out();

          // Zero, form residual.
          zero_vector(Arand_guess, lats[i-1]->get_size_cv());
          mg_object->get_stencil(i-1)->apply_M(Arand_guess, rand_guess);
          cax(-1.0, Arand_guess, lats[i-1]->get_size_cv());

          // Solve residual equation.
          minv_vector_bicgstab_l(null_vectors[j], Arand_guess, lats[i-1]->get_size_cv(), 500, 5e-5, 6, apply_stencil_2D_M, (void*)mg_object->get_stencil(i-1), &verb);

          // Undo residual equation.
          cxpy(rand_guess, null_vectors[j], lats[i-1]->get_size_cv());

          // Check in.
          mg_object->get_storage(i-1)->check_in(rand_guess);
          mg_object->get_storage(i-1)->check_in(Arand_guess);

          // Orthogonalize against previous vectors.
          for (k = 0; k < j; k++)
            orthogonal(null_vectors[j], null_vectors[k], lats[i-1]->get_size_cv());
        }
      }

      // Perform chiral projection.
      for (j = 0; j < coarse_dof/2; j++)
      {
        // Perform chiral projection, putting the "down" projection into the second
        // vector and keeping the "up" projection in the first vector.
        mg_object->get_stencil(i-1)->chiral_projection_both(null_vectors[j], null_vectors[j+lats[i]->get_nc()/2]);

        // Normalize.
        normalize(null_vectors[j], lats[i-1]->get_size_cv());
        normalize(null_vectors[j+lats[i]->get_nc()/2], lats[i-1]->get_size_cv());
      }
    }

    // Create and populate a transfer object.
    // Fine lattice, coarse lattice, null vector(s), perform the block ortho, save coeffs, coarse chirality type
    transfer_objs[i-1] = new TransferMG(lats[i-1], lats[i], null_vectors, true, false, QMG_DOUBLE_PROJECTION);

    // Prepare a new LevelSolveMG object for the new level.
    level_solve_objs[i-1] = new StatefulMultigridMG::LevelSolveMG;
    level_solve_objs[i-1]->fine_stencil_app = QMG_MATVEC_ORIGINAL;
    level_solve_objs[i-1]->intermediate_tol = inner_tol;
    level_solve_objs[i-1]->intermediate_iters = inner_max_iter;
    level_solve_objs[i-1]->intermediate_restart_freq = inner_restart_freq;
    level_solve_objs[i-1]->pre_tol = pre_smooth_tol;
    level_solve_objs[i-1]->pre_iters = n_pre_smooth;
    level_solve_objs[i-1]->post_tol = post_smooth_tol;
    level_solve_objs[i-1]->post_iters = n_post_smooth;

    // Push a new level on the multigrid object! Also, save the global null vector.
    // Arg 1: New lattice
    // Arg 2: New transfer object (between new and prev lattice)
    // Arg 3: Level solve object (specifies how to do intermediate solves and smooths)
    // Arg 3: Should we construct the coarse stencil?
    // Arg 4: Is the operator chiral? (True for Wilson)
    // Arg 5: What should we construct the coarse stencil from? (Not relevant yet.)
    // Arg 6: Non-block-orthogonalized null vector.
    mg_object->push_level(lats[i], transfer_objs[i-1], level_solve_objs[i-1], true, true, MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL, null_vectors);

    // Clean up local vector, since they get copied in.
    for (j = 0; j < coarse_dof; j++)
      deallocate_vector(&null_vectors[j]);
    delete[] null_vectors;
  }

  // Prepare storage and a guess right hand side.

  // Create a right hand side, fill with gaussian random numbers.
  //complex<double>* b = allocate_vector<complex<double>>(lats[0]->get_size_cv());
  complex<double>* b = mg_object->check_out(0);
  gaussian(b, lats[0]->get_size_cv(), generator);
  double bnorm = sqrt(norm2sq(b, lats[0]->get_size_cv()));

  // Create a place to accumulate a solution. Assume a zero initial guess.
  complex<double>* x = mg_object->check_out(0);
  zero_vector(x, lats[0]->get_size_cv());

  // Create a place to compute Ax. Since we have a zero initial guess, this
  // starts as zero.
  complex<double>* Ax = mg_object->check_out(0);
  zero_vector(Ax, lats[0]->get_size_cv());

  ///////////////////
  // Non-MG solve! //
  ///////////////////
  /*invif = minv_vector_gcr_restart(x, b, lats[0]->get_size_cv(),
              max_iter, tol, restart_freq,
              apply_stencil_2D_M, (void*)mg_object->get_stencil(0));

  cout << "Simple GCR solve " << (invif.success ? "converged" : "failed to converge")
          << " in " << invif.iter << " iterations with alleged tolerance "
          << sqrt(invif.resSq)/bnorm << ".\n";
  zero_vector(Ax, lats[0]->get_size_cv());
  mg_object->apply_stencil(Ax, x, 0);
  cout << "Check tolerance " << sqrt(diffnorm2sq(b, Ax, lats[0]->get_size_cv()))/bnorm << "\n";
  */

  ////////////////////
  // K-cycle solve! //
  ////////////////////

  // Reset values.
  zero_vector(x, lats[0]->get_size_cv());
  zero_vector(Ax, lats[0]->get_size_cv());

  // Run a VPGCR solve!
  invif = minv_vector_gcr_var_precond_restart(x, b, lats[0]->get_size_cv(),
              max_iter, tol, restart_freq,
              apply_stencil_2D_M, (void*)mg_object->get_stencil(0),
              StatefulMultigridMG::mg_preconditioner, (void*)mg_object, &verb); 

  cout << "Multigrid " << (invif.success ? "converged" : "failed to converge")
          << " in " << invif.iter << " iterations with alleged tolerance "
          << sqrt(invif.resSq)/bnorm << ".\n";

  // Check solution.
  zero_vector(Ax, lats[0]->get_size_cv());
  mg_object->apply_stencil(Ax, x, 0);
  cout << "Check tolerance " << sqrt(diffnorm2sq(b, Ax, lats[0]->get_size_cv()))/bnorm << "\n";

  // Check vectors back in.
  mg_object->check_in(Ax, 0);
  mg_object->check_in(x, 0);
  mg_object->check_in(b, 0);

  /////////////////////////
  // Check the spectrum. //
  /////////////////////////

  if (do_spectrum)
  {
    // Declare an arpack object and some storage.
    arpack_dcn* arpack;
    complex<double>* eigs;

    // Get spectrum of Wilson op.
    arpack = new arpack_dcn(lats[0]->get_size_cv(), 4000, 1e-7, apply_stencil_2D_M, (void*)mg_object->get_stencil(0));
    eigs = new complex<double>[lats[0]->get_size_cv()];
    /*evecs = new complex<double>*(lats[1]->get_size_cv());
    for (i = 0; i < lats[1]->get_size_cv())
    {
      evecs[i] = allocate_vector<complex<double> >(lats[1]->get_size_cv());
    }*/

    arpack->get_entire_eigensystem(eigs, arpack_dcn::ARPACK_SMALLEST_REAL);

    for (i = 0; i < lats[0]->get_size_cv(); i++)
      std::cout << "[ORIG-SPECTRUM]: " << i << " " << real(eigs[i]) << " + I " << imag(eigs[i]) << "\n";

    delete[] eigs;
    delete arpack;

    // Get spectrum of coarsened op.
    arpack = new arpack_dcn(lats[1]->get_size_cv(), 4000, 1e-7, apply_stencil_2D_M, (void*)mg_object->get_stencil(1));
    eigs = new complex<double>[lats[1]->get_size_cv()];
    /*evecs = new complex<double>*(lats[1]->get_size_cv());
    for (i = 0; i < lats[1]->get_size_cv())
    {
      evecs[i] = allocate_vector<complex<double> >(lats[1]->get_size_cv());
    }*/

    arpack->get_entire_eigensystem(eigs, arpack_dcn::ARPACK_SMALLEST_REAL);

    for (i = 0; i < lats[1]->get_size_cv(); i++)
      std::cout << "[COARSE-SPECTRUM]: " << i << " " << real(eigs[i]) << " + I " << imag(eigs[i]) << "\n";

    delete[] eigs;
    delete arpack;
  }

  //////////////////////////////////////////
  // Do various local co-linearity checks //
  //////////////////////////////////////////

  if (do_colinear)
  {
    // Get eigenvalues and eigenvectors of fine op.

    // Declare an arpack object and some storage.
    arpack_dcn* arpack;
    complex<double>* eigs;
    complex<double>** evecs;

    arpack = new arpack_dcn(lats[0]->get_size_cv(), 4000, 1e-7, apply_stencil_2D_M, (void*)mg_object->get_stencil(0));
    eigs = new complex<double>[lats[0]->get_size_cv()];
    evecs = new complex<double>*[lats[0]->get_size_cv()];
    for (i = 0; i < lats[0]->get_size_cv(); i++)
    {
      evecs[i] = allocate_vector<complex<double> >(lats[0]->get_size_cv());
    }

    arpack->get_entire_eigensystem(eigs, evecs, arpack_dcn::ARPACK_SMALLEST_MAGNITUDE);
    delete arpack;

    //for (i = 0; i < lats[0]->get_size_cv(); i++)
    //  std::cout << "[CIRCLE-SPECTRUM]: " << i << " " << real(eigs[i]) << " + I " << imag(eigs[i]) << "\n";

    // Build some lists.
    std::vector<double> onePP;//(lats[0]->get_size_cv());
    std::vector<double> onePAPA;//(lats[0]->get_size_cv());

    // Check ||(1 - P P^\dagger) v||
    complex<double>* PdagV = mg_object->get_storage(1)->check_out();
    complex<double>* PPdagV = mg_object->get_storage(0)->check_out();
    for (i = 0; i < lats[0]->get_size_cv(); i++)
    {
      zero_vector(PdagV, lats[1]->get_size_cv());
      zero_vector(PPdagV, lats[0]->get_size_cv());

      mg_object->get_transfer(0)->restrict_f2c(evecs[i], PdagV);
      mg_object->get_transfer(0)->prolong_c2f(PdagV, PPdagV);

      onePP.push_back(sqrt(diffnorm2sq(evecs[i], PPdagV, lats[0]->get_size_cv())/norm2sq(evecs[i], lats[0]->get_size_cv())));
    }

    mg_object->get_storage(1)->check_in(PdagV);
    mg_object->get_storage(0)->check_in(PPdagV);


    // Check ||(1-P(P^\dagger A P)^{-1}P^\dagger A)v||
    complex<double>* AV = mg_object->get_storage(0)->check_out();
    complex<double>* PdagAV = mg_object->get_storage(1)->check_out();
    complex<double>* AcInvPdagAV = mg_object->get_storage(1)->check_out();
    complex<double>* PAcInvPdagAV = mg_object->get_storage(0)->check_out();
    for (i = 0; i < lats[0]->get_size_cv(); i++)
    {
      zero_vector(AV, lats[0]->get_size_cv());
      zero_vector(PdagAV, lats[1]->get_size_cv());
      zero_vector(AcInvPdagAV, lats[1]->get_size_cv());
      zero_vector(PAcInvPdagAV, lats[0]->get_size_cv());

      mg_object->get_stencil(0)->apply_M(AV, evecs[i]);
      mg_object->get_transfer(0)->restrict_f2c(AV, PdagAV);
      minv_vector_bicgstab_l(AcInvPdagAV, PdagAV, lats[1]->get_size_cv(), 1000, 1e-10, 6, apply_stencil_2D_M, (void*)mg_object->get_stencil(1));
      mg_object->get_transfer(0)->prolong_c2f(AcInvPdagAV, PAcInvPdagAV);

      onePAPA.push_back(sqrt(diffnorm2sq(evecs[i], PAcInvPdagAV, lats[0]->get_size_cv())/norm2sq(evecs[i], lats[0]->get_size_cv())));
    }

    mg_object->get_storage(0)->check_in(AV);
    mg_object->get_storage(1)->check_in(PdagAV);
    mg_object->get_storage(1)->check_in(AcInvPdagAV);
    mg_object->get_storage(0)->check_in(PAcInvPdagAV);

    for (i = 0; i < lats[0]->get_size_cv(); i++)
    {
      std::cout << "[QMG-OVERLAP]: " << i <<
                " " << real(eigs[i]) <<
                " + I " << imag(eigs[i]) <<
                " " << abs(eigs[i]) << 
                " | " << onePP[i] <<
                " | " << onePAPA[i] << "\n";
    }


    delete[] eigs;
    for (i = 0; i < lats[0]->get_size_cv(); i++)
    {
      deallocate_vector(&evecs[i]);
    }
    delete[] evecs;

  }

  ///////////////
  // Clean up. //
  ///////////////

  deallocate_vector(&gauge_field);

  // Delete MultigridMG.
  delete mg_object;

  // Delete transfer objects.
  for (i = 0; i < n_refine; i++)
  {
    delete transfer_objs[i];
  }
  delete[] transfer_objs;

  // Delete stencil.
  delete wilson_op;

  // Delete coarsest solve objects.
  delete coarsest_solve_obj;

  // Delete level solve objects.
  for (i = 0; i < n_refine; i++)
  {
    delete level_solve_objs[i];
  }
  delete[] level_solve_objs;

  // Delete lattices.
  for (i = 0; i <= n_refine; i++)
  {
    delete lats[i];
  }
  delete[] lats; 

  return 0;
}

