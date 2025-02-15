// Copyright (c) 2017 Evan S Weinberg
// A test of a K-cycle for the interacting
// Wilson operator. Null vector generation
// based on https://arxiv.org/pdf/1307.6101.pdf.

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
#include "inverters/generic_richardson.h"
#include "inverters/generic_minres.h"
#include "interfaces/arpack/generic_arpack.h"

// QMG
#include "lattice/lattice.h"
#include "transfer/transfer.h"
#include "stencil/stencil_2d.h"
#include "multigrid/stateful_multigrid.h"

// Grab Wilson operator (we just use unit gauge fields -> free laplace)
#include "operators/wilson.h"
#include "u1/u1_utils.h"

// Build new coarse levels by restricting+relaxing test vectors from upper levels.
// See better documentation below.
TransferMG* build_coarse_by_restrict(StatefulMultigridMG* mg_object, complex<double>*** test_vectors, int fine_level, Lattice2D* coarse_lat, StatefulMultigridMG::LevelSolveMG* new_level_solve, bool fresh_build, std::mt19937& generator, inversion_verbose_struct verb);

int main(int argc, char** argv)
{
  if (argc != 6)
  {
    std::cout << "Error: ./wilson_kcycle expects five arguments, L, mass, beta, n_refine, and n_setup. Try mass = -0.075 for beta 6.0.\n";
    return -1;
  }

  // Iterators.
  int i,j,k,m;

  // Check the spectrum?
  const bool do_spectrum = false;

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
  double mass = stod(argv[2]); // -0.075 is m_crit at beta = 6

  // Blocking size.
  const int x_block = 4;
  const int y_block = 4;

  // Number of null vectors.
  int coarse_dof = 8;

  // How many times to refine. 
  const int n_refine = stoi(argv[4]);

  // Number of setup iterations.
  const int n_setup = stoi(argv[5]);

  // Information about the outermost solve.
  const double tol = 1e-10; 
  const int max_iter = 1000;
  const int restart_freq = 64;

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

  // Create coarse lattices, transfer objects.
  // Push into MultigridMG object. 
  int curr_x_len = x_len;
  int curr_y_len = y_len;
  TransferMG** transfer_objs = new TransferMG*[n_refine];
  complex<double>*** test_vectors = new complex<double>**[n_refine]; // gross. I need better data structures.

  // Initial setup. 
  for (i = 0; i < n_refine; i++)
  {
    const int fine_idx = i; // Index the fine level.
    const int coarse_idx = i+1; // Index the coarse level.

    // Update to the new lattice size.
    curr_x_len /= x_block;
    curr_y_len /= y_block;

    // Create a new lattice object.
    lats[coarse_idx] = new Lattice2D(curr_x_len, curr_y_len, coarse_dof);

    // Create space for test vectors.
    test_vectors[fine_idx] = new complex<double>*[coarse_dof/2];
    for (j = 0; j < coarse_dof/2; j++)
    {
      test_vectors[fine_idx][j] = allocate_vector<complex<double> >(lats[fine_idx]->get_size_cv());
      zero_vector(test_vectors[fine_idx][j], lats[fine_idx]->get_size_cv());
    }

    // Zero out transfer object.
    transfer_objs[fine_idx] = 0;

    // Prepare a new LevelSolveMG object for each level.
    level_solve_objs[fine_idx] = new StatefulMultigridMG::LevelSolveMG;
    level_solve_objs[fine_idx]->fine_stencil_app = QMG_MATVEC_ORIGINAL;
    level_solve_objs[fine_idx]->intermediate_tol = 1e-10; //inner_tol;
    level_solve_objs[fine_idx]->intermediate_iters = 8; //inner_max_iter;
    level_solve_objs[fine_idx]->intermediate_restart_freq = 1024;//inner_restart_freq;
    level_solve_objs[fine_idx]->pre_tol = pre_smooth_tol;
    level_solve_objs[fine_idx]->pre_iters = n_pre_smooth;
    level_solve_objs[fine_idx]->post_tol = post_smooth_tol;
    level_solve_objs[fine_idx]->post_iters = n_post_smooth;
  }

  // Throw and relax initial test vectors.
  {
    const int fine_idx = 0;
    const int coarse_idx = 1;

    // Create a new null vectors. These are copied into local memory in the
    // transfer object, so we can create and destroy these in this loop.
    complex<double>** null_vectors = new complex<double>*[coarse_dof];

    // First level of setup. Smooth vectors.
    for (j = 0; j < coarse_dof/2; j++)
    {
      // Update verbosity string.
      verb.verb_prefix = "Level " + to_string(fine_idx) + " Init, Null Vector " + to_string(j) + " ";

      // Will become up chiral projection
      null_vectors[j] = allocate_vector<complex<double> >(lats[fine_idx]->get_size_cv());
      null_vectors[j+coarse_dof/2] = allocate_vector<complex<double> >(lats[fine_idx]->get_size_cv());
      zero_vector(null_vectors[j], lats[fine_idx]->get_size_cv());
      zero_vector(null_vectors[j+coarse_dof/2], lats[fine_idx]->get_size_cv());

      // Grab a temporary.
      complex<double>* temp_rand = mg_object->get_storage(fine_idx)->check_out();

      // Fill with random numbers on the top level.
      gaussian(temp_rand, lats[fine_idx]->get_size_cv(), generator);

      // Smooth with MR, 10 hits.
      //minv_vector_bicgstab_l(test_vectors[fine_idx][j], temp_rand, lats[fine_idx]->get_size_cv(), 100, 1e-10, 8, apply_stencil_2D_M, (void*)mg_object->get_stencil(fine_idx), &verb);
      invif = minv_vector_richardson(test_vectors[fine_idx][j], temp_rand, lats[fine_idx]->get_size_cv(), 10, 1e-10, 0.33, 250, apply_stencil_2D_M, (void*)mg_object->get_stencil(fine_idx), &verb);
      //invif = minv_vector_gcr(test_vectors[fine_idx][j], temp_rand, lats[fine_idx]->get_size_cv(), 10, 1e-10, apply_stencil_2D_M, (void*)mg_object->get_stencil(fine_idx), &verb);
      //invif = minv_vector_minres(test_vectors[fine_idx][j], temp_rand, lats[fine_idx]->get_size_cv(), 10, 1e-15, 0.66, apply_stencil_2D_M, (void*)mg_object->get_stencil(fine_idx), &verb);
      mg_object->add_tracker_count(QMG_DSLASH_TYPE_NULLVEC, invif.ops_count, fine_idx);
      mg_object->get_storage(fine_idx)->check_in(temp_rand);

      // Orthogonalize against previous vectors.
      for (k = 0; k < j; k++)
        orthogonal(test_vectors[fine_idx][j], test_vectors[fine_idx][k], lats[fine_idx]->get_size_cv());

      normalize(test_vectors[fine_idx][j], lats[fine_idx]->get_size_cv());
      copy_vector(null_vectors[j], test_vectors[fine_idx][j], lats[fine_idx]->get_size_cv());
      mg_object->get_stencil(fine_idx)->chiral_projection_both(null_vectors[j], null_vectors[j+coarse_dof/2]);
    }

    // Create a first transfer object.

    // Create and populate a transfer object.
    // Fine lattice, coarse lattice, null vector(s), perform the block ortho, don't save Cholesky, doubled by projection
    transfer_objs[fine_idx] = new TransferMG(lats[fine_idx], lats[coarse_idx], null_vectors, true, false, QMG_DOUBLE_PROJECTION);

    // Push a new level on the multigrid object! Also, save the global null vector.
    // Arg 1: New lattice
    // Arg 2: New transfer object (between new and prev lattice)
    // Arg 3: Level solve object (specifies how to do intermediate solves and smooths)
    // Arg 3: Should we construct the coarse stencil?
    // Arg 4: Is the operator chiral? (True for Wilson)
    // Arg 5: What should we construct the coarse stencil from? (Not relevant yet.)
    // Arg 6: Non-block-orthogonalized null vector.
    mg_object->push_level(lats[coarse_idx], transfer_objs[fine_idx], level_solve_objs[fine_idx], true, true, MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL, null_vectors);

    // Clean up a bit.
    for (j = 0; j < coarse_dof; j++)
      deallocate_vector(&null_vectors[j]);
    delete[] null_vectors;
  }

  // Prepare initial levels.
  for (i = 1; i < n_refine; i++)
  {
    const int fine_idx = i;
    const int coarse_idx = i+1;

    transfer_objs[fine_idx] = build_coarse_by_restrict(mg_object, test_vectors, fine_idx, lats[coarse_idx], level_solve_objs[fine_idx], true, generator, verb);
  }

  // Alright, initial setup is done. Let's do a setup update.
  for (m = 0; m < n_setup; m++)
  {
    for (i = 0; i < n_refine; i++)
    {    
      std::cout << "\n\n";

      const int fine_idx = i; // Index the fine level.
      const int coarse_idx = i+1; // Index the coarse level.

      // Create a new null vectors. These are copied into local memory in the
      // transfer object, so we can create and destroy these in this loop.
      complex<double>** null_vectors = new complex<double>*[coarse_dof];

      // Smooth each test vector using a K-cycle.
      for (j = 0; j < coarse_dof/2; j++)
      {
        verb.verb_prefix = "Level " + to_string(fine_idx) + " Update " + to_string(m) + ", Null Vector " + to_string(j) + " ";

        null_vectors[j] = allocate_vector<complex<double> >(lats[fine_idx]->get_size_cv());
        null_vectors[j+coarse_dof/2] = allocate_vector<complex<double> >(lats[fine_idx]->get_size_cv());
        zero_vector(null_vectors[j], lats[fine_idx]->get_size_cv());
        zero_vector(null_vectors[j+coarse_dof/2], lats[fine_idx]->get_size_cv());

        complex<double>* temp_rand = mg_object->get_storage(fine_idx)->check_out();

        // if on the top level, grab the old test vector. Otherwise restrict from a level up.
        if (i == 0)
        {
          copy_vector(temp_rand, test_vectors[fine_idx][j], lats[fine_idx]->get_size_cv());
        }
        else
        {
          zero_vector(temp_rand, lats[fine_idx]->get_size_cv());
          mg_object->get_transfer(fine_idx-1)->restrict_f2c(test_vectors[fine_idx-1][j], temp_rand);
        }

        zero_vector(test_vectors[fine_idx][j], lats[fine_idx]->get_size_cv());
        invif = minv_vector_gcr_var_precond(test_vectors[fine_idx][j], temp_rand, lats[fine_idx]->get_size_cv(),
              10, 1e-10,
              apply_stencil_2D_M, (void*)mg_object->get_stencil(fine_idx),
              StatefulMultigridMG::mg_preconditioner, (void*)mg_object, &verb); 
        mg_object->get_storage(fine_idx)->check_in(temp_rand);

        mg_object->add_tracker_count(QMG_DSLASH_TYPE_NULLVEC, invif.ops_count+1, fine_idx);

        // Update null vectors. 
        zero_vector(null_vectors[j], lats[fine_idx]->get_size_cv());
        zero_vector(null_vectors[j+coarse_dof/2], lats[fine_idx]->get_size_cv());

         // Orthogonalize against previous vectors.
        for (k = 0; k < j; k++)
          orthogonal(test_vectors[fine_idx][j], test_vectors[fine_idx][k], lats[fine_idx]->get_size_cv());

        normalize(test_vectors[fine_idx][j], lats[fine_idx]->get_size_cv());
        copy_vector(null_vectors[j], test_vectors[fine_idx][j], lats[fine_idx]->get_size_cv());
        mg_object->get_stencil(fine_idx)->chiral_projection_both(null_vectors[j], null_vectors[j+coarse_dof/2]);
        std::cout << "\n";
      }

      // Build a new transfer object,
      delete transfer_objs[fine_idx];
      transfer_objs[fine_idx] = new TransferMG(lats[fine_idx], lats[coarse_idx], null_vectors, true, false, QMG_DOUBLE_PROJECTION);

      // Update a level.
      mg_object->update_level(coarse_idx, lats[coarse_idx], transfer_objs[fine_idx], level_solve_objs[fine_idx], true, true, MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL, null_vectors);

      // Rebuild all lower levels.
      for (j = i+1; j < n_refine; j++)
      {
        const int fine_idx = j;
        const int coarse_idx = j+1;

        delete transfer_objs[fine_idx];
        transfer_objs[fine_idx] = build_coarse_by_restrict(mg_object, test_vectors, fine_idx, lats[coarse_idx], level_solve_objs[fine_idx], false, generator, verb);
      }

      for (j = 0; j < coarse_dof; j++)
        deallocate_vector(&null_vectors[j]);
      delete[] null_vectors;

      // Need to rebuild all lower levels, too. I guess I can just pop instead of updating.

      // Go down a level.
      if (i < n_refine-1)
        mg_object->go_coarser();
    }

    // Pop back up.
    for (i = 0; i < n_refine-1; i++)
      mg_object->go_finer();
  }

  // Shift tracker to null vector counts.
  for (i = 0; i <= n_refine; i++)
  {
    mg_object->shift_all_to_nullvec(i);
  }

  // Do the actual solve differently from the setup.
  // Initial setup. 
  for (i = 0; i < n_refine; i++)
  {
    const int fine_idx = i; // Index the fine level.
    const int coarse_idx = i+1; // Index the coarse level.

    level_solve_objs[fine_idx]->fine_stencil_app = QMG_MATVEC_ORIGINAL;
    level_solve_objs[fine_idx]->intermediate_tol = inner_tol;
    level_solve_objs[fine_idx]->intermediate_iters = inner_max_iter;
    level_solve_objs[fine_idx]->intermediate_restart_freq = inner_restart_freq;
    level_solve_objs[fine_idx]->pre_tol = pre_smooth_tol;
    level_solve_objs[fine_idx]->pre_iters = n_pre_smooth;
    level_solve_objs[fine_idx]->post_tol = post_smooth_tol;
    level_solve_objs[fine_idx]->post_iters = n_post_smooth;
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
  std::cout << "\n";
  verb.verb_prefix = "[QMG-MG-SOLVE-INFO]: Level 0 ";
  invif = minv_vector_gcr_var_precond_restart(x, b, lats[0]->get_size_cv(),
              max_iter, tol, restart_freq,
              apply_stencil_2D_M, (void*)mg_object->get_stencil(0),
              StatefulMultigridMG::mg_preconditioner, (void*)mg_object, &verb); 
  mg_object->add_tracker_count(QMG_DSLASH_TYPE_KRYLOV, invif.ops_count, 0);
  mg_object->add_iterations_count(invif.iter, 0);

  cout << "Multigrid " << (invif.success ? "converged" : "failed to converge")
          << " in " << invif.iter << " iterations with alleged tolerance "
          << sqrt(invif.resSq)/bnorm << ".\n";

  // Print stats for each level.
  for (i = 0; i < n_refine+1; i++)
  {
    std::cout << "[QMG-OPS-STATS]: Level " << i << " NullVec " << mg_object->get_tracker_count(QMG_DSLASH_TYPE_NULLVEC, i)
                                                << " PreSmooth "  << mg_object->get_tracker_count(QMG_DSLASH_TYPE_PRESMOOTH, i)
                                                << " Krylov " << mg_object->get_tracker_count(QMG_DSLASH_TYPE_KRYLOV, i)
                                                << " PostSmooth " << mg_object->get_tracker_count(QMG_DSLASH_TYPE_PRESMOOTH, i)
                                                << " Total " << mg_object->get_total_count(i)
                                                << "\n";
  }

  // Query average number of iterations on each level.
  std::cout << "\n";
  std::vector<double> avg_iter = mg_object->query_average_iterations();
  for (i = 0; i < n_refine+1; i++)
  {
    std::cout << "[QMG-ITER-STATS]: Level " << i << " AverageIters " << avg_iter[i] << "\n";
  }

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


// Build a new level by restricting then smoothing vectors
// from a level up.
// Arg 1: MG object.
// Arg 2: the current fine level.
// Arg 3: new coarse lattice. 
// Arg 4: level solve object
// Arg 5: if true: this is a fresh build of the new level. if false: need to do it via update.
// Arg 6: random number generator.
// Arg 7: verbosity object
TransferMG* build_coarse_by_restrict(StatefulMultigridMG* mg_object, complex<double>*** test_vectors, int fine_level, Lattice2D* coarse_lat, StatefulMultigridMG::LevelSolveMG* new_level_solve, bool fresh_build, std::mt19937& generator, inversion_verbose_struct verb)
{
  int j,k;

  const int fine_idx = fine_level;
  const int coarse_idx = fine_level+1;
  const int coarse_dof = coarse_lat->get_nc();
  const int fine_size_cv = mg_object->get_lattice(fine_idx)->get_size_cv();

  // Somewhere to solve inversion info.
  inversion_info invif;

  // Create a new null vectors. These are copied into local memory in the
  // transfer object, so we can create and destroy these in this loop.
  complex<double>** null_vectors = new complex<double>*[coarse_dof];

  // First level of setup. Smooth restricted vectors.
  for (j = 0; j < coarse_dof/2; j++)
  {
    // Update verbosity string.
    verb.verb_prefix = "Level " + to_string(fine_idx) + " Init, Null Vector " + to_string(j) + " ";

    // Will become up chiral projection
    null_vectors[j] = allocate_vector<complex<double> >(fine_size_cv);
    null_vectors[j+coarse_dof/2] = allocate_vector<complex<double> >(fine_size_cv);
    zero_vector(null_vectors[j], fine_size_cv);
    zero_vector(null_vectors[j+coarse_dof/2], fine_size_cv);

    // Grab a temporary.
    complex<double>* temp_rand = mg_object->get_storage(fine_idx)->check_out();

    // Fill with random numbers on the top level.
    gaussian(temp_rand, fine_size_cv, generator);

    // Smooth with MR, 10 hits.
    //minv_vector_bicgstab_l(test_vectors[fine_idx][j], temp_rand, fine_size_cv, 100, 1e-10, 8, apply_stencil_2D_M, (void*)mg_object->get_stencil(fine_idx), &verb);
    invif = minv_vector_richardson(test_vectors[fine_idx][j], temp_rand, fine_size_cv, 10, 1e-10, 0.33, 250, apply_stencil_2D_M, (void*)mg_object->get_stencil(fine_idx), &verb);
    //invif = minv_vector_gcr(test_vectors[fine_idx][j], temp_rand, fine_size_cv, 10, 1e-10, apply_stencil_2D_M, (void*)mg_object->get_stencil(fine_idx), &verb);
    //invif = minv_vector_minres(test_vectors[fine_idx][j], temp_rand, fine_size_cv, 10, 1e-15, 0.66, apply_stencil_2D_M, (void*)mg_object->get_stencil(fine_idx), &verb);
    mg_object->add_tracker_count(QMG_DSLASH_TYPE_NULLVEC, invif.ops_count, fine_idx);
    mg_object->get_storage(fine_idx)->check_in(temp_rand);

    // Orthogonalize against previous vectors.
    for (k = 0; k < j; k++)
      orthogonal(test_vectors[fine_idx][j], test_vectors[fine_idx][k], fine_size_cv);

    normalize(test_vectors[fine_idx][j], fine_size_cv);
    copy_vector(null_vectors[j], test_vectors[fine_idx][j], fine_size_cv);
    mg_object->get_stencil(fine_idx)->chiral_projection_both(null_vectors[j], null_vectors[j+coarse_dof/2]);
  }

  // Build a new transfer object.
  TransferMG* transfer_obj = new TransferMG(mg_object->get_lattice(fine_idx), coarse_lat, null_vectors, true);

  if (fresh_build)
  {
    // Push a new level on the multigrid object! Also, save the global null vector.
    // Arg 1: New lattice
    // Arg 2: New transfer object (between new and prev lattice)
    // Arg 3: Level solve object (specifies how to do intermediate solves and smooths)
    // Arg 3: Should we construct the coarse stencil?
    // Arg 4: Is the operator chiral? (True for Wilson)
    // Arg 5: What should we construct the coarse stencil from? (Not relevant yet.)
    // Arg 6: Non-block-orthogonalized null vector.
    mg_object->push_level(coarse_lat, transfer_obj, new_level_solve, true, true, MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL, null_vectors);
  }
  else
  {
    // Update a level.
    mg_object->update_level(coarse_idx, coarse_lat, transfer_obj, new_level_solve, true, true, MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL, null_vectors);
  }

  // Clean up a bit.
  for (j = 0; j < coarse_dof; j++)
    deallocate_vector(&null_vectors[j]);
  delete[] null_vectors;

  return transfer_obj;
}
