// Copyright (c) 2017 Evan S Weinberg
// A test of a geometric K-cycle for the free laplace equation.
// Also includes a test of popping a level.

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

// QMG
#include "lattice/lattice.h"
#include "transfer/transfer.h"
#include "stencil/stencil_2d.h"
#include "multigrid/stateful_multigrid.h"

// Grab laplace operator (we just use unit gauge fields -> free laplace)
#include "operators/gaugedlaplace.h"
#include "u1/u1_utils.h"




int main(int argc, char** argv)
{
  // Iterators.
  int i;

  // Set output precision to be long.
  cout << setprecision(20);

  // Random number generator.
  std::mt19937 generator (1337u);

  // Basic information for fine level.
  const int x_len = 128;
  const int y_len = 128;
  const int dof = GaugedLaplace2D::get_dof(); 

  // Information on the Laplace operator.
  const double mass = 0.001;

  // Blocking size.
  const int x_block = 2;
  const int y_block = 2;

  // Number of null vectors.
  // We're just doing the free laplace, so we only need one.
  const int coarse_dof = 1;

  // How many times to refine. 
  const int n_refine = 6; // (64 -> 32 -> 16 -> 8 -> 4 -> 2)

  // Information about the outermost solve.
  const double tol = 1e-8; 
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
  verb.verb_prefix = "[QMG-MG-SOLVE-INFO]: Level 0 ";
  verb.precond_verbosity = VERB_DETAIL; //VERB_DETAIL;
  verb.precond_verb_prefix = "Prec ";

  // Create a lattice object for the fine lattice.
  Lattice2D** lats = new Lattice2D*[n_refine+1];
  lats[0] = new Lattice2D(x_len, y_len, dof);

  // Create a unit gauged laplace stencil.
  complex<double>* unit_gauge = allocate_vector<complex<double>>(lats[0]->get_size_gauge());
  unit_gauge_u1(unit_gauge, lats[0]);
  // Need to change relaxation params for gauged laplace.
  //read_gauge_u1(unit_gauge, lats[0], "../common_cfgs_u1/l64t64b60_heatbath.dat");
  GaugedLaplace2D* laplace_op = new GaugedLaplace2D(lats[0], mass*mass, unit_gauge);

  // Prepare level solve objects for the top level.
  StatefulMultigridMG::LevelSolveMG** level_solve_objs = new StatefulMultigridMG::LevelSolveMG*[n_refine];

  // Prepare coarsest solve object for the coarsest level.
  StatefulMultigridMG::CoarsestSolveMG* coarsest_solve_obj = new StatefulMultigridMG::CoarsestSolveMG;
  coarsest_solve_obj->coarsest_stencil_app = QMG_MATVEC_ORIGINAL;
  coarsest_solve_obj->coarsest_tol = coarsest_tol;
  coarsest_solve_obj->coarsest_iters = coarsest_max_iter;
  coarsest_solve_obj->coarsest_restart_freq = coarsest_restart_freq;

  // Create a StatefulMultigridMG object, push top level onto it!
  StatefulMultigridMG* mg_object = new StatefulMultigridMG(lats[0], laplace_op, coarsest_solve_obj);

  // Create coarse lattices, unit null vectors, transfer objects.
  // Push into MultigridMG object. 
  int curr_x_len = x_len;
  int curr_y_len = y_len;
  TransferMG** transfer_objs = new TransferMG*[n_refine];
  for (i = 1; i <= n_refine; i++)
  {
    // Update to the new lattice size.
    curr_x_len /= x_block;
    curr_y_len /= y_block;

    // Create a new lattice object.
    lats[i] = new Lattice2D(curr_x_len, curr_y_len, coarse_dof);

    // Create a new null vector. These are copied into local memory in the
    // transfer object, so we can create and destroy these in this loop.
    complex<double>* null_vector = allocate_vector<complex<double>>(lats[i-1]->get_size_cv());
    constant_vector(null_vector, 1.0, lats[i-1]->get_size_cv());

    // Create and populate a transfer object.
    // Fine lattice, coarse lattice, null vector(s), perform the block ortho.
    transfer_objs[i-1] = new TransferMG(lats[i-1], lats[i], &null_vector, true);

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
    // Arg 4: Should we construct the coarse stencil?
    // Arg 5: Does the operator have a sense of chirality?
    // Arg 6: What should we construct the coarse stencil from? (Not relevant yet.)
    // Arg 7: Non-block-orthogonalized null vector.
    mg_object->push_level(lats[i], transfer_objs[i-1], level_solve_objs[i-1],  true, false, MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL, &null_vector);

    // Clean up local vector, since they get copied in.
    deallocate_vector(&null_vector);
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

  cout << "Begin " << n_refine << " level solve!\n";
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

  ///////////////////////////
  // Test popping a level! //
  ///////////////////////////

  mg_object->pop_level();

  cout << "\n\nBegin " << n_refine-1 << " level solve!\n";

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

  /////////////////////////////////
  // Test popping another level! //
  /////////////////////////////////

  mg_object->pop_level();

  cout << "\n\nBegin " << n_refine-2 << " level solve!\n";

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

  ///////////////////
  // And one last! //
  ///////////////////

  mg_object->pop_level();

  cout << "\n\nBegin " << n_refine-3 << " level solve!\n";

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


  ///////////////
  // Clean up. //
  ///////////////

  // Check vectors back in.
  mg_object->check_in(Ax, 0);
  mg_object->check_in(x, 0);
  mg_object->check_in(b, 0);

  deallocate_vector(&unit_gauge);

  // Delete MultigridMG.
  delete mg_object;

  // Delete transfer objects.
  for (i = 0; i < n_refine; i++)
  {
    delete transfer_objs[i];
  }
  delete[] transfer_objs;

  // Delete stencil.
  delete laplace_op;

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

