// Copyright (c) Evan S Weinberg 2017
// Generate non-compact quenched U(1) gauge fields
// via heatbath, measure the would-be goldstone
// correlator at zero momentum to extract the mass...
// Using multigrid. Hey, we made it. 

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <random>
#include <vector>
#include <sstream>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"
#include "verbosity/verbosity.h"
#include "inverters/inverter_struct.h"
#include "inverters/generic_bicgstab_l.h"
#include "inverters/generic_gcr.h"
#include "inverters/generic_gcr_var_precond.h"

// QMG
#include "lattice/lattice.h"
#include "transfer/transfer.h"
#include "stencil/stencil_2d.h"
#include "multigrid/stateful_multigrid.h"

#include "u1/u1_utils.h"
#include "operators/wilson.h"
#include "reductions/reductions.h"

int main(int argc, char** argv)
{
  cout << setiosflags(ios::fixed) << setprecision(6);

  // Iterator
  int i,j,k,n;

  // Random number generator
  std::mt19937 generator (1337u);

  // A string variable
  std::string string_tmp = "";

  // How verbose? False -> Not very verbose, True -> WOW TOO MUCH DETAIL
  bool how_verbose = false; 

  ////////////////////////
  // LATTICE DIMENSIONS //
  ////////////////////////

  // Some basic fields.
  const int x_len = 64;
  const int y_len = 64;
  const int dof = Wilson2D::get_dof();

  /////////////////////////
  // HEATBATH PARAMETERS //
  /////////////////////////

  // beta
  const double beta = 6.0; 
  // How many updates to do between measurements?
  const int n_update = 100;
  // How many thermalization updates do we do before we start measuring?
  const int n_therm = 1000;
  // How many updates do we do in total?
  const int n_max = 4000;

  ////////////////////////
  // FERMION PARAMETERS //
  ////////////////////////

  // mass. For beta = 6.0, m_critical ~ -0.0706(15),
  //       extracted from 100000 cfgs (separated by 100)
  //       on a 32^2 volume.
  const double mass = -0.01;

  ///////////////////////
  // SOLVER PARAMETERS //
  ///////////////////////

  // Define inverter parameters, inversion struct.
  const int max_iter = 4000; 
  const double tol = 1e-10;
  const int restart_freq = 32; // since outer solver is GCR 
  inversion_info invif;

  inversion_verbose_struct* verb = new inversion_verbose_struct(how_verbose ? VERB_DETAIL : VERB_SUMMARY, std::string("[QMG-MG-SOLVE-INFO]: Level 0 "));
  // Hack: This isn't "faithful" to how preconditioner verbosity should work.
  verb->precond_verbosity = how_verbose ? VERB_SUMMARY : VERB_NONE; // set to VERB_SUMMARY to get output. 

  ///////////////////////////////////////
  // NULL VECTOR GENERATION PARAMETERS //
  ///////////////////////////////////////

  // We currently generate null vectors algebraically with BiCGstab-L.
  // These are "magic values" that empirically work well.
  const int null_max_iter = 500;
  const double null_tol = 5e-5;
  const int null_bicgstab_l = 6; // Use BiCGstab-6, specifically.

  // We'll build the verbosity string on the fly. 
  inversion_verbose_struct* null_verb = new inversion_verbose_struct(how_verbose ? VERB_SUMMARY : VERB_NONE, std::string("[QMG-NULL-GEN]: Level # Null Vector # "));

  //////////////////////////
  // MULTIGRID PARAMETERS //
  //////////////////////////

  // Blocking size.
  const int x_block = 4;
  const int y_block = 4;

  // Number of times to refine. KNOWN BUG: n_refine = 0 gives a segfault.
  const int n_refine = 3; // (64 -> 16 -> 4 -> 1)

  // Number of null vectors. This could vary from level to level,
  // but we just use the same number of each level for now.
  const int coarse_dof = 8; // generate 4, parity proj gives 8.

  // What to do for pre-smoother, post-smoother, and coarse solve.
  // This code is currently written where we do the same type of
  // smoother and coarse solver is used on every level. There's 
  // support for changing it on each level, we're just keeping it simple here.
  // Some things (such as what solver we use) ARE currently hard coded.

  // Pre-smoother: GCR(2)
  const int n_pre_smooth = 2;
  const double pre_smooth_tol = 1e-15; // Force # iterations to be the stopping cond.

  // Post-smoother: GCR(2)
  const int n_post_smooth = 2;
  const double post_smooth_tol = 1e-15; // Force # iterations to be the stopping cond.

  // Intermediate solver: GCR(32)
  const int inner_max_iter = 1000000; // never
  const double inner_tol = 0.2; // Force tolerance to be stopping condition.
  const int inner_restart = 32;

  // Coarsest solve
  const double coarsest_tol = 0.2;
  const int coarsest_max_iter = 1000;
  const int coarsest_restart_freq = 32;

  //////////////////////////
  // INITIAL GAUGE FIELDS //
  //////////////////////////

  // Create a lattice object for the u(1) gauge fields.
  Lattice2D* lat_gauge = new Lattice2D(x_len, y_len, 1);

  // Allocate a gauge field
  complex<double>* gauge_field = allocate_vector<complex<double>>(lat_gauge->get_size_gauge());

  // Also need to keep around phases. If you truncate the phases to [-pi, pi),
  // you don't properly sample the gaussian distribution.
  double* phases = allocate_vector<double>(lat_gauge->get_size_gauge());

  // Create a unit gauge field -> zero phases.
  zero_vector(phases, lat_gauge->get_size_gauge());

  // Create the compact links.
  polar_vector(phases, gauge_field, lat_gauge->get_size_gauge());


  /////////////////////
  // CREATE LATTICES //
  /////////////////////

  // Since we've fixed the number of refinements
  // and the block sizes, we can create all of the lattices now.

  Lattice2D** lats = new Lattice2D*[n_refine+1];
  lats[0] = new Lattice2D(x_len, y_len, dof);
  const int cv_size = lats[0]->get_size_cv(); // for convenience.

  // Create the coarse lattices.
  int curr_x_len = x_len;
  int curr_y_len = y_len;
  for (i = 1; i <= n_refine; i++)
  {
    // Update to new size.
    curr_x_len /= x_block;
    curr_y_len /= y_block;

    // Create a new lattice object.
    lats[i] = new Lattice2D(curr_x_len, curr_y_len, coarse_dof);
  }

  ///////////////////////////////
  // CREATE TOP WILSON STENCIL //
  ///////////////////////////////

  // Since we can "update" the Wilson stencil with new links,
  // we only have to create it once. 
  Wilson2D* wilson = new Wilson2D(lats[0], mass, gauge_field);

  //////////////////////////////////////////////////
  // CREATE NAMED VARIABLES FOR MULTIGRID OBJECTS //
  //////////////////////////////////////////////////

  // This code currently works by tearing down and rebuilding
  // the entire stack of coarse operators each time we perform
  // measurements.

  // We'll just give some names to the variables we'll re-allocate
  // each time around...

  // Transfer objects that prolong/restrict you between levels.
  TransferMG** transfer_objs = new TransferMG*[n_refine];
  for (i = 0; i < n_refine; i++)
    transfer_objs[i] = 0;

  // Stateful MG object: holds all transfer objects, exposes operators,
  // builds coarse stencils, maintains managed memory. It also includes all
  // variables for an MG solve, such as which level we're currently solving.
  StatefulMultigridMG* mg_object = 0;

  // Structure which contains the smoothing and coarse solve
  // parameters on each level. Even though we've hard-coded
  // this to be the same on each level, we'll define it
  // inline below. 
  StatefulMultigridMG::LevelSolveMG** level_solve_objs = new StatefulMultigridMG::LevelSolveMG*[n_refine];

  // Structure which defines the coarsest solve. This is separate from
  // level_solve_objs in case, for example, you want to do the
  // lowest level solve with the normal equation.
  StatefulMultigridMG::CoarsestSolveMG* coarsest_solve_obj = new StatefulMultigridMG::CoarsestSolveMG;
  coarsest_solve_obj->coarsest_stencil_app = QMG_MATVEC_ORIGINAL;
  coarsest_solve_obj->coarsest_tol = coarsest_tol;
  coarsest_solve_obj->coarsest_iters = coarsest_max_iter;
  coarsest_solve_obj->coarsest_restart_freq = coarsest_restart_freq;

  ////////////////////////////
  // CREATE FERMION VECTORS //
  ////////////////////////////

  // Create vectors to store sources and propagators.
  complex<double>* src = allocate_vector<complex<double> >(cv_size);
  complex<double>* prop = allocate_vector<complex<double> >(cv_size);

  // Zero the src, drop a point.
  zero_vector(src, cv_size);
  src[lats[0]->cv_coord_to_index(0,0,0)] = 1.0;

  // Create a non-zero initial guess (I get nans when I use a point?)
  //gaussian(prop, cv_size, generator);

  //////////////////////////////////////////
  // CREATE ACCUMULATORS FOR MEASUREMENTS //
  //////////////////////////////////////////

  // Count number of measurements.
  int count = 0;

  // Create a place to accumulate the plaquette.
  double plaq = 0.0;
  double plaq_sq = 0.0;

  // Create a place to accumulate the would-be pion correlator. 
  double pion[y_len];
  double pion_sq[y_len];
  double pion_up[y_len];
  double pion_down[y_len];
  for (j = 0; j < y_len; j++)
  {
    pion[j] = 0.0;
    pion_sq[j] = 0.0;
  }

  // Do an initial measurement of the plaquette and topology.
  n = 0;
  cout << "[QMG-GAUGE]: "<< n << " " << get_plaquette_u1(gauge_field, lat_gauge) << " " << get_topo_u1(gauge_field, lat_gauge) << "\n";

  //////////////////////////////
  // BEGIN THE HEATBATH CYCLE //
  //////////////////////////////

  for (n = n_update; n < n_max; n+=n_update)
  {
    cout << "\n[QMG-MG]: Starting iteration " << n << "\n";
    // Perform non-compact update.
    heatbath_noncompact_update(phases, lat_gauge, beta, n_update, generator);

    // TO DO: Add an instanton + accept/reject sweep.
    // We don't sample topology well at beta = 6.0. 

    // Get compact links for the Wilson operator. 
    polar_vector(phases, gauge_field, lat_gauge->get_size_gauge());

    // Always measure the plaquette and topology.
    double plaq_tmp = std::real(get_plaquette_u1(gauge_field, lat_gauge));
    cout << "[QMG-GAUGE]: Iter " << n << " Plaq " << plaq_tmp << " Topo " << get_topo_u1(gauge_field, lat_gauge) << "\n";

    // If we're done thermalizing, start measuring!
    if (n > n_therm)
    {
      plaq += plaq_tmp;
      plaq_sq += plaq_tmp*plaq_tmp;

      // Update the Wilson operator.
      wilson->update_links(gauge_field);

      ////////////////////////////
      // BUILD MULTIGRID STACK! //
      ////////////////////////////

      // This is only so illuminating, and could really be packed in a separate
      // function. For the uninterested, skip to line 441.

      cout << "[QMG-MG]: Building coarse operator\n";

      // Create a fresh MultigridMG object, push the top level on it.
      mg_object = new StatefulMultigridMG(lats[0], wilson, coarsest_solve_obj);

      // Generate null vectors, build coarse operator
      // on each level. 
      for (i = 1; i <= n_refine; i++)
      {
        // For a given 'i', 'i-1' is the fine index, 'i' is the coarse index.
        int fine_idx = i-1;
        int coarse_idx = i;

        // Allocate space for null vectors. These get copied into
        // the local memory of each transfer object, so we can
        // destroy them at the end of the loop.
        complex<double>** null_vectors = new complex<double>*[lats[coarse_idx]->get_nc()];

        // Generate coarse_dof/2 null vectors. We currently build
        // them by relaxing on random sources via
        // solving the residual equation with BiCGstab-6.
        for (j = 0; j < lats[coarse_idx]->get_nc()/2; j++)
        {
          // Allocate and zero a new null vector.
          null_vectors[j] = allocate_vector<complex<double>>(lats[fine_idx]->get_size_cv());
          zero_vector(null_vectors[j], lats[fine_idx]->get_size_cv());

          // The MultigridMG object contains a memory manager. 
          // This avoids performing more allocations and deallocations
          // than needed.

          // Check out a vector, fill with random numbers.
          complex<double>* rand_guess = mg_object->get_storage(fine_idx)->check_out();
          gaussian(rand_guess, lats[fine_idx]->get_size_cv(), generator);

          // Check out a vector for the residual equation solve,
          // zero it out, form the residual r = -A rand_guess.
          complex<double>* Arand_guess = mg_object->get_storage(fine_idx)->check_out();
          zero_vector(Arand_guess, lats[fine_idx]->get_size_cv());
          mg_object->get_stencil(fine_idx)->apply_M(Arand_guess, rand_guess);
          cax(-1.0, Arand_guess, lats[fine_idx]->get_size_cv());

          // Solve the residual equation. A e = r.
          // "apply_stencil_2D_M" is a special
          // wrapper function that, in this case, knows to apply the current
          // fine stencil when generating the null vector. 
          null_verb->verb_prefix = "[QMG-NULL-GEN]: Level " + to_string(fine_idx) + " Null Vector " + to_string(j) + " ";
          minv_vector_bicgstab_l(null_vectors[j], Arand_guess, lats[fine_idx]->get_size_cv(),
            null_max_iter, null_tol, null_bicgstab_l,
            apply_stencil_2D_M, (void*)mg_object->get_stencil(fine_idx), null_verb);

          // Undo the residual equation null_vector = rand_guess + e.
          cxpy(rand_guess, null_vectors[j], lats[fine_idx]->get_size_cv());

          // Check managed vectors back in.
          mg_object->get_storage(fine_idx)->check_in(rand_guess);
          mg_object->get_storage(fine_idx)->check_in(Arand_guess);

          // Orthogonalize against previous vectors.
          for (k = 0; k < j; k++)
            orthogonal(null_vectors[j], null_vectors[k], lats[fine_idx]->get_size_cv());
        }

        // Once we have coarse_dof/2 null vectors, we split them into "up" and 
        // "down" chiral projectors, preserving the \gamma_5 Hermiticity of
        // the coarse op.
        for (j = 0; j < lats[coarse_idx]->get_nc()/2; j++)
        {
          // Allocate a new null vector.
          null_vectors[j+lats[coarse_idx]->get_nc()/2] = allocate_vector<complex<double>>(lats[fine_idx]->get_size_cv());

          // Perform chiral projection, putting the "down" projection into the
          // new vector and keeping the "up" projection into the first vector.
          // Since a chiral projection means something different for different
          // fermion discretizations (and for the coarse operator), we
          // just let the stencil manage this as a design choice.
          mg_object->get_stencil(fine_idx)->chiral_projection_both(null_vectors[j], null_vectors[j+lats[coarse_idx]->get_nc()/2]);

          // Normalize vectors.
          normalize(null_vectors[j], lats[fine_idx]->get_size_cv());
          normalize(null_vectors[j+lats[coarse_idx]->get_nc()/2], lats[fine_idx]->get_size_cv());
        }

        // Create a new transfer object. Transfer objects are assigned
        // to the fine level.
        // Arg 1: Fine lattice.
        // Arg 2: Coarse lattice.
        // Arg 3: Null vectors.
        // Arg 4: Should we perform the block orthonormalization?
        //        At some point, we'll support loading/saving null
        //        vectors, so there could be a reason why the vectors
        //        are already block orthonormalized.
        // Arg 5: Ignore (specifies if we should save Cholesky decomp of block ortho)
        // Arg 6: How we doubled the number of null vectors.
        transfer_objs[fine_idx] = new TransferMG(lats[fine_idx], lats[coarse_idx],
                                    null_vectors, true, false, QMG_DOUBLE_PROJECTION);

        // Fill a new LevelSolveMG object for the new level.
        // This defines how an outer solve is done with level 'fine_idx',
        // as well as how it's used to smooth.
        // When fine_idx is 0, the intermediate values are ignored,
        // and are instead defered to when the inverse is called.
        level_solve_objs[fine_idx] = new StatefulMultigridMG::LevelSolveMG;
        level_solve_objs[fine_idx]->fine_stencil_app = QMG_MATVEC_ORIGINAL;
        level_solve_objs[fine_idx]->intermediate_tol = inner_tol;
        level_solve_objs[fine_idx]->intermediate_iters = inner_max_iter;
        level_solve_objs[fine_idx]->intermediate_restart_freq = inner_restart;
        level_solve_objs[fine_idx]->pre_tol = pre_smooth_tol;
        level_solve_objs[fine_idx]->pre_iters = n_pre_smooth;
        level_solve_objs[fine_idx]->post_tol = post_smooth_tol;
        level_solve_objs[fine_idx]->post_iters = n_post_smooth;

        // Push a new level on the multigrid object! 
        // Arg 1: Coarse lattice
        // Arg 2: New transfer object (between new and prev lattice)
        // Arg 3: New LevelSolve object, describing outer solves and smooths.
        // Arg 4: Should we construct the coarse stencil?
        //          At some point, we'll support loading/saving coarse stencils.
        //          For now, this should always be true.
        // Arg 5: Is the operator chiral? (True for Wilson)
        // Arg 6: What should we construct the coarse stencil from?
        //          At some point, we'll support coarsening 
        //          block preconditioned operators. 
        // Arg 7: Optionally store non-block-orthonormalized null vectors.
        //          We don't need this for now, but it could be useful
        //          if we wanted to store null vectors and refine them
        //          as the gauge field evolves.
        mg_object->push_level(lats[coarse_idx], transfer_objs[fine_idx], level_solve_objs[fine_idx], true,
          true, MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL, null_vectors);

        // The transfer object and multigrid object create their own
        // local copy of the null vectors, so we can clean this up here.
        for (j = 0; j < lats[coarse_idx]->get_nc(); j++)
          deallocate_vector(&null_vectors[j]);
        delete[] null_vectors;
      }

      ///////////////////////
      // PERFORM INVERSION //
      ///////////////////////

      cout << setiosflags(ios::scientific) << setprecision(6);

      // We need to perform two inversions: one for each parity component.

      ////////////////
      // Parity up: //
      ////////////////

      std::cout << "[QMG-MG]: Parity UP Solve\n";

      src[lats[0]->cv_coord_to_index(0,0,0)] = 1.0;
      src[lats[0]->cv_coord_to_index(0,0,1)] = 0.0;

      invif = minv_vector_gcr_var_precond_restart(prop, src, lats[0]->get_size_cv(),
              max_iter, tol, restart_freq,
              apply_stencil_2D_M, (void*)mg_object->get_stencil(0),
              StatefulMultigridMG::mg_preconditioner, (void*)mg_object, verb); 

      // Compute the norm2sq, update into accumulator.
      norm2sq_cv_timeslice(pion_up, prop, lats[0]);

      // Fold the pion.
      for (j = 1; j < y_len/2; j++)
      {
        double tmp = 0.5*(pion_up[j] + pion_up[y_len-j]);
        pion_up[j] = pion_up[y_len-j] = tmp;
      }

      //////////////////
      // Parity down: //
      //////////////////

      std::cout << "[QMG-MG]: Parity DOWN Solve\n";

      src[lats[0]->cv_coord_to_index(0,0,0)] = 0.0;
      src[lats[0]->cv_coord_to_index(0,0,1)] = 1.0;
      //constant_vector(prop, 1.0, cv_size); // For some reason we can't recycle the
                                           // old solution as an initial guess.
      invif = minv_vector_gcr_var_precond_restart(prop, src, lats[0]->get_size_cv(),
              max_iter, tol, restart_freq,
              apply_stencil_2D_M, (void*)mg_object->get_stencil(0),
              StatefulMultigridMG::mg_preconditioner, (void*)mg_object, verb); 

      // Compute the norm2sq, update into accumulator.
      norm2sq_cv_timeslice(pion_down, prop, lats[0]);

      // Fold the pion.
      for (j = 1; j < y_len/2; j++)
      {
        double tmp = 0.5*(pion_down[j] + pion_down[y_len-j]);
        pion_down[j] = pion_down[y_len-j] = tmp;
      }

      // And we're done.

      // Accumulate pions.
      for (j = 0; j < y_len; j++)
      {
        pion[j] += pion_up[j] + pion_down[j];
        pion_sq[j] += (pion_up[j] + pion_down[j])*(pion_up[j] + pion_down[j]);
      }

      // Reset the output precision.
      cout << setiosflags(ios::fixed) << setprecision(6);

      // And update the counter. 
      count++;

      ////////////////////////////////
      // CLEAN UP MULTIGRID OBJECTS //
      ////////////////////////////////

      // We could optionally keep them around if we thought
      // the solver could be a good preconditioner for multiple
      // updates... not likely with heatbath. 

      delete mg_object; mg_object = 0;
      for (i = 0; i < n_refine-1; i++)
      {
        delete level_solve_objs[i]; level_solve_objs[i] = 0;
      }
      for (i = 0; i < n_refine; i++)
      {
        delete transfer_objs[i]; transfer_objs[i] = 0;
      }

    }
  }

  cout << "[QMG-GAUGE-FINAL]: The plaquette is " << plaq/count << " +/- " << sqrt((plaq_sq/count - plaq*plaq/(count*count))/(count)) << "\n";

  cout << "[QMG-BEGIN-PION]\n";
  // Print the pion correlator.
  for (j = 0; j < y_len; j++)
    cout << j << " " << pion[j]/count << " +/- " << sqrt((pion_sq[j]/count - pion[j]*pion[j]/(count*count))/(count)) << "\n";
  cout << "[QMG-END-PION]\n";

  cout << "[QMG-BEGIN-PION-EFFMASS]\n";
  // Print the pion effmass.
  for (j = 1; j < y_len-1; j++)
  {
    cout << j << " " << std::acosh((pion[j+1]+pion[j-1])/(2.0*pion[j])) << "\n";
  }
  cout << "[QMG-END-PION-EFFMASS]\n";


  // Clean up.
  deallocate_vector(&src);
  deallocate_vector(&prop);
  deallocate_vector(&phases);
  deallocate_vector(&gauge_field);

  delete wilson;
  delete verb; 
  delete null_verb; 

  // Delete coarse solve object.
  delete coarsest_solve_obj;

  // Delete storage for level solve objects.
  delete[] level_solve_objs;
  
  // Delete storage for transfer objects.
  delete[] transfer_objs;

  // Delete lattices.
  for (i = 0; i <= n_refine; i++)
  {
    delete lats[i];
  }
  delete[] lats; 
  delete lat_gauge; 

}

