// Copyright (c) 2019 Evan S Weinberg
// A test of multigrid on g5 domain wall operator!

// Define NO_ARPACK in the Makefile if there's no arpack support.
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <random>
#include <vector>
#include <chrono>
#include <string.h>

using namespace std;

// Borrow dense matrix eigenvalue routines.

#include <Eigen/Dense>
using namespace Eigen;
typedef Matrix<std::complex<double>, Dynamic, Dynamic, ColMajor> cMatrix;

// QLINALG
#include "blas/generic_vector.h"
#include "inverters/generic_cg.h"
#include "inverters/generic_cr.h"
#include "inverters/generic_gcr.h"
#include "inverters/generic_gcr_var_precond.h"
#include "inverters/generic_bicgstab.h"
#include "inverters/generic_bicgstab_l.h"

// STILL A TEST! Lanczos
#include <functional>
#include "tests/lanczos_tests/operator.h"
#include "tests/lanczos_tests/poly_operator.h"
#include "tests/lanczos_tests/lanczos.h"
#include "tests/lanczos_tests/thick_deflate_lanczos.h"

// QMG
#include "lattice/lattice.h"
#include "transfer/transfer.h"
#include "stencil/stencil_2d.h"
#include "multigrid/stateful_multigrid.h"

// Grab wilson operator
#include "operators/wilson.h"
#include "u1/u1_utils.h"

// Utilities for tests
#include "util.h"

#define MAX_REFINEMENT 8 // [Maximum number of refinement levels]

// Utilities for generating near-null vectors
// Should get integrated into QMG
#include "nullvec_gen.h"

// For inputs
struct InputData
{
  // Properties of the lattice and gauge field
  int length = 32;
  double beta = 6.0;

  // Properties of the fermions
  double mass_wilson = -0.065;

  // Properties of MG solve
  int n_refine = 2;// [?]
  int blocksize[MAX_REFINEMENT] = { 4, 2, 2, 2, 2, 2, 2, 2 };// [blocksize on each level]
  int coarse_dof[MAX_REFINEMENT] = { 8, 12, 12, 12, 12, 12, 12, 12 };// [numbers of testvectors on each level]
  bool nulls_are_evecs[MAX_REFINEMENT] = { false, false, false, false,
                                                  false, false, false, false };

  bool do_normal_coarsest = false; // whether or not we do cgne on the coarsest level[?]

  bool preserve_eo = false; // test preserving e/o -> eye structure[It should not affect the main body]

  bool test_wilson_mg = true;

  // instantons
  double add_topology = 0.0;

  // Spectrum tests [what spectrum?]
  bool spectrum_wilson = false;
  bool spectrum_g5_wilson = false;
  bool spectrum_g5_alternative = false;
};

void usage()
{
  std::cout << "--help                                    Show this message.\n";
  std::cout << "--length [n]                              Length of both dimensions [default 32]\n";
  std::cout << "--beta [beta]                             Gauge field beta [default 6.0]\n";
  std::cout << "--mass-wilson [mass]                      Wilson mass (for near-null gen) [default -0.065]\n";
  std::cout << "                                            Note: (1.0,-0.267); (3.0,-0.126); (6.0,-0.082); (10.0,-0.057)\n";
  std::cout << "--n-refine [n]                            Number of times to coarsen [default 2]\n";
  std::cout << "--blocksize [level] [n]                   Amount to block level n [default 4 on level 0, 2 otherwise]\n";
  std::cout << "--nvec [level] [n]                        How many null vectors to generate on level n [default 8 on level 0, 12 otherwise]\n";
  std::cout << "--use-cgne-coarsest [yes/no]              Whether or not to use CGNE on the coarsest level [default no]\n";
  std::cout << "--preserve-eo [yes/no]                    Preserve even/odd from fine level onto coarser levels [default no]\n";
#ifndef NO_ARPACK  
  std::cout << "--nulls-are-evecs [level] [yes/no]        Use eigenvectors as near-null vectors [default no on all levels]\n";
#endif
  std::cout << "--test-wilson-mg [yes/no]                 Test Wilson MG [default yes]\n";

  // instanton tests
  std::cout << "--add-topology [n]                        Add an instanton with the given charge [default 0, i.e. node]\n";

  // spectrum tests
  std::cout << "--spectrum-wilson [yes/no]                Compute the wilson spectrum [default no]\n";
  std::cout << "--spectrum-g5-wilson [yes/no]             Compute the gamma5 wilson spectrum [default no]\n";
  std::cout << "--spectrum-g5-alternative [yes/no]        Power iterations flavor of lowest 20 eigenvalues of g5 spectrum [default no]\n";

  exit(0);
}

InputData parse_inputs(int argc, char** argv) {
  int i = 1;

  InputData inp;

  while (i < argc) {
    if (strcmp(argv[i],"--help") == 0) {
      usage();
    } else if (strcmp(argv[i],"--length") == 0) {
      if (i+1 < argc) {
        inp.length = stoi(argv[i+1]);
        i+=2;
        continue;
      }
    } else if (strcmp(argv[i], "--beta") == 0) {
      if (i+1 < argc) {
        inp.beta = atof(argv[i+1]);
        i+=2;
        continue;
      }
    } else if (strcmp(argv[i], "--mass-wilson") == 0) {
      if (i+1 < argc) {
        inp.mass_wilson = atof(argv[i+1]);
        i+=2;
        continue;
      }
    } else if (strcmp(argv[i], "--n-refine") == 0) {
      if (i+1 < argc) {
        inp.n_refine = atoi(argv[i+1]);
        i+=2;
        continue;
      }
    } else if (strcmp(argv[i], "--blocksize") == 0) {
      if (i+2 < argc) {
        inp.blocksize[atoi(argv[i+1])] = atoi(argv[i+2]);
        i+=3;
        continue;
      }
    } else if (strcmp(argv[i], "--nvec") == 0) {
      if (i+2 < argc) {
        inp.coarse_dof[atoi(argv[i+1])] = atoi(argv[i+2]);
        i+=3;
        continue;
      }
    } else if (strcmp(argv[i], "--use-cgne-coarsest") == 0) {
      if (i+1 < argc) {
        if (strcmp(argv[i+1], "yes") == 0) {
          inp.do_normal_coarsest = true;
          i+=2;
          continue;
        } else if (strcmp(argv[i+1],"no") == 0) {
          inp.do_normal_coarsest = false;
          i+=2;
          continue;
        }
      }
    } else if (strcmp(argv[i], "--preserve-eo") == 0) {
      if (i+1 < argc) {
        if (strcmp(argv[i+1], "yes") == 0) {
          inp.preserve_eo = true;
          i+=2;
          continue;
        } else if (strcmp(argv[i+1],"no") == 0) {
          inp.preserve_eo = false;
          i+=2;
          continue;
        }
      }
    }
#ifndef NO_ARPACK
     else if (strcmp(argv[i], "--nulls-are-evecs") == 0) {
      if (i+2 < argc) {
        if (strcmp(argv[i+2], "yes") == 0) {
          inp.nulls_are_evecs[atoi(argv[i+1])] = true;
          i+=3;
          continue;
        } else if (strcmp(argv[i+2],"no") == 0) {
          inp.nulls_are_evecs[atoi(argv[i+1])] = false;
          i+=3;
          continue;
        }
      }
    }
#endif
    else if (strcmp(argv[i], "--test-wilson-mg") == 0) {
      if (i+1 < argc) {
        if (strcmp(argv[i+1], "yes") == 0) {
          inp.test_wilson_mg = true;
          i+=2;
          continue;
        } else if (strcmp(argv[i+1],"no") == 0) {
          inp.test_wilson_mg = false;
          i+=2;
          continue;
        }
      }
    } else if (strcmp(argv[i],"--add-topology") == 0) {
      if (i+1 < argc) {
        inp.add_topology = atof(argv[i+1]);
        i+=2;
        continue;
      }
    } else if (strcmp(argv[i], "--spectrum-wilson") == 0) {
      if (i+1 < argc) {
        if (strcmp(argv[i+1], "yes") == 0) {
          inp.spectrum_wilson = true;
          i+=2;
          continue;
        } else if (strcmp(argv[i+1],"no") == 0) {
          inp.spectrum_wilson = false;
          i+=2;
          continue;
        }
      }
    } else if (strcmp(argv[i], "--spectrum-g5-wilson") == 0) {
      if (i+1 < argc) {
        if (strcmp(argv[i+1], "yes") == 0) {
          inp.spectrum_g5_wilson = true;
          i+=2;
          continue;
        } else if (strcmp(argv[i+1],"no") == 0) {
          inp.spectrum_g5_wilson = false;
          i+=2;
          continue;
        }
      }
    } else if (strcmp(argv[i], "--spectrum-g5-alternative") == 0) {
      if (i+1 < argc) {
        if (strcmp(argv[i+1], "yes") == 0) {
          inp.spectrum_g5_alternative = true;
          i+=2;
          continue;
        } else if (strcmp(argv[i+1],"no") == 0) {
          inp.spectrum_g5_alternative = false;
          i+=2;
          continue;
        }
      }
    }
    
    // If we made it here, there was an error.
    usage();
  }

  return inp;
}

int main(int argc, char** argv)
{

  // Grab inputs
  InputData inp = parse_inputs(argc,argv);

  // Iterators.
  int i,j;

  // Set output precision to be long.
  cout << setprecision(20);

  // Random number generator.
  std::mt19937 generator (1337u);

  // Basic information for fine level.
  const int x_len = inp.length;
  const int y_len = inp.length;
  const double beta = inp.beta;
  const int dof = Wilson2D::get_dof();

  // Mass of the Wilson operator
  double mass = inp.mass_wilson;

  // Should we just look at the free (unit) fields?
  bool do_free = (beta > 1000.0) ? true : false; 


  // Do we test Wilson MG (to make sure we generated
  // good near-null vecs)
  const bool do_wilmg = inp.test_wilson_mg;

  // Are we computing the Wilson spectrum?
  const bool do_wil_spectrum = inp.spectrum_wilson;

  // Are we computing the gamma5 Wilson spectrum?
  const bool do_g5_wil_spectrum = inp.spectrum_g5_wilson;

  // Are we power iterations computing the low gamma5 Wilson spectrum?
  const bool do_g5_alternative_spectrum = inp.spectrum_g5_alternative;

  // How many times to refine (beyond first rotation)
  int n_refine = inp.n_refine; // (x_len -> x_len/2 -> x_len/(2*x_block) -> ...)

  // Blocking size.
  int x_block[MAX_REFINEMENT];
  int y_block[MAX_REFINEMENT];

  for (i = 0; i < MAX_REFINEMENT; i++) {
    x_block[i] = y_block[i] = inp.blocksize[i];
  }
  if (inp.preserve_eo) x_block[0] *= 2;

  // Number of coarse degrees of freedom.
  int coarse_dof[MAX_REFINEMENT];
  for (i = 0; i < MAX_REFINEMENT; i++) {
    coarse_dof[i] = inp.coarse_dof[i];
  }

  // Somewhere to solve inversion info.
  inversion_info invif;

  // Verbosity.
  inversion_verbose_struct verb;
  verb.verbosity = VERB_DETAIL;
  verb.verb_prefix = "Level 0: ";
  verb.precond_verbosity = VERB_DETAIL;
  verb.precond_verb_prefix = "Prec ";

  // Information about the outermost solve.
  const double tol = 1e-10; 
  const int max_iter = 4096*16;
  const int restart_freq = 32; //64;

  // What operator do we use for each solve?
  QMGStencilType outer_stencil[n_refine+1];
  outer_stencil[0] = QMG_MATVEC_ORIGINAL;
  // Optimal for herm preserving
  
  outer_stencil[1] = QMG_MATVEC_ORIGINAL;
  if (n_refine > 1)
    outer_stencil[2] = QMG_MATVEC_ORIGINAL;
  for (i = 3; i < n_refine; i++)
  {
    outer_stencil[i] = QMG_MATVEC_ORIGINAL;
  }

  // How are we generating near-null vectors?
  NullVecSetupMG null_setup[n_refine];

  // Variables we use to determine what operators
  // we use to generate near-null vectors.

  // Do we generate with the normal operator?
  bool do_normal_null = false;

  // M_Mdagger or Mdagger_M?
  bool use_M_Mdagger = false;

  for (i = 0; i < n_refine; i++) {

    // What's the fine level?
    null_setup[i].fine_idx = i;

    // How many coarse degrees of freedom?
    null_setup[i].coarse_dof = coarse_dof[i];

    // Which stencil do we coarsen on each level?
    // Set by what stencil we're using on each solve.
    if (outer_stencil[i] == QMG_MATVEC_RIGHT_SCHUR || outer_stencil[i] == QMG_MATVEC_RIGHT_JACOBI)
      null_setup[i].outer_stencil_coarsen = MultigridMG::QMG_MULTIGRID_PRECOND_RIGHT_BLOCK_JACOBI;
    else
      null_setup[i].outer_stencil_coarsen = MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL;

    // Are we preserving eo?
    null_setup[i].preserve_eo = inp.preserve_eo;

    // What operator do we use to generate near-null vectors?
    if (outer_stencil[i] == QMG_MATVEC_ORIGINAL)
      null_setup[i].null_op_type = (do_normal_null ? (use_M_Mdagger ? QMG_MATVEC_M_MDAGGER : QMG_MATVEC_MDAGGER_M) : QMG_MATVEC_ORIGINAL);
    else // QMG_MATVEC_RIGHT_JACOBI
      null_setup[i].null_op_type = (do_normal_null ? (use_M_Mdagger ? QMG_MATVEC_RBJ_M_MDAGGER : QMG_MATVEC_RBJ_MDAGGER_M) : QMG_MATVEC_RIGHT_JACOBI);

    // Do we generate separate left and right null vectors?
    // Seems to be broken?
    null_setup[i].separate_left_right_nulls = false;

    // Tolerance of null vector solve.
    null_setup[i].nulls_tolerance = do_normal_null ? 1e-4 : 5e-5;
    
    // Max iter of null vector solve.
    null_setup[i].nulls_max_iter = do_normal_null ? 250 : 500;

    // rng generator
    null_setup[i].generator = &generator;

    // Verbosity
    null_setup[i].verb = verb;

#ifndef NO_ARPACK
    // Do we use eigenvectors as null vectors?
    null_setup[i].nulls_are_eigenvectors = inp.nulls_are_evecs[i];

    // Do we grab just positive eigenvectors for null vectors, or all?
    null_setup[i].nulls_positive_evec_only = true;

    // What's the maximum number of iterations used for eigenvector
    // generation?
    null_setup[i].nulls_arpack_max_iter = 4000;

    // And the tolerance?
    null_setup[i].nulls_arpack_tol = 1e-12;
#endif

  }

  // Are we using CGNE on the coarsest level?[why CGNE]
  bool do_normal_coarsest = inp.do_normal_coarsest;

  // Are we adding a stabilizing shift to the coarsest level? (Only for normal solve)
  double normal_coarsest_shift = 0.0;


  // Information about intermediate solves.
  const double inner_tol = 0.25;
  const int inner_max_iter = 8;
  const int inner_restart_freq = 32;

  // Information about pre- and post-smooths. 
  const int n_pre_smooth = 2;
  const double pre_smooth_tol = 1e-15; // never
  const int n_post_smooth = 2;
  const double post_smooth_tol = 1e-15; // never

  // Information about the coarsest solve.
  const double coarsest_tol = 0.25;
  const int coarsest_max_iter = 8192;
  const int coarsest_restart_freq = 10000;//1024;

  // Start building lattices
  Lattice2D** lats = new Lattice2D*[1+n_refine];
  lats[0] = new Lattice2D(x_len, y_len, dof); 
  
  // Convenient function to load gauge fields
  // See ../util/util.h
  complex<double>* gauge_field = allocate_vector<complex<double>>(lats[0]->get_volume()*lats[0]->get_nd()*1); // 1*1 is because U(1) gauge field
  test_get_gauge_field(gauge_field, lats[0], beta, do_free, generator, true /* be verbose */, inp.add_topology);

  // Create a Wilson
  Wilson2D* wilson_op = new Wilson2D(lats[0], mass, gauge_field);
  wilson_op->build_dagger_stencil();
  wilson_op->build_rbjacobi_stencil();
  wilson_op->build_rbj_dagger_stencil();

  // Prepare level solve objects.
  StatefulMultigridMG::LevelSolveMG** level_solve_objs = new StatefulMultigridMG::LevelSolveMG*[n_refine];

  // Prepare coarsest solve object for the coarsest level.
  StatefulMultigridMG::CoarsestSolveMG* coarsest_solve_obj = new StatefulMultigridMG::CoarsestSolveMG;
  if (outer_stencil[n_refine] == QMG_MATVEC_RIGHT_SCHUR || outer_stencil[n_refine] == QMG_MATVEC_RIGHT_JACOBI)
  {
    coarsest_solve_obj->coarsest_stencil_app = (do_normal_coarsest ? QMG_MATVEC_RBJ_M_MDAGGER : QMG_MATVEC_RIGHT_JACOBI);
  }
  else
  {
    coarsest_solve_obj->coarsest_stencil_app = (do_normal_coarsest ? QMG_MATVEC_M_MDAGGER : QMG_MATVEC_ORIGINAL);
  }
  coarsest_solve_obj->coarsest_tol = coarsest_tol;
  coarsest_solve_obj->coarsest_iters = coarsest_max_iter;
  coarsest_solve_obj->coarsest_restart_freq = (do_normal_coarsest ? -1 : coarsest_restart_freq);
#ifndef NO_ARPACK
  //coarsest_solve_obj->deflate = do_deflate_coarsest;
#endif
  coarsest_solve_obj->normal_shift = normal_coarsest_shift;

  // Create a MultigridMG object, push top level onto it!
  StatefulMultigridMG* mg_object = new StatefulMultigridMG(lats[0], wilson_op, coarsest_solve_obj);

  // Create a transfer objects.
  TransferMG** transfer_objs = new TransferMG*[n_refine];

  // Now we need to create the other levels.
  if (n_refine > 0)
  {
    int curr_x_len = x_len; // starting x dimension.
    int curr_y_len = y_len; // starting y dimension.
    for (i = 0; i < n_refine; i++)
    {
      const int fine_idx = i; // Index the fine level.
      const int coarse_idx = i+1; // Index the coarse level.

      if (curr_x_len == 6 && curr_y_len == 6) // safety check for L = 192.
      {
        curr_x_len = 2;
        curr_y_len = 2;
      }
      else
      {
        // Update to the new lattice size.
        curr_x_len /= x_block[i];
        curr_y_len /= y_block[i];
      }


      std::cout << "Level " << fine_idx << " X: " << curr_x_len << " Y: " << curr_y_len << "\n";

      // Create a new lattice object.
      lats[coarse_idx] = new Lattice2D(curr_x_len, curr_y_len, coarse_dof[fine_idx]);

      // Create new null vectors. These are copied into local memory
      // in the transfer object, so we can create and destroy these
      // in this loop. 
      // If we're doing a symmetric coarsening, we just alias the two.
      complex<double>** null_prolong = new complex<double>*[coarse_dof[fine_idx]];
      for (j = 0; j < coarse_dof[fine_idx]; j++)
      {
        null_prolong[j] = mg_object->get_storage(fine_idx)->check_out();
      }
      complex<double>** null_restrict;
      if (null_setup[fine_idx].separate_left_right_nulls)
      {
        null_restrict = new complex<double>*[coarse_dof[fine_idx]];
        for (j = 0; j < coarse_dof[fine_idx]; j++)
        {
          null_restrict[j] = mg_object->get_storage(fine_idx)->check_out();
        }
      }
      else
      {
        null_restrict = null_prolong;
      }

      // null_restrict gets ignored if we're doing a 
      // symmetric setup;
      int null_dslash_count = generate_null_vectors(mg_object->get_stencil(fine_idx),
                                null_setup[fine_idx],
                                null_prolong,
                                null_restrict);

      if (null_dslash_count == -1) {
        std::cout << "[QMG-ERROR]: Error in null vector generation. Exiting...";
        return -1;
      }

      mg_object->add_tracker_count(QMG_DSLASH_TYPE_NULLVEC, null_dslash_count, fine_idx);

      // Fine lattice, coarse lattice, null vector(s), perform the block ortho.
      if (null_setup[fine_idx].separate_left_right_nulls)
      {
        transfer_objs[fine_idx] = new TransferMG(lats[fine_idx], lats[coarse_idx], null_prolong, null_restrict, true, false, QMG_DOUBLE_PROJECTION);
      }
      else
      {
        transfer_objs[fine_idx] = new TransferMG(lats[fine_idx], lats[coarse_idx], null_prolong, true, false, QMG_DOUBLE_PROJECTION);
      }

      // Prepare a new LevelSolveMG object for the new level.
      // Largely ignored for fine_idx = 0, since that's the outer solve.
      level_solve_objs[fine_idx] = new StatefulMultigridMG::LevelSolveMG;
      level_solve_objs[fine_idx]->fine_stencil_app = outer_stencil[fine_idx];
      level_solve_objs[fine_idx]->intermediate_tol = inner_tol;
      level_solve_objs[fine_idx]->intermediate_iters = inner_max_iter;
      level_solve_objs[fine_idx]->intermediate_restart_freq = inner_restart_freq;
      level_solve_objs[fine_idx]->pre_tol = pre_smooth_tol;
      level_solve_objs[fine_idx]->pre_iters = n_pre_smooth;
      level_solve_objs[fine_idx]->post_tol = post_smooth_tol;
      level_solve_objs[fine_idx]->post_iters = n_post_smooth;

      // Push a new level on the multigrid object! Also, save the global null vector.
      // Arg 1: New lattice
      // Arg 2: New transfer object (between new and prev lattice)
      // Arg 3: Should we construct the coarse stencil?
      // Arg 4: Is the operator chiral? (True for Wilson)
      // Arg 5: What should we construct the coarse stencil from?
      // Arg 6: Should we prep dagger or rbjacobi stencil (rbjacobi, for this test)
      // Arg 7: Non-block-orthogonalized null vector.
      mg_object->push_level(lats[coarse_idx], transfer_objs[fine_idx], level_solve_objs[fine_idx], true, Wilson2D::has_chirality(), null_setup[fine_idx].outer_stencil_coarsen, CoarseOperator2D::QMG_COARSE_BUILD_ALL/*, null_vectors*/);

      // Clean up null vectors.
      if (null_setup[fine_idx].separate_left_right_nulls)
      {
        for (j = 0; j < coarse_dof[fine_idx]; j++)
        {
          mg_object->get_storage(fine_idx)->check_in(null_restrict[j]);
        }
        delete[] null_restrict;
      }
      for (j = 0; j < coarse_dof[fine_idx]; j++)
      {
        mg_object->get_storage(fine_idx)->check_in(null_prolong[j]);
      }
      delete[] null_prolong;

      // Clean up some extra memory.
      mg_object->get_storage(fine_idx)->consolidate();

    }

  }

  ////////////////////////////
  // PREPARE FOR INVERSIONS //
  ////////////////////////////

  complex<double>* b;
  complex<double>* x;
  complex<double>* Ax;
  double bnorm; 

  //////////////////////////////////////////////
  // Get the spectrum of the Wilson operator. //
  //////////////////////////////////////////////
std::cout << "###do_wilmg: " << do_wilmg << "\n";
std::cout << "###do_wil_spectrum: " << do_wil_spectrum << "\n";
std::cout << "###do_g5_wil_spectrum: " << do_g5_wil_spectrum << "\n";
std::cout << "###do_g5_alternative_spectrum: " << do_g5_alternative_spectrum << "\n";
std::cout << "###n_refine: " << n_refine << "\n";
std::cout << "###beta: " << beta << "\n";
std::cout << "###mass: " << mass << "\n";
std::cout << "###x_len: " << x_len << "\n";
std::cout << "###y_len: " << y_len << "\n";
std::cout << "###dof: " << dof << "\n";
std::cout << "###do_free: " << do_free << "\n";

  if (do_wil_spectrum)
  {


    for (j = 0; j <= n_refine; j++)
    {
      // HACK
      //mg_object->get_stencil(j)->update_shift(-1.05);

      std::cout << "[REMARK]: Beginning eigenvalue calculation for level " << j << ".\n";
      const int fine_idx = j;

      const int length = lats[fine_idx]->get_size_cv();

      //mg_object->get_stencil(fine_idx)->print_stencil_site(0,0);

      // Allocate a sufficiently gigantic matrix (ugh)
      cMatrix mat_cplx = cMatrix::Zero(length, length);

      // Allocate a vector for matrix elements.
      complex<double>* in_cplx = allocate_vector<complex<double>>(length);
      zero_vector(in_cplx, length);

      // Begin taking matrix elements.
      for (int i = 0; i < length; i++)
      {
        if (i > 0)
          in_cplx[i-1] = 0.0;
        in_cplx[i] = 1.0;

        // apply mat-vec
        mg_object->apply_stencil(&(mat_cplx(i*length)), in_cplx, j, QMG_MATVEC_ORIGINAL);
      }

      // Get the eigenvalues and eigenvectors.
      ComplexEigenSolver< cMatrix > eigsolve_cplx_indef(length);
      eigsolve_cplx_indef.compute(mat_cplx);

      cMatrix eigs = eigsolve_cplx_indef.eigenvalues();

      for (i = 0; i < lats[fine_idx]->get_size_cv(); i++)
        std::cout << "[ORIG-SPECTRUM]: Level " << fine_idx << " Eval " << i << " " << real(eigs(i)) << " + I " << imag(eigs(i)) << "\n";

    }
  }

  ////////////////////////////////////////
  // Get the spectrum of gamma5 Wilson. //
  ////////////////////////////////////////

  if (do_g5_wil_spectrum)
  {

    double mass_shift = 0.0;

    for (int m = 0; m <= 100; m++) {

      mass_shift = -m*0.02; 

      for (j = 0; j <= n_refine; j++)
      {
        // HACK
        mg_object->get_stencil(j)->update_shift(mass_shift);

        std::cout << "[REMARK]: Beginning eigenvalue calculation for level " << j << ".\n";
        const int fine_idx = j;

        const int length = lats[fine_idx]->get_size_cv();

        //mg_object->get_stencil(fine_idx)->print_stencil_site(0,0);

        // Allocate a sufficiently gigantic matrix (ugh)
        cMatrix mat_cplx = cMatrix::Zero(length, length);

        // Allocate a vector for matrix elements.
        complex<double>* in_cplx = allocate_vector<complex<double>>(length);
        complex<double>* inter_vec = allocate_vector<complex<double>>(length);
        zero_vector(in_cplx, length);
        zero_vector(inter_vec, length);

        // Begin taking matrix elements.
        for (int i = 0; i < length; i++)
        {
          if (i > 0)
            in_cplx[i-1] = 0.0;
          in_cplx[i] = 1.0;

          // apply mat-vec
          zero_vector(inter_vec, length);

          mg_object->apply_stencil(inter_vec, in_cplx, j, QMG_MATVEC_ORIGINAL);
          mg_object->get_stencil(j)->gamma5(&(mat_cplx(i*length)), inter_vec);
        }

        // Get the eigenvalues and eigenvectors.
        SelfAdjointEigenSolver< cMatrix > eigsolve_cplx_indef(length);
        eigsolve_cplx_indef.compute(mat_cplx);

        cMatrix eigs = eigsolve_cplx_indef.eigenvalues();

        for (i = 0; i < lats[fine_idx]->get_size_cv(); i++)
          std::cout << "[G5-SPECTRUM]: Step " << m << " Mass " << mass_shift << " Level " << fine_idx << " Eval " << i << " " << real(eigs(i)) << " + I " << imag(eigs(i)) << "\n";

        mg_object->get_stencil(j)->update_shift(mass);

      }
    }
  }

  //////////////////////////////////////////////
  // Get the lowest spectrum of gamma5 Wilson //
  //////////////////////////////////////////////

  if (do_g5_alternative_spectrum)
  {

    double mass_shift = 0.0;

    const int n_eig = 20;

    for (int m = 0; m <= 100; m++) {

      mass_shift = -m*0.02;

      std::cout << "[REMARK]: Mass shift " << mass_shift << "\n";

      for (j = 0; j <= n_refine; j++)
      {
        // HACK
        mg_object->get_stencil(j)->update_shift(mass_shift);

        std::cout << "[REMARK]: Beginning eigenvalue calculation for level " << j << ".\n";
        const int fine_idx = j;

        const int length = lats[fine_idx]->get_size_cv();


        FunctionWrapper<complex<double>> wil_M_dagger_M_fcn(Stencil2D::get_apply_function(QMG_MATVEC_MDAGGER_M),
                                                            (void*)mg_object->get_stencil(j), 
                                                            length);

        const int n_mini = 50;
        // Get approximate bounds from a small Lanczos...
        SimpleComplexLanczos<double> simp_lanczos(&wil_M_dagger_M_fcn, n_mini, generator);
        simp_lanczos.compute();
        double approx_eigs[n_mini];
        simp_lanczos.ritzvalues((double*)approx_eigs);

        //std::cout << "The " << n_mini << " Ritz values are:\n";
        //for (int i = 0; i < n_mini; i++) {
        //  std::cout << approx_eigs[i] << "\n";
        //}
        //std::cout << "\n";

        double approx_min = approx_eigs[0]; // get the approximate min
        double approx_max = approx_eigs[n_mini-1]*1.2; // and overshoot the max
        //std::cout << "The linear op window is " << approx_min << " to " << approx_max << "\n\n";

        // Make a linear interpolation of the laplace op.
        LinearMapToUnit<complex<double>,double> linear_mg(&wil_M_dagger_M_fcn, approx_min, approx_max);

        // And make the 20th order poly accelerated form
        OneOverOnePlusX<complex<double> > poly_accel(&linear_mg, 10);

        
        // Fill a struct
        TRCLStruct poly_wil_props;
        poly_wil_props.n_ev = n_eig; // get 20 eigenvalues
        poly_wil_props.m = max(72, 2*n_eig); // subspace size of 20
        poly_wil_props.tol = 1e-8; // lock at a tolerance of 1e-8
        poly_wil_props.max_restarts = 1000; // maximum of 10 restarts
        poly_wil_props.preserved_space = 5*n_eig/4; // space preserved after restart,
                                        // set to -1 to default to m/4+1
        poly_wil_props.deflate = true; // deflate locked eigenvalues
        poly_wil_props.generator = &generator; // passed by reference
        poly_wil_props.verbose = false;

        // Get the largest eigenvalues, since we're poly acceling
        ThickRestartComplexLanczos<double,std::greater<double>> poly_lanczos(&poly_accel, poly_wil_props);
        poly_lanczos.compute();
        int n_poly_converged = poly_lanczos.num_converged();
        //std::cout << n_poly_converged << "\n";

        // Get the poly eigenvalues
        double* poly_eigenvalues = new double[n_poly_converged];
        poly_lanczos.ritzvalues(poly_eigenvalues);

        // Print the Ritz values
        //std::cout << "The " << n_poly_converged << " converged eigenvalues are:\n";
        //for (int i = 0; i < n_poly_converged; i++) {
        //  std::cout << poly_eigenvalues[i] << "\n";
        //}

        delete[] poly_eigenvalues;

        complex<double>* coarsest_evals = new complex<double>[n_eig];
        complex<double>** coarsest_evecs = new complex<double>*[n_eig];
        for (i = 0; i < n_eig; i++)
        {
          coarsest_evecs[i] = mg_object->check_out(j);
        }
        complex<double>* interm = mg_object->check_out(j);

        poly_lanczos.ritzvectors(coarsest_evecs);

        // get actual eigenvalues
        for (int i = 0; i < n_eig; i++) {
          zero_vector(interm, length);
          mg_object->apply_stencil(interm, coarsest_evecs[i], j, QMG_MATVEC_ORIGINAL);
          mg_object->get_stencil(j)->gamma5(interm);
          auto eval_check = dot(coarsest_evecs[i], interm, length);
          coarsest_evals[i] = real(eval_check);
          std::cout << eval_check << " ";

          // also check ritz value with gamma_5
          zero_vector(interm, length);
          mg_object->get_stencil(j)->gamma5(interm, coarsest_evecs[i]);
          eval_check = dot(coarsest_evecs[i], interm, length);
          std::cout << eval_check << "\n";
        }

        for (int n = 0; n < n_eig; n++) {
          mg_object->check_in(coarsest_evecs[n], j);
        }
        mg_object->check_in(interm, j);

        // clean up
        delete[] coarsest_evecs;
        delete[] coarsest_evals;


        mg_object->get_stencil(j)->update_shift(mass);

      }
    }
  }

  ////////////////////////////////////////////////////////////////
  // WILSON MG SOLVE: MAKES SURE WE GENERATED GOOD NULL VECTORS //
  ////////////////////////////////////////////////////////////////

  if (do_wilmg)
  {
    std::cout << "Starting Wilson solve.\n" << flush;

    // What type of solve are we doing?
    matrix_op_cplx apply_stencil_op = Stencil2D::get_apply_function(outer_stencil[0]);
    int solve_size;
    if (outer_stencil[0] == QMG_MATVEC_RIGHT_SCHUR)
      solve_size = lats[0]->get_size_cv()/2;
    else
      solve_size = lats[0]->get_size_cv();

    // Create a staggered right hand side, fill with gaussian random numbers.
    b = mg_object->check_out(0);
    gaussian(b, lats[0]->get_size_cv(), generator);
    bnorm = sqrt(norm2sq(b, lats[0]->get_size_cv()));

    // Create a place to accumulate a solution. Assume a zero initial guess.
    x = mg_object->check_out(0);
    zero_vector(x, lats[0]->get_size_cv());

    // Create a place to compute Ax. Since we have a zero initial guess, this
    // starts as zero.
    Ax = mg_object->check_out(0);
    zero_vector(Ax, lats[0]->get_size_cv());

    // Prepare b.
    complex<double>* b_prep = mg_object->check_out(0);
    zero_vector(b_prep, lats[0]->get_size_cv());
    mg_object->get_stencil(0)->prepare_M(b_prep, b, outer_stencil[0]);
    double bprepnorm = sqrt(norm2sq(b_prep, solve_size));

    // Run a VPGCR solve!
    std::cout << "\n";
    verb.verb_prefix = "[QMG-MG-SOLVE-INFO]: Level 0 ";
    if (n_refine > 0) {
      invif = minv_vector_gcr_var_precond_restart(x, b_prep, solve_size,
                  max_iter, tol*bnorm/bprepnorm, restart_freq,
                  apply_stencil_op, (void*)mg_object->get_stencil(0),
                  StatefulMultigridMG::mg_preconditioner, (void*)mg_object, &verb);
    } else {
      invif = minv_vector_bicgstab_l(x, b_prep, solve_size, max_iter, tol*bnorm/bprepnorm, 6, apply_stencil_op, (void*)mg_object->get_stencil(0), &verb);
    }
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


    // Reconstruct x.
    complex<double>* x_reconstruct = mg_object->check_out(0);
    zero_vector(x_reconstruct, lats[0]->get_size_cv());
    mg_object->get_stencil(0)->reconstruct_M(x_reconstruct, x, b, outer_stencil[0]);

    // Check solution.
    zero_vector(Ax, lats[0]->get_size_cv());
    mg_object->apply_stencil(Ax, x_reconstruct, 0);
    cout << "[QMG-TOLERANCE-BLOCK-TEST]: " << sqrt(diffnorm2sq(b, Ax, lats[0]->get_size_cv()))/bnorm << "\n";


    // Check vectors back in.
    mg_object->check_in(b_prep, 0);
    mg_object->check_in(x_reconstruct, 0);
    mg_object->check_in(Ax, 0);
    mg_object->check_in(x, 0);
    mg_object->check_in(b, 0);
  }


  ///////////////
  // Clean up. //
  ///////////////

  deallocate_vector(&gauge_field);

#ifndef NO_ARPACK
  if (do_deflate_coarsest)
  {
    if (coarsest_evals != 0)
    {
      delete[] coarsest_evals;
      coarsest_evals = 0;
    }
    if (coarsest_evecs != 0)
    {
      for (i = 0; i < deflate_coarsest_low; i++)
      {
        deallocate_vector(&coarsest_evecs[i]);
      }
      delete[] coarsest_evecs;
    }
  }
#endif

  // Delete MultigridMG.
  delete mg_object;

  // Delete transfer object.
  for (i = 0; i < n_refine; i++)
  {
    delete transfer_objs[i];
  }
  delete[] transfer_objs;

  // Delete stencil.
  delete wilson_op;

  // Delete coarsest solve objects.
  delete coarsest_solve_obj;

  // Delete level solve object.
  for (i = 0; i < n_refine; i++)
  {
    delete level_solve_objs[i];
  }
  delete[] level_solve_objs;

  // Delete lattices.
  for (i = 0; i < n_refine+1; i++)
  {
    delete lats[i];
  }
  delete[] lats; 

  return 0;
}

