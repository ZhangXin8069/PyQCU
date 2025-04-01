// // Copyright (c) 2018 Evan S Weinberg
// Useful setup for generating near-null vectors
// in a handful of different ways

// QLINALG
#include "blas/generic_vector.h"
#include "inverters/generic_cg.h"
#include "inverters/generic_cr.h"
#include "inverters/generic_gcr.h"
#include "inverters/generic_gcr_var_precond.h"
#include "inverters/generic_bicgstab_l.h"

// QMG
#include "lattice/lattice.h"
#include "transfer/transfer.h"
#include "stencil/stencil_2d.h"
#include "multigrid/stateful_multigrid.h"

#ifndef NO_ARPACK
#include "interfaces/arpack/generic_arpack.h"
#endif

struct NullVecSetupMG
{
  // What's the fine level?
  int fine_idx;

  // How many coarse degrees of freedom?
  int coarse_dof;

  // Which stencil do we coarsen on each level?
  MultigridMG::QMGMultigridPrecondStencil outer_stencil_coarsen;

  // Are we preserving even/odd (as a proxy to preserve symmetry I guess)
  bool preserve_eo;

  // What operator do we use to generate near-null vectors?
  QMGStencilType null_op_type;

  // Tolerance of null vector solve
  double nulls_tolerance;

  // Max iters of null vector solve
  int nulls_max_iter;

  // Generating separate left/right near-null vectors
  bool separate_left_right_nulls;

  // RNG
  std::mt19937 *generator;

  // Verbosity structure
  inversion_verbose_struct verb;

#ifndef NO_ARPACK
  // Do we use eigenvectors as null vectors?
  bool nulls_are_eigenvectors;

  // Do we grab just positive eigenvectors for null vectors, or all?
  bool nulls_positive_evec_only;

  // What's the maximum number of iterations we should spend on
  // generating eigenvectors?
  int nulls_arpack_max_iter;

  // What tolerance are we generating eigenvectors to?
  double nulls_arpack_tol;
#endif

  // set some defaults
  NullVecSetupMG()
      : fine_idx(0), coarse_dof(8),
        outer_stencil_coarsen(MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL),
        preserve_eo(false),
        null_op_type(QMG_MATVEC_ORIGINAL),
        nulls_tolerance(5e-5), nulls_max_iter(500),
        separate_left_right_nulls(false),
        generator(nullptr), verb()
  {
#ifndef NO_ARPACK
    nulls_are_eigenvectors = false;
    nulls_positive_evec_only = true;
    nulls_arpack_max_iter = 4000;
    nulls_arpack_tol = 1e-7;
#endif
  }
};

// Arg 1: Fine operator
// Arg 2: Setup parameter
// Arg 3: Array of near null vectors (used for prolongate in symmetric case)
// Arg 4: Optional: array of near null vectors for restriction in asymmetric case
// Returns: number of dslash. [Bug: only for near-null, not eigenvectors...]
int generate_null_vectors(Stencil2D *fine_op, NullVecSetupMG &null_setup, complex<double> **null_prolong, complex<double> **null_restrict = nullptr)
{

  // Iterators
  int j, k;

  // What's the index of the fine level?
  int fine_idx = null_setup.fine_idx;

  // Accumulate dslash counter here.
  int dslash_count = 0;

  // Somewhere to solve inversion info.
  inversion_info invif;

  // For convenience, grab the lattice object and color-vector size
  Lattice2D *lat_fine = fine_op->get_lattice();
  int volume = lat_fine->get_volume();
  int fine_nc = lat_fine->get_nc();
  int cv_size = lat_fine->get_size_cv();

  // And grab the number of coarse dof
  int coarse_dof = null_setup.coarse_dof;

  // How many do we need to generate before doubling?
  int gen_coarse = null_setup.preserve_eo ? coarse_dof / 4 : coarse_dof / 2;

  // Figure out if we're generating near-null vectors with a normal operator.
  bool do_normal_null = false;
  if (null_setup.null_op_type == QMG_MATVEC_M_MDAGGER || null_setup.null_op_type == QMG_MATVEC_MDAGGER_M ||
      null_setup.null_op_type == QMG_MATVEC_RBJ_M_MDAGGER || null_setup.null_op_type == QMG_MATVEC_RBJ_MDAGGER_M)
  {
    do_normal_null = true;
  }

  // If we're generating with the normal op, there's a multiplicative factor
  // on the dslash count.
  int null_op_factor = (do_normal_null ? 2 : 1);

  // Figure out if we're generating separate left and right near-null vectors
  bool separate_left_right_nulls = null_setup.separate_left_right_nulls;

  // Max iters/tolerance of near-null vector solve.
  int nulls_max_iter = null_setup.nulls_max_iter;
  double nulls_tolerance = null_setup.nulls_tolerance;

  // Grab the RNG.
  std::mt19937 &generator = *null_setup.generator;

  // And the verbosity object.
  inversion_verbose_struct verb = null_setup.verb;

  // Pre-allocate some vectors we'll need later.
  complex<double> *rand_guess = allocate_vector<complex<double>>(cv_size);
  complex<double> *Arand_guess = allocate_vector<complex<double>>(cv_size);

  // only get used for rbj schur generation
  complex<double> *rhs_prep = allocate_vector<complex<double>>(cv_size);
  complex<double> *lhs_preconstruct = allocate_vector<complex<double>>(cv_size);

#ifndef NO_ARPACK
  bool nulls_are_eigenvectors = null_setup.nulls_are_eigenvectors;

  // Are we only grabbing positive evecs?
  bool nulls_positive_evec_only = null_setup.nulls_positive_evec_only;

  // How many iterations are we doing/what tolerance are we specifying
  // for the eigenvalue calculation?
  int arpack_max_iter = null_setup.nulls_arpack_max_iter;
  double arpack_tol = null_setup.nulls_arpack_tol;

  if (nulls_are_eigenvectors)
  {
    /*if (do_normal_null)
    {
      std::cout << "[QMG-ERROR]: Cannot use eigenvectors when doing null vector generation with normal op.\n";
      return -1;
    }*/
    arpack_dcn *arpack;
    complex<double> *coarsest_evals_right = new complex<double>[coarse_dof];
    complex<double> **coarsest_evecs_right = new complex<double> *[coarse_dof];
    for (j = 0; j < coarse_dof; j++)
    {
      coarsest_evecs_right[j] = allocate_vector<complex<double>>(cv_size);
    }

    // Grab lowest coarse_dof eigenvectors of D.
    arpack = new arpack_dcn(cv_size, arpack_max_iter, arpack_tol,
                            Stencil2D::get_apply_function(null_setup.null_op_type),
                            /*(null_setup.null_op_type == QMG_MATVEC_ORIGINAL) ? apply_stencil_2D_M : apply_stencil_2D_M_rbjacobi,*/
                            fine_op, coarse_dof, 3 * coarse_dof);

    arpack->prepare_eigensystem(arpack_dcn::ARPACK_SMALLEST_MAGNITUDE, coarse_dof, 3 * coarse_dof);
    arpack->get_eigensystem(coarsest_evals_right, coarsest_evecs_right, arpack_dcn::ARPACK_SMALLEST_MAGNITUDE);
    delete arpack;
    for (j = 0; j < coarse_dof; j++)
    {
      std::cout << "Right eval " << j << " " << coarsest_evals_right[j] << "\n";
      normalize(coarsest_evecs_right[j], cv_size);
    }

    // Grab eigenvectors
    if (nulls_positive_evec_only &&
        (null_setup.null_op_type == QMG_MATVEC_ORIGINAL || null_setup.null_op_type == QMG_MATVEC_RIGHT_JACOBI))
    {
      int evec_count = 0;
      int index = 0;
      while (evec_count < gen_coarse)
      {
        // Check for real eigenvalue
        if (fabs(coarsest_evals_right[index].imag()) < 1e-10)
        {
          copy_vector(null_prolong[evec_count], coarsest_evecs_right[index], cv_size);
          std::cout << "Eval " << evec_count << " " << coarsest_evals_right[index] << "\n";
          index++;
        }
        else if (coarsest_evals_right[index].imag() > coarsest_evals_right[index + 1].imag())
        {
          copy_vector(null_prolong[evec_count], coarsest_evecs_right[index], cv_size);
          std::cout << "Eval " << evec_count << " " << coarsest_evals_right[index] << "\n";
          index += 2;
        }
        else
        {
          copy_vector(null_prolong[evec_count], coarsest_evecs_right[index + 1], cv_size);
          std::cout << "Eval " << evec_count << " " << coarsest_evals_right[index + 1] << "\n";
          index += 2;
        }
        evec_count++;
      }
    }
    else
    {
      for (j = 0; j < gen_coarse; j++)
      {
        copy_vector(null_prolong[j], coarsest_evecs_right[j], cv_size);
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
#endif // ifndef NO_ARPACK
    for (j = 0; j < gen_coarse; j++)
    {

      // Update verbosity string.
      verb.verb_prefix = "Level " + to_string(fine_idx) + " Null Vector " + to_string(j) + " ";

      // Will become up chiral projection
      zero_vector(null_prolong[j], cv_size);

      // Fill with random numbers.
      gaussian(rand_guess, cv_size, generator);

      // Make orthogonal to previous vectors.
      for (k = 0; k < j; k++)
        orthogonal(rand_guess, null_prolong[k], cv_size);

      // First, properly prepare the residual equation.
      // Are we using the normal op? RBJ or not?
      zero_vector(Arand_guess, cv_size);
      fine_op->apply_M(Arand_guess, rand_guess, null_setup.null_op_type);
      dslash_count += 1 * null_op_factor;
      cax(-1.0, Arand_guess, cv_size);

      // Solve the residual equation.
      // If we're doing rbjacobi, actually do the Schur system.
      // There's plenty of flex here.
      double local_bnorm = 0.0, local_bprepnorm = 0.0; // not necc. needed.
      switch (null_setup.null_op_type)
      {
      case QMG_MATVEC_ORIGINAL:
        // Nothing to prepare, just do it!
        invif = minv_vector_bicgstab_l(null_prolong[j], Arand_guess, cv_size, nulls_max_iter, nulls_tolerance, 6, apply_stencil_2D_M, (void *)fine_op, &verb);
        dslash_count += invif.ops_count;
        break;
      case QMG_MATVEC_M_MDAGGER:
        // Nothing to prepare!
        invif = minv_vector_cg(null_prolong[j], Arand_guess, cv_size, nulls_max_iter, nulls_tolerance, apply_stencil_2D_M_M_dagger, (void *)fine_op, &verb);
        dslash_count += 2 * invif.ops_count;
        break;
      case QMG_MATVEC_MDAGGER_M:
        // Nothing to prepare!
        invif = minv_vector_cg(null_prolong[j], Arand_guess, cv_size, nulls_max_iter, nulls_tolerance, apply_stencil_2D_M_dagger_M, (void *)fine_op, &verb);
        dslash_count += 2 * invif.ops_count;
        break;
      case QMG_MATVEC_RIGHT_JACOBI:
        // This one takes preparation.
        zero_vector(rhs_prep, cv_size);
        zero_vector(lhs_preconstruct, cv_size);
        local_bnorm = sqrt(norm2sq(Arand_guess, cv_size));
        fine_op->prepare_M_rbjacobi_schur(rhs_prep, Arand_guess);
        local_bprepnorm = sqrt(norm2sq(rhs_prep, cv_size / 2));
        invif = minv_vector_bicgstab_l(lhs_preconstruct, rhs_prep, cv_size / 2, nulls_max_iter, nulls_tolerance * local_bnorm / local_bprepnorm, 6, apply_stencil_2D_M_rbjacobi_schur, (void *)fine_op, &verb);
        fine_op->reconstruct_M_rbjacobi_schur_to_rbjacobi(null_prolong[j], lhs_preconstruct, Arand_guess);
        dslash_count += invif.ops_count + 1;
        break;
      case QMG_MATVEC_RBJ_M_MDAGGER:
        // Nothing to prepare!
        invif = minv_vector_cg(null_prolong[j], Arand_guess, cv_size, nulls_max_iter, nulls_tolerance, apply_stencil_2D_M_rbjacobi_MMD, (void *)fine_op, &verb);
        dslash_count += 2 * invif.ops_count;
        break;
      case QMG_MATVEC_RBJ_MDAGGER_M:
        // Nothing to prepare!
        invif = minv_vector_cg(null_prolong[j], Arand_guess, cv_size, nulls_max_iter, nulls_tolerance, apply_stencil_2D_M_rbjacobi_MDM, (void *)fine_op, &verb);
        dslash_count += 2 * invif.ops_count;
        break;
      default:
        std::cout << "[QMG-ERROR]: Unsupported null_op_type on level " << fine_idx << "!\n"
                  << flush << "\n";
        return -1;
      }

      // Undo residual equation.
      cxpy(rand_guess, null_prolong[j], cv_size);

      // Orthogonalize against previous vectors.
      for (k = 0; k < j; k++)
        orthogonal(null_prolong[j], null_prolong[k], cv_size);
    }
#ifndef NO_ARPACK
  }
#endif

  // Generate separate left vectors as appropriate.
  if (separate_left_right_nulls)
  {
#ifndef NO_ARPACK
    if (nulls_are_eigenvectors)
    {
      if (do_normal_null)
      {
        std::cout << "[QMG-ERROR]: Cannot use eigenvectors when doing null vector generation with normal op.\n";
        return -1;
      }
      arpack_dcn *arpack;
      complex<double> *coarsest_evals_left = new complex<double>[coarse_dof];
      complex<double> **coarsest_evecs_left = new complex<double> *[coarse_dof];
      for (j = 0; j < coarse_dof; j++)
      {
        coarsest_evecs_left[j] = allocate_vector<complex<double>>(cv_size);
      }

      // Grab lowest coarse_dof eigenvectors of D.
      arpack = new arpack_dcn(cv_size, arpack_max_iter, arpack_tol,
                              (null_setup.null_op_type == QMG_MATVEC_ORIGINAL) ? apply_stencil_2D_M_dagger : apply_stencil_2D_M_rbj_dagger, fine_op,
                              coarse_dof, 3 * coarse_dof);

      arpack->prepare_eigensystem(arpack_dcn::ARPACK_SMALLEST_MAGNITUDE, coarse_dof, 3 * coarse_dof);
      arpack->get_eigensystem(coarsest_evals_left, coarsest_evecs_left, arpack_dcn::ARPACK_SMALLEST_MAGNITUDE);
      delete arpack;
      for (j = 0; j < coarse_dof; j++)
      {
        std::cout << "Left eval " << j << " " << coarsest_evals_left[j] << "\n";
        normalize(coarsest_evecs_left[j], cv_size);
      }

      // Grab eigenvectors
      if (nulls_positive_evec_only)
      {
        int evec_count = 0;
        int index = 0;
        while (evec_count < gen_coarse)
        {
          // Check for real eigenvalue
          if (fabs(coarsest_evals_left[index].imag()) < 1e-10)
          {
            copy_vector(null_restrict[evec_count++], coarsest_evecs_left[index++], cv_size);
          }
          else if (coarsest_evals_left[index].imag() < coarsest_evals_left[index + 1].imag())
          {
            copy_vector(null_restrict[evec_count++], coarsest_evecs_left[index], cv_size);
            std::cout << "Eval " << evec_count << " " << coarsest_evals_left[index] << "\n";
            index += 2;
          }
          else
          {
            copy_vector(null_restrict[evec_count++], coarsest_evecs_left[index + 1], cv_size);
            std::cout << "Eval " << evec_count << " " << coarsest_evals_left[index + 1] << "\n";
            index += 2;
          }
        }
      }
      else
      {
        for (j = 0; j < gen_coarse; j++)
        {
          copy_vector(null_restrict[j], coarsest_evecs_left[j], cv_size);
        }
      }

      for (j = 0; j < coarse_dof; j++)
      {
        deallocate_vector(&coarsest_evecs_left[j]);
      }
      delete[] coarsest_evecs_left;
      delete[] coarsest_evals_left;
    }
    else
    {
#endif // ifndef NO_ARPACK
      for (j = 0; j < gen_coarse; j++)
      {
        // Update verbosity string.
        verb.verb_prefix = "Level " + to_string(fine_idx) + " Left Null Vector " + to_string(j) + " ";

        // Will become up chiral projection
        zero_vector(null_restrict[j], cv_size);

        // Fill with random numbers.
        gaussian(rand_guess, cv_size, generator);

        // Make orthogonal to previous vectors. (Should be bi-orthogonal?)
        for (k = 0; k < j; k++)
          orthogonal(rand_guess, null_restrict[k], cv_size);

        // First, properly prepare the residual equation.
        // Are we using the normal op? RBJ or not?
        zero_vector(Arand_guess, cv_size);
        switch (null_setup.null_op_type)
        {
        case QMG_MATVEC_ORIGINAL:
          fine_op->apply_M(Arand_guess, rand_guess, QMG_MATVEC_DAGGER);
          break;
        case QMG_MATVEC_M_MDAGGER:
          fine_op->apply_M(Arand_guess, rand_guess, QMG_MATVEC_MDAGGER_M);
          break;
        case QMG_MATVEC_MDAGGER_M:
          fine_op->apply_M(Arand_guess, rand_guess, QMG_MATVEC_M_MDAGGER);
          break;
        case QMG_MATVEC_RIGHT_JACOBI:
          fine_op->apply_M(Arand_guess, rand_guess, QMG_MATVEC_RBJ_DAGGER);
          break;
        case QMG_MATVEC_RBJ_M_MDAGGER:
          fine_op->apply_M(Arand_guess, rand_guess, QMG_MATVEC_RBJ_MDAGGER_M);
          break;
        case QMG_MATVEC_RBJ_MDAGGER_M:
          fine_op->apply_M(Arand_guess, rand_guess, QMG_MATVEC_RBJ_M_MDAGGER);
          break;
        default:
          std::cout << "[QMG-ERROR]: Unsupported null_op_type on level " << fine_idx << "!\n"
                    << flush << "\n";
          return -1;
        }
        dslash_count += 1 * null_op_factor;
        cax(-1.0, Arand_guess, cv_size);

        // Solve the residual equation.
        switch (null_setup.null_op_type)
        {
        case QMG_MATVEC_ORIGINAL:
          // Nothing to prepare, just do it!
          invif = minv_vector_bicgstab_l(null_restrict[j], Arand_guess, cv_size, nulls_max_iter, 5e-5, 6, apply_stencil_2D_M_dagger, (void *)fine_op, &verb);
          dslash_count += invif.ops_count;
          break;
        case QMG_MATVEC_M_MDAGGER:
          // Nothing to prepare!
          invif = minv_vector_cg(null_restrict[j], Arand_guess, cv_size, nulls_max_iter, 5e-5, apply_stencil_2D_M_dagger_M, (void *)fine_op, &verb);
          dslash_count += 2 * invif.ops_count;
          break;
        case QMG_MATVEC_MDAGGER_M:
          // Nothing to prepare!
          invif = minv_vector_cg(null_restrict[j], Arand_guess, cv_size, nulls_max_iter, 5e-5, apply_stencil_2D_M_M_dagger, (void *)fine_op, &verb);
          dslash_count += 2 * invif.ops_count;
          break;
        case QMG_MATVEC_RIGHT_JACOBI:
          // Nothing to prepare (I mean, we should have Schur for the dagger, but eh.)
          invif = minv_vector_bicgstab_l(null_prolong[j], Arand_guess, cv_size, nulls_max_iter, 5e-5, 6, apply_stencil_2D_M_rbj_dagger, (void *)fine_op, &verb);
          dslash_count += invif.ops_count;
          break;
        case QMG_MATVEC_RBJ_M_MDAGGER:
          // Nothing to prepare!
          invif = minv_vector_cg(null_restrict[j], Arand_guess, cv_size, nulls_max_iter, 5e-5, apply_stencil_2D_M_rbjacobi_MDM, (void *)fine_op, &verb);
          dslash_count += 2 * invif.ops_count;
          break;
        case QMG_MATVEC_RBJ_MDAGGER_M:
          // Nothing to prepare!
          invif = minv_vector_cg(null_restrict[j], Arand_guess, cv_size, nulls_max_iter, 5e-5, apply_stencil_2D_M_rbjacobi_MMD, (void *)fine_op, &verb);
          dslash_count += 2 * invif.ops_count;
          break;
        default:
          std::cout << "[QMG-ERROR]: Unsupported null_op_type on level " << fine_idx << "!\n"
                    << flush << "\n";
          return -1;
        }

        // Undo residual equation.
        cxpy(rand_guess, null_restrict[j], cv_size);

        // Orthogonalize against previous vectors.
        for (k = 0; k < j; k++)
          orthogonal(null_restrict[j], null_restrict[k], cv_size);
      }
#ifndef NO_ARPACK
    }
#endif
  }

  if (null_setup.preserve_eo)
  {
    for (j = 0; j < coarse_dof / 4; j++)
    {

      if (fine_idx == 0)
      {
        copy_vector(null_prolong[j + coarse_dof / 4] + cv_size / 2,
                    null_prolong[j] + cv_size / 2, cv_size / 2);
        zero_vector(null_prolong[j + coarse_dof / 4], cv_size / 2);
        zero_vector(null_prolong[j] + cv_size / 2, cv_size / 2);
      }
      else
      {
        for (k = 0; k < fine_nc / 4; k++)
        {
          copy_vector_blas(null_prolong[j + coarse_dof / 4] + fine_nc / 4 + k,
                           null_prolong[j] + fine_nc / 4 + k,
                           fine_nc, volume);
          copy_vector_blas(null_prolong[j + coarse_dof / 4] + 3 * fine_nc / 4 + k,
                           null_prolong[j] + 3 * fine_nc / 4 + k,
                           fine_nc, volume);

          zero_vector_blas(null_prolong[j] + fine_nc / 4 + k, fine_nc, volume);
          zero_vector_blas(null_prolong[j] + 3 * fine_nc / 4 + k, fine_nc, volume);

          zero_vector_blas(null_prolong[j + coarse_dof / 4] + k, fine_nc, volume);
          zero_vector_blas(null_prolong[j + coarse_dof / 4] + 2 * fine_nc / 4 + k, fine_nc, volume);
        }
      }

      if (separate_left_right_nulls)
      {
        copy_vector(null_restrict[j + coarse_dof / 4] + cv_size / 2,
                    null_restrict[j] + cv_size / 2, cv_size / 2);
        zero_vector(null_restrict[j + coarse_dof / 4], cv_size / 2);
        zero_vector(null_restrict[j] + cv_size / 2, cv_size / 2);
      }
    }
  }

  for (j = 0; j < coarse_dof / 2; j++)
  {
    // Perform chiral projection, putting "down" projection into the second
    // vector and keeping the "up" projection in the first vector.
    fine_op->chiral_projection_both(null_prolong[j], null_prolong[j + coarse_dof / 2]);

    if (separate_left_right_nulls)
    {
      fine_op->chiral_projection_both(null_restrict[j], null_restrict[j + coarse_dof / 2]);
    }
  }

  std::cout << "[QMG-NOTE]: Doubled null vectors.\n"
            << flush;

  // Cleanup.
  deallocate_vector(&rand_guess);
  deallocate_vector(&Arand_guess);
  deallocate_vector(&rhs_prep);
  deallocate_vector(&lhs_preconstruct);

  return dslash_count;
}