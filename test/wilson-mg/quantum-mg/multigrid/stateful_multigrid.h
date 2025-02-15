// Copyright (c) 2017 Evan S Weinberg
// Header file for a stateful multigrid object,
// which keeps the state of an actual multigrid solve.
// It's also aware of how to perform smearing, preconditioning,
// etc on each level.

// To do: Add a counter for the number of op applications at each level.
// This way we can track the number of operator applications over 
// the entire solve. 

#ifndef QMG_STATEFUL_MG
#define QMG_STATEFUL_MG

#include <map>

// QLINALG
#include "blas/generic_vector.h"
#include "inverters/generic_gcr.h"
#include "inverters/generic_cg.h"
#include "inverters/generic_bicgstab_l.h"
#include "inverters/generic_minres.h"
#include "inverters/generic_richardson.h"
#include "inverters/generic_gcr_var_precond.h"

// QMG
#include "stencil/stencil_2d.h"
#include "multigrid/multigrid.h"

// A structure that indicates where different dslash
// counts can come from.
enum QMGDslashType
{
  QMG_DSLASH_TYPE_NULLVEC = 0,
  QMG_DSLASH_TYPE_KRYLOV = 1,
  QMG_DSLASH_TYPE_PRESMOOTH = 2,
  QMG_DSLASH_TYPE_POSTSMOOTH = 3,
};


// A class that inherits from MultigridMG, adding extra functions
// which track the state of an MG solve, as well as what type of 
// preconditioning to use at each level. 
class StatefulMultigridMG : public MultigridMG
{
private:
  // Get rid of copy, assignment.
  StatefulMultigridMG(StatefulMultigridMG const &);
  StatefulMultigridMG& operator=(StatefulMultigridMG const &);

  int current_level;

public: 

  // Structure that contains information about the solve
  // at a given level: how to perform the outer solve,
  // as well as the pre-smooth and post-smooth. This
  // object exists for level except the last, though the
  // outer solve info is ignored if a given level
  // is used as the outermost level in a recursive solve.
  // There is a separate object for the solve of the
  // coarsest level.
  struct LevelSolveMG
  {
    /* FINE OP */

    // Whether we're using the original, rbjacobi, or schur solve.
    QMGStencilType fine_stencil_app;

    /* OUTER SOLVE */

    // Tolerance for outer solve.
    double intermediate_tol;

    // Maximum number of iterations for outer solve.
    int intermediate_iters;

    // Restart frequency. -1 means don't restart.
    int intermediate_restart_freq;

    /* PRESMOOTHER */

    // Tolerance for presmoother (for a flexible presmoother)
    double pre_tol;

    // Number of iterations for presmoother
    int pre_iters;

    // Use CGNE for the smoother instead of MR
    bool pre_cgne;

    // What solver to use, relaxation params, other params,
    // preconditioning...

    /* POSTSMOOTHER */

    // Tolerance for postsmoother (for a flexible postsmoother)
    double post_tol;

    // Number of iterations for presmoother
    int post_iters;

    // Use CGNE for the smoother instead of MR
    bool post_cgne;

    // By default, there's "no" stopping condition.
    LevelSolveMG()
      : fine_stencil_app(QMG_MATVEC_ORIGINAL), 
        intermediate_tol(1e-20), intermediate_iters(10000000),
        intermediate_restart_freq(32),
        pre_tol(1e-20), pre_iters(1000000), pre_cgne(false),
        post_tol(1e-20), post_iters(1000000), post_cgne(false)
    { ; }

  };

  // Class that tracks how many Dslash operations are
  // performed at each iteration. 
  class DslashTrackerMG
  {
  private:
    // Get rid of copy, assignment.
    DslashTrackerMG(DslashTrackerMG const &);
    DslashTrackerMG& operator=(DslashTrackerMG const &);

    // Track different dslash counts.
    std::map<QMGDslashType,int> tracker;
    int iterations;
    int total;

  public:

    // Constructor.
    DslashTrackerMG()
    {
      // Set initial values.
      tracker[QMG_DSLASH_TYPE_NULLVEC] = 0;
      tracker[QMG_DSLASH_TYPE_KRYLOV] = 0;
      tracker[QMG_DSLASH_TYPE_PRESMOOTH] = 0;
      tracker[QMG_DSLASH_TYPE_POSTSMOOTH] = 0;
      total = 0;
      iterations = 0;
    }

    // Nothing in destructor.

    // Update routine.
    void add_tracker_count(QMGDslashType type, int accum)
    {
      tracker[type] += accum;
      total += accum;
    }

    // Count Krylov iterations. 
    void add_iterations_count(int accum)
    {
      iterations += accum;
    }

    // Add all non-nullvec counts into null vecs. Relevant for
    // adaptive setups.
    void shift_all_to_nullvec()
    {
      tracker[QMG_DSLASH_TYPE_NULLVEC] += tracker[QMG_DSLASH_TYPE_KRYLOV];
      tracker[QMG_DSLASH_TYPE_KRYLOV] = 0;
      tracker[QMG_DSLASH_TYPE_NULLVEC] += tracker[QMG_DSLASH_TYPE_PRESMOOTH];
      tracker[QMG_DSLASH_TYPE_PRESMOOTH] = 0;
      tracker[QMG_DSLASH_TYPE_NULLVEC] += tracker[QMG_DSLASH_TYPE_POSTSMOOTH];
      tracker[QMG_DSLASH_TYPE_POSTSMOOTH] = 0;
      iterations = 0;
    }

    // get_count
    int get_tracker_count(QMGDslashType type)
    {
      return tracker[type];
    }

    int get_total_count()
    {
      return total;
    }

    int get_iterations_count()
    {
      return iterations;
    }

    // Reset
    void reset_tracker()
    {
      // Set initial values.
      tracker[QMG_DSLASH_TYPE_NULLVEC] = 0;
      tracker[QMG_DSLASH_TYPE_KRYLOV] = 0;
      tracker[QMG_DSLASH_TYPE_PRESMOOTH] = 0;
      tracker[QMG_DSLASH_TYPE_POSTSMOOTH] = 0;
      total = 0;
      iterations = 0;
    }

  };


  // Structure that describes how to do the coarsest solve.
  struct CoarsestSolveMG
  {
    /* FINE OP */

    // Whether we're using the original, rbjacobi, schur,
    // CGNE, or CGNR solve.
    QMGStencilType coarsest_stencil_app;

    /* OUTER SOLVE */

    // Tolerance for outer solve.
    double coarsest_tol;

    // Maximum number of iterations for outer solve.
    int coarsest_iters;

    // Restart frequency. -1 means don't restart.
    int coarsest_restart_freq;

#ifndef NO_ARPACK
    // Deflate coarsest. Ignored if we aren't doing CGNE/CGNR
    bool deflate;
#endif

    // Shift coarsest. Ignored if we aren't doing CGNE/CGNR.
    double normal_shift;

    // By default, there's "no" stopping condition.
    CoarsestSolveMG()
      : coarsest_stencil_app(QMG_MATVEC_ORIGINAL),
        coarsest_tol(1e-20), coarsest_iters(100000000),
        coarsest_restart_freq(32),
#ifndef NO_ARPACK
        deflate(true),
#endif
        normal_shift(0.0)
    { ; }
  };

protected:

  // Vector of LevelSolveMG. Should be of length mg_object->get_num_levels()-1.
  vector<LevelSolveMG*> level_solve_list;

  // Information on Dslash counts on each level.
  // Should always be of length mg_object->get_num_levels().
  vector<DslashTrackerMG*> dslash_tracker_list;

  // Information on the coarsest level solve.
  CoarsestSolveMG* coarsest_solve; 

#ifndef NO_ARPACK
  // Storage for eigenvalues, vectors on coarsest level.
  int coarsest_deflated;
  complex<double>* coarsest_evals;
  complex<double>** coarsest_evecs;
#endif
public:

  // Simple constructor.
  StatefulMultigridMG(Lattice2D* in_lat, Stencil2D* in_stencil, CoarsestSolveMG* in_coarsest_solve)
    : MultigridMG(in_lat, in_stencil), coarsest_solve(in_coarsest_solve)
#ifndef NO_ARPACK
      , coarsest_deflated(0), coarsest_evals(0), coarsest_evecs(0)
#endif
  {
    // Set the current level to zero.
    current_level = 0;

    // Prepare dslash counter.
    dslash_tracker_list.push_back(new DslashTrackerMG());
  }

  ~StatefulMultigridMG()
  {
    for (int i = 0; i < get_num_levels(); i++)
    {
      delete dslash_tracker_list[i];
      dslash_tracker_list[i] = 0;
    }

#ifndef NO_ARPACK
    // Clean up eigenvectors as appropriate.
    if (coarsest_evals != 0)
    {
      delete[] coarsest_evals;
    }
    if (coarsest_evecs != 0)
    {
      for (int i = 0; i < coarsest_deflated; i++)
      {
        if (coarsest_evecs[i] != 0)
          deallocate_vector(&coarsest_evecs[i]);
      }
      delete[] coarsest_evecs;
    }
#endif
  }

  // Set the multigrid level.
  void set_multigrid_level(int level)
  {
    if (level >= 0 && level < get_num_levels())
    {
      current_level = level;
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: StatefulMultigridMG->current_level " << level << " is outside of [0,max_level-1].\n";
    }
  }

  // Go one level finer.
  void go_finer()
  {
    if (current_level > 0)
    {
      current_level--;
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot go finer than the top level in StatefulMultigridMG.\n";
    }
  }

  // Go one level coarser.
  void go_coarser()
  {
    if (current_level < get_num_levels()-2)
    {
      current_level++;
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot go coarser than the second-coarsest level in StatefulMultigridMG.\n";
    }
  }

  // Get the level.
  int get_multigrid_level()
  {
    return current_level;
  }

  // Get the LevelSolveMG structure for a given fine level.
  LevelSolveMG* get_level_solve(int i)
  {
    if (i >= 0 && i < get_num_levels()-1 && level_solve_list[i] != 0)
    {
      return level_solve_list[i];
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: LevelSolveMG level " << i << " does not exist in StatefulMultigridMG object.\n";
      return 0;
    }
  }

  LevelSolveMG* get_level_solve()
  {
    return get_level_solve(current_level);
  }

  // Get the CoarsestSolveMG structure.
  CoarsestSolveMG* get_coarsest_solve()
  {
    return coarsest_solve;
  }

  // Virtual overloads of "push_level" which set the level_solve to a zero pointer...
  void push_level(Lattice2D* new_lat, TransferMG* new_transfer, bool build_stencil = false, bool is_chiral = false, QMGMultigridPrecondStencil build_stencil_from = QMG_MULTIGRID_PRECOND_ORIGINAL, CoarseOperator2D::QMGCoarseBuildStencil build_extra = CoarseOperator2D::QMG_COARSE_BUILD_ORIGINAL, complex<double>** nvecs = 0)
  {
    MultigridMG::push_level(new_lat, new_transfer, build_stencil, is_chiral, build_stencil_from, build_extra, nvecs);
    level_solve_list.push_back(0); // oiiii
    dslash_tracker_list.push_back(new DslashTrackerMG());
  }

  void push_level(Lattice2D* new_lat, TransferMG* new_transfer, bool build_stencil = false, bool is_chiral = false, QMGMultigridPrecondStencil build_stencil_from = QMG_MULTIGRID_PRECOND_ORIGINAL, complex<double>** nvecs = 0)
  {
    MultigridMG::push_level(new_lat, new_transfer, build_stencil, is_chiral, build_stencil_from, nvecs);
    level_solve_list.push_back(0); // oiiii
    dslash_tracker_list.push_back(new DslashTrackerMG());
  }

  void push_level(Lattice2D* new_lat, TransferMG* new_transfer,complex<double>** nvecs)
  {
    MultigridMG::push_level(new_lat, new_transfer, nvecs);
    level_solve_list.push_back(0); // oiiii
    dslash_tracker_list.push_back(new DslashTrackerMG());
  }

  // Safe versions of push_level which also push a new LevelSolveMG object.
  // See multigrid.h for the purpose of the other arguments! 
  void push_level(Lattice2D* new_lat, TransferMG* new_transfer, LevelSolveMG* in_solve, bool build_stencil = false, bool is_chiral = false, QMGMultigridPrecondStencil build_stencil_from = QMG_MULTIGRID_PRECOND_ORIGINAL, CoarseOperator2D::QMGCoarseBuildStencil build_extra = CoarseOperator2D::QMG_COARSE_BUILD_ORIGINAL, complex<double>** nvecs = 0)
  {
    MultigridMG::push_level(new_lat, new_transfer, build_stencil, is_chiral, build_stencil_from, build_extra, nvecs);

    // Push the level info. An outer solve can only
    // be QMG_MATVEC_ORIGINAL, QMG_MATVEC_RIGHT_JACOBI, QMG_MATVEC_RIGHT_SCHUR.
    if (in_solve->fine_stencil_app != QMG_MATVEC_ORIGINAL &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_JACOBI &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_SCHUR)
    {
      std::cout << "[QMG-ERROR]: In StatefulMultigridMG:;push_level, LevelSolveMG::fine_stencil_app should only be original, right jacobi, or schur.\n";
    }

    level_solve_list.push_back(in_solve);

    dslash_tracker_list.push_back(new DslashTrackerMG());
  }

  void push_level(Lattice2D* new_lat, TransferMG* new_transfer, LevelSolveMG* in_solve, bool build_stencil = false, bool is_chiral = false, QMGMultigridPrecondStencil build_stencil_from = QMG_MULTIGRID_PRECOND_ORIGINAL, complex<double>** nvecs = 0)
  {
    MultigridMG::push_level(new_lat, new_transfer, build_stencil, is_chiral, build_stencil_from, nvecs);

    // Push the level info. An outer solve can only
    // be QMG_MATVEC_ORIGINAL, QMG_MATVEC_RIGHT_JACOBI, QMG_MATVEC_RIGHT_SCHUR.
    if (in_solve->fine_stencil_app != QMG_MATVEC_ORIGINAL &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_JACOBI &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_JACOBI)
    {
      std::cout << "[QMG-ERROR]: In StatefulMultigridMG:;push_level, LevelSolveMG::fine_stencil_app should only be original, right jacobi, or schur.\n";
    }

    level_solve_list.push_back(in_solve);

    dslash_tracker_list.push_back(new DslashTrackerMG());
  }

  void push_level(Lattice2D* new_lat, TransferMG* new_transfer, LevelSolveMG* in_solve, complex<double>** nvecs)
  {
    MultigridMG::push_level(new_lat, new_transfer, nvecs);

    // Push the level info. An outer solve can only
    // be QMG_MATVEC_ORIGINAL, QMG_MATVEC_RIGHT_JACOBI, QMG_MATVEC_RIGHT_SCHUR.
    if (in_solve->fine_stencil_app != QMG_MATVEC_ORIGINAL &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_JACOBI &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_JACOBI)
    {
      std::cout << "[QMG-ERROR]: In StatefulMultigridMG:;push_level, LevelSolveMG::fine_stencil_app should only be original, right jacobi, or schur.\n";
    }

    level_solve_list.push_back(in_solve);

    dslash_tracker_list.push_back(new DslashTrackerMG());
  }

  // Overloaded version of pop_level which also pops the level_solve_list.
  void pop_level()
  {

    // Pop level solve info.
    level_solve_list.pop_back();

    int i = num_levels-1;

    // Deallocate vectors.
    if (dslash_tracker_list[i] != 0)
    {
      delete dslash_tracker_list[i];
      dslash_tracker_list.pop_back();
    }

    MultigridMG::pop_level();

  }

  // A function to update a level. Sort of good blend of push and pop level.
  void update_level(int level, Lattice2D* new_lat, TransferMG* new_transfer, LevelSolveMG* in_solve, bool build_stencil = false, bool is_chiral = false, QMGMultigridPrecondStencil build_stencil_from = QMG_MULTIGRID_PRECOND_ORIGINAL, CoarseOperator2D::QMGCoarseBuildStencil build_extra = CoarseOperator2D::QMG_COARSE_BUILD_ORIGINAL, complex<double>** nvecs = 0)
  {
    // Push the level info. An outer solve can only
    // be QMG_MATVEC_ORIGINAL, QMG_MATVEC_RIGHT_JACOBI, QMG_MATVEC_RIGHT_SCHUR.
    if (in_solve->fine_stencil_app != QMG_MATVEC_ORIGINAL &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_JACOBI &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_SCHUR)
    {
      std::cout << "[QMG-ERROR]: In StatefulMultigridMG:;update_level, LevelSolveMG::fine_stencil_app should only be original, right jacobi, or schur.\n";
      return; 
    }

    MultigridMG::update_level(level, new_lat, new_transfer, build_stencil, is_chiral, build_stencil_from, build_extra, nvecs);

    
    // Update level solve.
    level_solve_list[level-1] = in_solve;

    // Preserve the dslash_tracker!

  }

  void update_level(int level, Lattice2D* new_lat, TransferMG* new_transfer, LevelSolveMG* in_solve, bool build_stencil = false, bool is_chiral = false, QMGMultigridPrecondStencil build_stencil_from = QMG_MULTIGRID_PRECOND_ORIGINAL, complex<double>** nvecs = 0)
  {
    update_level(level, new_lat, new_transfer, in_solve, build_stencil, is_chiral, build_stencil_from, CoarseOperator2D::QMG_COARSE_BUILD_ORIGINAL, nvecs);
  }

  // Update the track at a given level.
  void add_tracker_count(QMGDslashType type, int accum, int i)
  {
    if (i >= 0 && i < num_levels)
    {
      dslash_tracker_list[i]->add_tracker_count(type, accum);
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot update tracker at level " << i << ".\n";
    }
  }

  // Count Krylov iterations.
  void add_iterations_count(int accum, int i)
  {
    if (i >= 0 && i < num_levels)
    {
      dslash_tracker_list[i]->add_iterations_count(accum);
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot update tracker at level " << i << ".\n";
    }
  }

  // Add all non-nullvec counts into null vecs. Relevant for
  // adaptive setups.
  void shift_all_to_nullvec(int i)
  {
    if (i >= 0 && i < num_levels)
    {
      dslash_tracker_list[i]->shift_all_to_nullvec();
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot shift to null vectors at level " << i << ".\n";
    }
  }

  // Query the track at a given level.
  int get_tracker_count(QMGDslashType type, int i)
  {
    if (i >= 0 && i < num_levels)
    {
      return dslash_tracker_list[i]->get_tracker_count(type);
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot query tracker at level " << i << ".\n";
      return -1;
    }
  }

  // Get total number of ops
  int get_total_count(int i)
  {
    if (i >= 0 && i < num_levels)
    {
      return dslash_tracker_list[i]->get_total_count();
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot query tracker at level " << i << ".\n";
      return -1;
    }
  }

  // Get total number of ops
  int get_iterations_count(int i)
  {
    if (i >= 0 && i < num_levels)
    {
      return dslash_tracker_list[i]->get_iterations_count();
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot query tracker at level " << i << ".\n";
      return -1;
    }
  }

  // Get average iterations per level
  std::vector<double> query_average_iterations()
  {
    std::vector<double> avg(num_levels);
    avg[0] = dslash_tracker_list[0]->get_iterations_count();
    for (int i = 1; i < num_levels; i++)
    {
      avg[i] = ((double)dslash_tracker_list[i]->get_iterations_count())/((double)dslash_tracker_list[i-1]->get_iterations_count());
    }
    return avg; 
  }

  // Reset the tracker at a given level, or all for -1.
  void reset_tracker(int i = -1)
  {
    if (i == -1)
    {
      for (int j = 0; j < num_levels; j++)
        dslash_tracker_list[j]->reset_tracker();
    }
    else if (i >= 0 && i < num_levels)
    {
      dslash_tracker_list[i]->reset_tracker();
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot reset tracker at level " << i << ".\n";
    }
  }

#ifndef NO_ARPACK
  // Routine to deflate coarsest level.
  void deflate_coarsest(int num_low, int num_high, bool print_evals = false)
  {
#ifndef QLINALG_INTERFACE_ARPACK
    std::cout << "[QMG-ERROR]: Cannot deflate coarest operator without ARPACK support.\n";
    return;
#else

    if (!coarsest_solve->deflate)
    {
      std::cout << "[QMG-WARNING]: Coarsest level is not set to deflate. Skipping computing eigenvectors.\n";
    }

    if (coarsest_solve->coarsest_stencil_app != QMG_MATVEC_M_MDAGGER &&
          coarsest_solve->coarsest_stencil_app != QMG_MATVEC_MDAGGER_M &&
          coarsest_solve->coarsest_stencil_app != QMG_MATVEC_RBJ_M_MDAGGER &&
          coarsest_solve->coarsest_stencil_app != QMG_MATVEC_RBJ_MDAGGER_M)
    {
      std::cout << "[QMG-ERROR]: Cannot deflate coarsest operator unless it's a normal op solve.\n";
      return;
    }

    if (coarsest_deflated != 0 || coarsest_evals != 0 || coarsest_evecs != 0)
    {
      std::cout << "[QMG-WARNING]: Coarsest operator space already deflated.\n";
      return;
    }

    if (num_low + num_high == 0)
    {
      return;
    }
    
    coarsest_deflated = num_low + num_high;
    coarsest_evals = new complex<double>[coarsest_deflated];
    coarsest_evecs = new complex<double>*[coarsest_deflated];
    for (int i = 0; i < coarsest_deflated; i++)
    {
      coarsest_evecs[i] = allocate_vector<complex<double>>(get_stencil(get_num_levels()-1)->get_lattice()->get_size_cv());
    }

    // Declare an arpack object.
    arpack_dcn* arpack;

    if (num_low > 0)
    {
      // Grab bottom of spectrum.
      arpack = new arpack_dcn(get_stencil(get_num_levels()-1)->get_lattice()->get_size_cv(), 100000, 1e-5,
                    get_stencil(get_num_levels()-1)->get_apply_function(coarsest_solve->coarsest_stencil_app),
                    get_stencil(get_num_levels()-1),
                    num_low, 3*num_low);

      arpack->prepare_eigensystem(arpack_dcn::ARPACK_SMALLEST_REAL, num_low, 3*num_low);
      arpack->get_eigensystem(coarsest_evals, coarsest_evecs, arpack_dcn::ARPACK_SMALLEST_REAL);
      delete arpack;
    }

    if (num_high > 0)
    {
      // Get top of spectrum.
      arpack = new arpack_dcn(get_stencil(get_num_levels()-1)->get_lattice()->get_size_cv(), 100000, 1e-5,
                    get_stencil(get_num_levels()-1)->get_apply_function(coarsest_solve->coarsest_stencil_app),
                    get_stencil(get_num_levels()-1),
                    num_high, 3*num_high);

      arpack->prepare_eigensystem(arpack_dcn::ARPACK_LARGEST_REAL, num_high, 3*num_high);
      arpack->get_eigensystem(coarsest_evals + num_low, coarsest_evecs + num_low, arpack_dcn::ARPACK_SMALLEST_REAL);
      delete arpack;
    }

    for (int i = 0; i < coarsest_deflated; i++)
    {
      normalize(coarsest_evecs[i], get_stencil(get_num_levels()-1)->get_lattice()->get_size_cv());
    }

    if (print_evals)
    {
      for (int i = 0; i < coarsest_deflated; i++)
      {
        std::cout << "[QMG-COARSEST-EVALS]: " << i << " " << real(coarsest_evals[i]) << "\n";
      }
    }

#endif
  }

  unsigned int get_coarsest_deflated()
  {
    return coarsest_deflated;
  }

  complex<double>* get_coarsest_evals()
  {
    return coarsest_evals;
  }

  complex<double>** get_coarsest_evecs()
  {
    return coarsest_evecs;
  }
#endif // ifndef NO_ARPACK

protected:
  // Special structure, function if we need to do a shifted solve.
  struct ShiftedFunctionStruct
  {
    matrix_op_cplx function;
    void* extra_data;
    complex<double> extra_shift;
    int length;
  };

  static void shift_function(complex<double>* out, complex<double>* in, void* data)
  {
    ShiftedFunctionStruct* shift_struct = (ShiftedFunctionStruct*)data;
    shift_struct->function(out, in, shift_struct->extra_data);
    caxpy(shift_struct->extra_shift, in, out, shift_struct->length);
  }

public:

  // Need to add counters for prepare/reconstruct. 
  static void mg_preconditioner(complex<double>* lhs, complex<double>* rhs, int size, void* extra_data, inversion_verbose_struct* verb)
  {
    // Expose the Multigrid objects.

    // State.
    StatefulMultigridMG* mg_object = (StatefulMultigridMG*)extra_data;
    int level = mg_object->get_multigrid_level();
    //cout << "Entered level " << level << "\n" << flush;

    // Stencils.
    Stencil2D* fine_stencil = mg_object->get_stencil(level);
    Stencil2D* coarse_stencil = mg_object->get_stencil(level+1);

    // Transfer object.
    TransferMG* transfer = mg_object->get_transfer(level);

    // Storage objects.
    ArrayStorageMG<complex<double>>* fine_storage = mg_object->get_storage(level);
    ArrayStorageMG<complex<double>>* coarse_storage = mg_object->get_storage(level+1);

    // Sizes.
    int fine_size = mg_object->get_lattice(level)->get_size_cv();
    int coarse_size = mg_object->get_lattice(level+1)->get_size_cv();

    // Number of levels.
    int total_num_levels = mg_object->get_num_levels();

    // Verbosity and inversion info. Currently this is hacky.
    inversion_info invif;
    inversion_verbose_struct verb2(VERB_SUMMARY, std::string(" "));
    if (verb->verbosity == VERB_NONE)
    {
      verb2.verbosity = VERB_NONE;
      verb2.precond_verbosity = VERB_NONE;
    }
    else
    {
      verb2.precond_verbosity = VERB_SUMMARY;
    }
    verb2.verb_prefix = "  ";
    for (int i = 1; i < level+1; i++)
      verb2.verb_prefix += "  ";
    verb2.verb_prefix += "[QMG-MG-SOLVE-INFO]: Level " + to_string(level+1) + " ";

    // Get info on pre- and post-smooth.
    LevelSolveMG* level_solve = mg_object->get_level_solve();
    if (level_solve == 0)
    {
      std::cout << "[QMG-MG-SOLVE-ERROR]: Level solve for level " << level << " does not exist.\n";
      return;
    }
    int n_pre_smooth = level_solve->pre_iters;
    double pre_smooth_tol = level_solve->pre_tol;
    bool pre_smooth_cgne = level_solve->pre_cgne;
    int n_post_smooth = level_solve->post_iters;
    double post_smooth_tol = level_solve->post_tol;
    bool post_smooth_cgne = level_solve->post_cgne;

    // Function for what type of fine apply we need to do.
    QMGStencilType fine_stencil_type = level_solve->fine_stencil_app;
    matrix_op_cplx apply_fine_M = Stencil2D::get_apply_function(level_solve->fine_stencil_app);

    // The fine solve depends on if we're doing a schur solve
    // or not.
    int fine_size_solve = fine_size;
    if (fine_stencil_type == QMG_MATVEC_RIGHT_SCHUR)
      fine_size_solve /= 2;

    // If the number of levels = 1, there's no cycle. Copy and return.
    if (total_num_levels == 1)
    {
      copy_vector(lhs, rhs, fine_size_solve);
      return;
    }

    // Learn about coarse solve.
    int coarse_max_iter;
    double coarse_tol;
    int coarse_restart;
    QMGStencilType coarse_stencil_type;
    if (level < total_num_levels-2)
    {
      coarse_stencil_type = mg_object->get_level_solve(level+1)->fine_stencil_app;
      coarse_max_iter = mg_object->get_level_solve(level+1)->intermediate_iters;
      coarse_tol = mg_object->get_level_solve(level+1)->intermediate_tol;
      coarse_restart = mg_object->get_level_solve(level+1)->intermediate_restart_freq;
    }
    else // coarsest solve.
    {
      coarse_stencil_type = mg_object->get_coarsest_solve()->coarsest_stencil_app;
      coarse_max_iter = mg_object->get_coarsest_solve()->coarsest_iters;
      coarse_tol = mg_object->get_coarsest_solve()->coarsest_tol;
      coarse_restart = mg_object->get_coarsest_solve()->coarsest_restart_freq;
    }
    // Function for what type of coarse apply we need to do.
    matrix_op_cplx apply_coarse_M = Stencil2D::get_apply_function(coarse_stencil_type);

    // The coarse size depends on if we're doing a schur solve
    // or not.
    int coarse_size_solve = coarse_size;
    if (coarse_stencil_type == QMG_MATVEC_RIGHT_SCHUR)
      coarse_size_solve /= 2;

    // We need a temporary vector for mat-vecs everywhere.
    complex<double>* Atmp = fine_storage->check_out();

    // Step 1: presmooth.
    // Solve A z1 = rhs, form new residual r1 = rhs - A z1
    complex<double>* z1 = fine_storage->check_out();
    zero_vector(z1, fine_size);
    complex<double>* r1 = fine_storage->check_out(); // gets initialized in the next code block
    if (n_pre_smooth > 0)
    {
      if (pre_smooth_cgne && (fine_stencil_type == QMG_MATVEC_ORIGINAL || fine_stencil_type == QMG_MATVEC_RIGHT_JACOBI))
      {
        complex<double>* z1_prec = fine_storage->check_out();
        zero_vector(z1_prec, fine_size);
        invif = minv_vector_minres(z1_prec, rhs, fine_size_solve, n_pre_smooth, pre_smooth_tol, 0.85,
                  Stencil2D::get_apply_function(fine_stencil_type == QMG_MATVEC_ORIGINAL ? QMG_MATVEC_M_MDAGGER : QMG_MATVEC_RBJ_M_MDAGGER), (void*)fine_stencil);
        fine_stencil->apply_M(z1, z1_prec, fine_stencil_type == QMG_MATVEC_ORIGINAL ? QMG_MATVEC_DAGGER : QMG_MATVEC_RBJ_DAGGER);
        mg_object->add_tracker_count(QMG_DSLASH_TYPE_PRESMOOTH, 2*invif.ops_count+1, level); // smoother ops + residual. 
        fine_storage->check_in(z1_prec);
      }
      else
      {
        //invif = minv_vector_gcr_restart(z1, rhs, fine_size_solve, n_pre_smooth, pre_smooth_tol, coarse_restart, apply_fine_M, (void*)fine_stencil);
        invif = minv_vector_minres(z1, rhs, fine_size_solve, n_pre_smooth, pre_smooth_tol, 0.85, apply_fine_M, (void*)fine_stencil);
        mg_object->add_tracker_count(QMG_DSLASH_TYPE_PRESMOOTH, invif.ops_count, level); // smoother ops + residual. 
      }
      zero_vector(Atmp, fine_size);
      fine_stencil->apply_M(Atmp, z1, fine_stencil_type);
      mg_object->add_tracker_count(QMG_DSLASH_TYPE_PRESMOOTH, 1, level); // smoother ops + residual. 
      caxpbyz(1.0, rhs, -1.0, Atmp, r1, fine_size_solve);
    }
    else
    {
      zero_vector(Atmp, fine_size_solve);
      copy_vector(r1, rhs, fine_size_solve);
      copy_vector(z1, rhs, fine_size_solve);
    }

    // Next stop: restrict, prep for coarse solve, recurse (or coarsest solve), prolong.
    complex<double>* r_coarse = coarse_storage->check_out();
    zero_vector(r_coarse, coarse_size);
    transfer->restrict_f2c(r1, r_coarse);
    fine_storage->check_in(r1);
    double rnorm = sqrt(norm2sq(r_coarse, coarse_size));
    complex<double>* r_coarse_prep = coarse_storage->check_out();
    zero_vector(r_coarse_prep, coarse_size);
    coarse_stencil->prepare_M(r_coarse_prep, r_coarse, coarse_stencil_type);
    double rnorm_prep = sqrt(norm2sq(r_coarse_prep, coarse_size));
    complex<double>* e_coarse = coarse_storage->check_out();
    zero_vector(e_coarse, coarse_size);
    if (level == total_num_levels-2) // if we're already on the coarsest level
    {
      bool coarsest_normal = (coarse_stencil_type == QMG_MATVEC_M_MDAGGER ||
                              coarse_stencil_type == QMG_MATVEC_MDAGGER_M ||
                              coarse_stencil_type == QMG_MATVEC_RBJ_M_MDAGGER ||
                              coarse_stencil_type == QMG_MATVEC_RBJ_MDAGGER_M);
#ifndef NO_ARPACK
      // Deflate if we're set to.
      if (coarsest_normal && mg_object->get_coarsest_solve()->deflate && mg_object->get_coarsest_deflated() > 0)
      {
        int num_evecs = mg_object->get_coarsest_deflated();
        complex<double>* evals = mg_object->get_coarsest_evals();
        complex<double>** evecs = mg_object->get_coarsest_evecs();

        // Deflate it upppppp.
        for (int i = 0; i < num_evecs; i++)
        {
          complex<double> bra_n_ket_b = dot(evecs[i], r_coarse_prep, coarse_size_solve);
          caxpy(bra_n_ket_b/evals[i], evecs[i], e_coarse, coarse_size_solve);
        }
      }
#endif

      if (coarse_restart == -1)
      {
        // Need to add norm factor to get proper un-preperared norm.
        if (!coarsest_normal)
        {
          invif = minv_vector_gcr(e_coarse, r_coarse_prep, coarse_size_solve,
                            coarse_max_iter, coarse_tol*rnorm/rnorm_prep, 
                            apply_coarse_M, (void*)coarse_stencil, &verb2);
        }
        else
        {
          if (mg_object->get_coarsest_solve()->normal_shift != 0.0)
          {
            ShiftedFunctionStruct shift_struct;
            shift_struct.function = apply_coarse_M;
            shift_struct.extra_data = (void*)coarse_stencil;
            shift_struct.extra_shift = mg_object->get_coarsest_solve()->normal_shift;
            shift_struct.length = coarse_size_solve;
            invif = minv_vector_cg(e_coarse, r_coarse_prep, coarse_size_solve,
                            coarse_max_iter, coarse_tol*rnorm/rnorm_prep, 
                            shift_function, (void*)&shift_struct, &verb2);
          }
          else
          {
            invif = minv_vector_cg(e_coarse, r_coarse_prep, coarse_size_solve,
                            coarse_max_iter, coarse_tol*rnorm/rnorm_prep, 
                            apply_coarse_M, (void*)coarse_stencil, &verb2);
          }
        }
      }
      else
      {
        /*invif = minv_vector_bicgstab_l(e_coarse, r_coarse_prep, coarse_size_solve,
                          coarse_max_iter, coarse_tol*rnorm/rnorm_prep, 6, 
                          apply_coarse_M, (void*)coarse_stencil, &verb2);*/
        if (!coarsest_normal)
        {
          invif = minv_vector_gcr_restart(e_coarse, r_coarse_prep, coarse_size_solve,
                            coarse_max_iter, coarse_tol*rnorm/rnorm_prep, coarse_restart, 
                            apply_coarse_M, (void*)coarse_stencil, &verb2);
        }
        else
        {
          if (mg_object->get_coarsest_solve()->normal_shift != 0.0)
          {
            ShiftedFunctionStruct shift_struct;
            shift_struct.function = apply_coarse_M;
            shift_struct.extra_data = (void*)coarse_stencil;
            shift_struct.extra_shift = mg_object->get_coarsest_solve()->normal_shift;
            shift_struct.length = coarse_size_solve;
            invif = minv_vector_cg_restart(e_coarse, r_coarse_prep, coarse_size_solve,
                            coarse_max_iter, coarse_tol*rnorm/rnorm_prep, coarse_restart, 
                            shift_function, (void*)&shift_struct, &verb2);
          }
          else
          {
            invif = minv_vector_cg_restart(e_coarse, r_coarse_prep, coarse_size_solve,
                            coarse_max_iter, coarse_tol*rnorm/rnorm_prep, coarse_restart, 
                            apply_coarse_M, (void*)coarse_stencil, &verb2);
          }
        }
      }
    }
    else
    {
      // Recurse.
      mg_object->go_coarser();
      // K cycle
      if (coarse_restart == -1)
      {
        invif = minv_vector_gcr_var_precond(e_coarse, r_coarse_prep, coarse_size_solve,
                          coarse_max_iter, coarse_tol*rnorm/rnorm_prep,
                          apply_coarse_M, (void*)coarse_stencil,
                          mg_preconditioner, (void*)mg_object, &verb2);
      }
      else
      {
        invif = minv_vector_gcr_var_precond_restart(e_coarse, r_coarse_prep, coarse_size_solve,
                          coarse_max_iter, coarse_tol*rnorm/rnorm_prep, coarse_restart,
                          apply_coarse_M, (void*)coarse_stencil,
                          mg_preconditioner, (void*)mg_object, &verb2);
      }
      // V cycle
      //mg_preconditioner(e_coarse, r_coarse, coarse_size, (void*)stateful_mg_object);
      mg_object->go_finer();
    }
    mg_object->add_tracker_count(QMG_DSLASH_TYPE_KRYLOV, invif.ops_count, level+1);
    mg_object->add_iterations_count(invif.iter, level+1);
    //cout << "Level " << level << " coarse preconditioner took " << invif.iter << " iterations.\n" << flush;
    coarse_storage->check_in(r_coarse_prep);
    complex<double>* e_coarse_reconstruct = coarse_storage->check_out();
    zero_vector(e_coarse_reconstruct, coarse_size);
    coarse_stencil->reconstruct_M(e_coarse_reconstruct, e_coarse, r_coarse, coarse_stencil_type);


    /*complex<double>* tmp = coarse_storage->check_out();
    zero_vector(tmp, coarse_size);
    coarse_stencil->apply_M(tmp, e_coarse_reconstruct);
    cout << "Coarse solve level " << level << " has norm2sqdiff " << sqrt(diffnorm2sq(tmp, r_coarse, coarse_size))/sqrt(norm2sq(r_coarse, coarse_size)) << "\n";
    coarse_storage->check_in(tmp);*/
    
    coarse_storage->check_in(r_coarse);
    coarse_storage->check_in(e_coarse);
    complex<double>* z2 = fine_storage->check_out();
    zero_vector(z2, fine_size);
    transfer->prolong_c2f(e_coarse_reconstruct, z2);
    if (coarse_stencil_type == QMG_MATVEC_RIGHT_SCHUR) { zero_vector(z2 + fine_size/2, fine_size/2); }
    coarse_storage->check_in(e_coarse_reconstruct);
    zero_vector(lhs, fine_size_solve);
    cxpyz(z1, z2, lhs, fine_size_solve);
    fine_storage->check_in(z1);
    fine_storage->check_in(z2);

    if (n_post_smooth > 0)
    {
      // Last stop, post smooth. Form r2 = r - A(z1 + z2) = r - Ae, solve A z3 = r2.
      zero_vector(Atmp, fine_size);
      fine_stencil->apply_M(Atmp, lhs, fine_stencil_type);
      complex<double>* r2 = fine_storage->check_out();
      caxpbyz(1.0, rhs, -1.0, Atmp, r2, fine_size_solve);
      complex<double>* z3 = fine_storage->check_out();
      zero_vector(z3, fine_size);

      if (post_smooth_cgne && (fine_stencil_type == QMG_MATVEC_ORIGINAL || fine_stencil_type == QMG_MATVEC_RIGHT_JACOBI))
      {
        complex<double>* z3_prec = fine_storage->check_out();
        zero_vector(z3_prec, fine_size);
        invif = minv_vector_minres(z3_prec, r2, fine_size_solve, n_post_smooth, post_smooth_tol, 0.85,
                  Stencil2D::get_apply_function(fine_stencil_type == QMG_MATVEC_ORIGINAL ? QMG_MATVEC_M_MDAGGER : QMG_MATVEC_RBJ_M_MDAGGER), (void*)fine_stencil);
        fine_stencil->apply_M(z3, z3_prec, fine_stencil_type == QMG_MATVEC_ORIGINAL ? QMG_MATVEC_DAGGER : QMG_MATVEC_RBJ_DAGGER);
        mg_object->add_tracker_count(QMG_DSLASH_TYPE_POSTSMOOTH, 2*(invif.ops_count)+1, level); // smoother ops + residual
        fine_storage->check_in(z3_prec);
      }
      else
      {
        //invif = minv_vector_gcr_restart(z1, rhs, fine_size_solve, n_pre_smooth, pre_smooth_tol, coarse_restart, apply_fine_M, (void*)fine_stencil);
        invif = minv_vector_minres(z3, r2, fine_size_solve, n_post_smooth, post_smooth_tol, 0.85, apply_fine_M, (void*)fine_stencil);
        mg_object->add_tracker_count(QMG_DSLASH_TYPE_POSTSMOOTH, invif.ops_count, level); // smoother ops + residual
      }
      
      cxpy(z3, lhs, fine_size_solve);

      // Check vectors back in.
      
      fine_storage->check_in(r2);
      fine_storage->check_in(z3);
    }
    fine_storage->check_in(Atmp);

    //cout << "Exited level " << level << "\n" << flush;
  }

};

#endif // QMG_STATEFUL_MG
