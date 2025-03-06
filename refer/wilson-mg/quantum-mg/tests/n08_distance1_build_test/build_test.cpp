// Copyright (c) 2017 Evan S Weinberg
// Verify a build of the coarse stencil against prolong-apply fine-restrict.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"

// QMG
#include "lattice/lattice.h"
#include "transfer/transfer.h"
#include "stencil/stencil_2d.h"
#include "multigrid/multigrid.h"

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
  const int x_len = 64;
  const int y_len = 64;
  const int dof = 1; 

  // Information on the Laplace operator.
  const double mass = 0.01;

  // Blocking size.
  const int x_block = 2;
  const int y_block = 2;

  // Number of null vectors. Since we're doing a test, we want 2.
  const int coarse_dof = 1;

  // How many times to refine. 
  const int n_refine = 6; // (64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)

  // Create a lattice object for the fine lattice.
  Lattice2D** lats = new Lattice2D*[n_refine+1];
  lats[0] = new Lattice2D(x_len, y_len, dof);

  // Create a unit gauged laplace stencil.
  complex<double>* unit_gauge = allocate_vector<complex<double>>(lats[0]->get_size_gauge());
  unit_gauge_u1(unit_gauge, lats[0]);
  GaugedLaplace2D* laplace_op = new GaugedLaplace2D(lats[0], mass*mass, unit_gauge);

  // Create a MultigridMG object, push top level onto it!
  MultigridMG* mg_object = new MultigridMG(lats[0], laplace_op);

  // Create coarse lattices, unit null vectors, transfer objects.
  // Push into MultigridMG object. 
  int curr_x_len = x_len;
  int curr_y_len = y_len;
  TransferMG** transfer_objs = new TransferMG*[n_refine];
  for (i = 1; i <= n_refine; i++)
  {
    cout << "Processing level " << i << "\n";
    // Update to the new lattice size.
    curr_x_len /= x_block;
    curr_y_len /= y_block;

    // Create a new lattice object.
    lats[i] = new Lattice2D(curr_x_len, curr_y_len, coarse_dof);

    // Create a new null vector. These are copied into local memory in the
    // transfer object, so we can create and destroy these in this loop.
    complex<double>* null_vector = allocate_vector<complex<double>>(lats[i-1]->get_size_cv());
    constant_vector(null_vector, 1.0, lats[i-1]->get_size_cv());
    complex<double>* null_vector2 = allocate_vector<complex<double>>(lats[i-1]->get_size_cv());
    constant_vector(null_vector2, 1.0, lats[i-1]->get_size_cv()/2);
    constant_vector(null_vector2+lats[i-1]->get_size_cv()/2, -1.0, lats[i-1]->get_size_cv()/2);
    complex<double>** null_vectors = new complex<double>*[2];
    null_vectors[0] = null_vector;
    null_vectors[1] = null_vector2;

    // Create and populate a transfer object.
    // Fine lattice, coarse lattice, null vector(s), perform the block ortho.
    transfer_objs[i-1] = new TransferMG(lats[i-1], lats[i], null_vectors, true);

    // Push a new level on the multigrid object! Also, save the global null vector.
    // Arg 1: New lattice
    // Arg 2: New transfer object (between new and prev lattice)
    // Arg 3: Should we construct the coarse stencil?
    // Arg 4: Is the operator chiral? (No for Laplace)
    // Arg 4: What should we construct the coarse stencil from? (Not relevant yet.)
    // Arg 5: Non-block-orthogonalized null vector.
    mg_object->push_level(lats[i], transfer_objs[i-1], true, GaugedLaplace2D::has_chirality(), MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL, null_vectors);

    //mg_object->get_stencil(i)->print_stencil_site(0,0,"Site: ");

    // Clean up local vector, since they get copied in.
    deallocate_vector(&null_vectors[0]);
    deallocate_vector(&null_vectors[1]);
    delete[] null_vectors; 
  }


  // Compare each coarse stencil with prolong, apply, restrict.
  for (i = 1; i <= n_refine; i++)
  {
    // Get a vector to store a rhs in.
    complex<double>* rhs = mg_object->check_out(i);
    gaussian(rhs, lats[i]->get_size_cv(), generator);

    // Get a vector to store the lhs from applying the stencil in.
    complex<double>* apply_lhs = mg_object->check_out(i);
    zero_vector(apply_lhs, lats[i]->get_size_cv());
    mg_object->apply_stencil(apply_lhs, rhs, i);
    double norm = sqrt(norm2sq(apply_lhs, lats[i]->get_size_cv()));
    //cout << "Level " << i << " apply_lhs norm is " << norm << "\n";

    // Get a vector to store the prolonged rhs in.
    complex<double>* rhs_pro = mg_object->check_out(i-1);
    zero_vector(rhs_pro, lats[i-1]->get_size_cv());
    mg_object->prolong_c2f(rhs, rhs_pro, i-1);

    // Get a vector to apply the fine stencil to.
    complex<double>* Arhs_pro = mg_object->check_out(i-1);
    zero_vector(Arhs_pro, lats[i-1]->get_size_cv());
    mg_object->apply_stencil(Arhs_pro, rhs_pro, i-1);

    // Get a vector to restrict into.
    complex<double>* proAres_lhs = mg_object->check_out(i);
    zero_vector(proAres_lhs, lats[i]->get_size_cv());
    mg_object->restrict_f2c(Arhs_pro, proAres_lhs, i-1);
    //cout << "Level " << i << " proAres_lhs norm is " << sqrt(norm2sq(proAres_lhs, lats[i]->get_size_cv())) << "\n";

    // Compare.
    cout << "Level " << i << " build has comparison norm " << sqrt(diffnorm2sq(apply_lhs, proAres_lhs, lats[i]->get_size_cv()))/norm << "\n";

    // Return vectors.
    mg_object->check_in(rhs, i);
    mg_object->check_in(apply_lhs, i);
    mg_object->check_in(rhs_pro, i-1);
    mg_object->check_in(Arhs_pro, i-1);
    mg_object->check_in(proAres_lhs, i);
  }


  ///////////////
  // Clean up. //
  ///////////////

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

  // Delete lattices.
  for (i = 0; i <= n_refine; i++)
  {
    delete lats[i];
  }
  delete[] lats; 

  return 0;
}
