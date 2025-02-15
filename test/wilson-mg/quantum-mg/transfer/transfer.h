// Copyright (c) 2017 Evan S Weinberg
// Header file for a transfer object, which contains everything (?)
// needed for MG prolong/restrict assuming regular, non-overlapping blocks.
// This file needs OpenMP/ACC pragmas to get thread-level optimization.

// MG prolong/restrict requires:
// * Knowledge of the fine and coarse lattice
// ** Can deduce blocksizes from that because of assumption of regular,
//    non-overlapping blocks.
// * Knowledge of the null vectors.
// ** Don't have to be block-orthonormalized on construction.
// * Knowledge of the one-to-many mapping between the coarse and fine lattices.
// ** Can construct based on coarse and fine lattice, plus assumption of
//    regular, non-overlapping blocks.
// * Implements block-orthonormalization as an internal function.

#ifndef QMG_TRANSFER_OBJECT
#define QMG_TRANSFER_OBJECT

#include <complex>
#include <cmath>
#include <iostream>

using std::complex;
using std::sqrt;
using std::conj;

// QLINALG
#include "blas/generic_vector.h"

// QMG
#include "lattice/lattice.h"

// What type of doubling, if any?
enum QMGDoublingType
{
  QMG_DOUBLE_NONE = 0,
  QMG_DOUBLE_PROJECTION = 1, // apply with a projector.
  QMG_DOUBLE_OPERATOR = 2, // apply by doubling with gamma_5^L/R.
};

class TransferMG
{
private:
  // Get rid of copy, assignment operator.
  TransferMG(TransferMG const &);
  TransferMG& operator=(TransferMG const &);

  // Lattices for fine and coarse lattice.
  Lattice2D* fine_lat;

  Lattice2D* coarse_lat; 

  // For safety during deconstructor, make sure we know the 
  // number of coarse dof ( = number of null vectors)
  int const_num_null_vec;

  // and coarse volume.
  const int const_coarse_volume;

  // Blocksizes in each dimension (deduced from fine, coarse lattice)
  int* blocksizes;

  // One-to-many mapping between coarse sites and fine sites+dof.
  int** coarse_map; 

  // Declaration of internal function to generate mapping between
  // coarse sites and fine sites+dof.
  void build_mapping();

  // Number of fine sites+dof per coarse site.
  int fine_sites_per_coarse; 

  // Hacky hack hack.
public:
  // Local copies of null vectors. Block orthonormalization
  // modifies the null vectors, and we don't want to assume
  // we can perturb the original copies of the null vectors.
  complex<double>** null_vectors; 

  // We do support separate left and right null vectors.
  complex<double>** restrict_null_vectors;

  // If we save the Cholesky (for symmetric) or LU (for asymmetric)
  // from the block (bi-)ortho, we save it here.
  complex<double>* block_cholesky;
  complex<double>* block_L;
  complex<double>* block_U;

private:

  // How we double the space.
  QMGDoublingType doubling;

  // Declaration of internal function to perform block orthonormalization.
  // We define it below.
  void block_orthonormalize();

  // Declaration of internal function to perform block bi-orthonormalization.
  // We define it below.
  void block_bi_orthonormalize();

  // Declarations of more generic prolong/restrict functions.
  // Supports an abuse of prolong/restrict to block-orthonormalize.
  // The user only sees a public coarse in, fine out (or vice versa) fcn.
  void prolong_c2f(complex<double>* coarse_cv, complex<double>* fine_cv, complex<double>** null_vectors, int nvec);
  void restrict_f2c(complex<double>* fine_cv, complex<double>* coarse_cv, complex<double>** null_vectors, int nvec);

  // Is the construction valid?
  bool is_init;
  
public:
  
  // Constructor. Takes in fine and coarse lattice, input null vectors, and
  // a flag about performing block orthonormalization (default true).
  // If the null vectors are pre-block orthonormalized, it might make sense
  // to not waste time re-orthonormalizing them. 
  TransferMG(Lattice2D* in_fine_lat, Lattice2D* in_coarse_lat, complex<double>** in_null_vectors, bool do_block_ortho = true, bool save_decomp = false, QMGDoublingType in_doubling = QMG_DOUBLE_NONE)
    : fine_lat(in_fine_lat), coarse_lat(in_coarse_lat), const_num_null_vec(coarse_lat->get_nc()),
      const_coarse_volume(coarse_lat->get_volume()), blocksizes(0), coarse_map(0),
      null_vectors(0), restrict_null_vectors(0), 
      block_cholesky(0), block_L(0), block_U(0),
      doubling(in_doubling), is_init(false)
  {
    // Learn the blocksizes. 
    blocksizes = new int[fine_lat->get_nd()];
    fine_sites_per_coarse = fine_lat->get_nc();
    for (int i = 0; i < fine_lat->get_nd(); i++)
    {
      if (fine_lat->get_dim_mu(i) % coarse_lat->get_dim_mu(i) != 0)
      {
        std::cout << "[QMG-ERROR]: Fine lattice dimension " << i << "isn't divided evenly by coarse dimension.\n";
        return;
      }
      blocksizes[i] = fine_lat->get_dim_mu(i)/coarse_lat->get_dim_mu(i);
      fine_sites_per_coarse *= blocksizes[i];
    }

    // Prepare the one-to-many mapping between coarse sites and fine sites+dof.
    coarse_map = allocate_vector<int*>(coarse_lat->get_volume());
    build_mapping();

    // Allocate null vectors, copy them in.
    null_vectors = new complex<double>*[coarse_lat->get_nc()];
    for (int i = 0; i < coarse_lat->get_nc(); i++)
    {
      null_vectors[i] = allocate_vector<complex<double>>(fine_lat->get_size_cv());
      copy_vector(null_vectors[i], in_null_vectors[i], fine_lat->get_size_cv());
    }

    // If we're saving the Cholesky from the block ortho,
    // allocate it here.
    if (save_decomp)
    {
      block_cholesky = allocate_vector<complex<double>>(in_coarse_lat->get_size_cm());
      zero_vector(block_cholesky, in_coarse_lat->get_size_cm());
    }

    // Block orthonormalize, if requested.
    if (do_block_ortho)
    {
      block_orthonormalize();
      if (block_cholesky != 0)
      {
        std::complex<double>* backup = block_cholesky;
        block_cholesky = 0;
        block_orthonormalize();
        block_cholesky = backup;
      }
      else
      {
        block_orthonormalize();
      }
    }

    // We're good!
    is_init = true;

  }

  // Constructor for separate prolongator, restrictor. Takes in fine and
  // coarse lattice, input prolong and restrict vectors, and a flag about
  // performing block orthonormalization. Takes advantage of constructor
  // above.
  TransferMG(Lattice2D* in_fine_lat, Lattice2D* in_coarse_lat,
      complex<double>** in_prolong_null_vectors, complex<double>** in_restrict_null_vectors,
      bool do_block_bi_ortho = true, bool save_decomp = false, QMGDoublingType in_doubling = QMG_DOUBLE_NONE)
    : TransferMG(in_fine_lat, in_coarse_lat, in_prolong_null_vectors, false, false, in_doubling)
  {
    // Copy in restrict vectors. 
    restrict_null_vectors = new complex<double>*[coarse_lat->get_nc()];
    for (int i = 0; i < coarse_lat->get_nc(); i++)
    {
      restrict_null_vectors[i] = allocate_vector<complex<double>>(fine_lat->get_size_cv());
      copy_vector(restrict_null_vectors[i], in_restrict_null_vectors[i], fine_lat->get_size_cv());
    }

    // If we're saving the LU from the block bi-ortho,
    // allocate it here.
    if (save_decomp)
    {
      block_L = allocate_vector<complex<double>>(in_coarse_lat->get_size_cm());
      block_U = allocate_vector<complex<double>>(in_coarse_lat->get_size_cm());
      zero_vector(block_L, in_coarse_lat->get_size_cm());
      zero_vector(block_U, in_coarse_lat->get_size_cm());
    }

    // Block bi-orthonormalize, if requested.
    if (do_block_bi_ortho)
    {
      block_bi_orthonormalize();

      if (block_L != 0 && block_U != 0)
      {
        std::complex<double>* backup_L = block_L;
        std::complex<double>* backup_U = block_U;
        block_L = 0; block_U = 0;
        block_bi_orthonormalize();
        block_L = backup_L; block_U = backup_U;
      }
      else
      {
        block_bi_orthonormalize();
      }
    }

    // We're good!
    is_init = true;

  }

  // Destructor. Clean up!
  ~TransferMG()
  {
    if (blocksizes != 0) { delete[] blocksizes; }
    if (null_vectors != 0)
    {
      for (int i = 0; i < const_num_null_vec; i++)
      {
        if (null_vectors[i] != 0) { deallocate_vector(&null_vectors[i]); }
      }
      delete[] null_vectors;
    }
    if (restrict_null_vectors != 0)
    {
      for (int i = 0; i < const_num_null_vec; i++)
      {
        if (restrict_null_vectors[i] != 0) { deallocate_vector(&restrict_null_vectors[i]); }
      }
      delete[] restrict_null_vectors;
    }
    if (block_cholesky != 0)
    {
      deallocate_vector(&block_cholesky);
    }
    if (block_L != 0)
    {
      deallocate_vector(&block_L);
    }
    if (block_U != 0)
    {
      deallocate_vector(&block_U);
    }
    if (coarse_map != 0)
    {
      for (int i = 0; i < const_coarse_volume; i++)
      {
        if (coarse_map[i] != 0) { delete[] coarse_map[i]; }
      }
      deallocate_vector(&coarse_map);
    }
  }

  // Query if the transfer object is properly initialized.
  bool is_initialized()
  {
    return is_init;
  }

  // The public 'prolong' operator (coarse_to_fine)
  // Arg 1: coarse vector.
  // Arg 2: fine vector.
  void prolong_c2f(complex<double>* coarse_cv, complex<double>* fine_cv)
  {
    prolong_c2f(coarse_cv, fine_cv, null_vectors, coarse_lat->get_nc());
  }

  // The public 'restrict' operator (fine to coarse)
  // Arg 1: fine vector.
  // Arg 2: coarse vector.
  void restrict_f2c(complex<double>* fine_cv, complex<double>* coarse_cv)
  {
    restrict_f2c(fine_cv, coarse_cv, (restrict_null_vectors == 0) ? null_vectors : restrict_null_vectors, coarse_lat->get_nc());
  }

  // Query if the restrictor equals P^\dagger.
  bool is_symmetric()
  {
    return (restrict_null_vectors == 0);
  }

  // Query if we saved the decompositions.
  bool has_decompositions()
  {
    if (is_symmetric())
    {
      return (block_cholesky != 0);
    }
    else
    {
      return (block_L != 0 && block_U != 0);
    }
  }

  // Expose cholesky blocks.
  void copy_cholesky(complex<double>* save_cholesky)
  {
    if (block_cholesky == 0)
    {
      std::cout << "[QMG-WARNING]: In expose_cholesky, block Cholesky has not been computed.\n";
    }
    else
    {
      copy_vector(save_cholesky, block_cholesky, coarse_lat->get_size_cm());
    }
  }

  // Expose LU blocks.
  void copy_LU(complex<double>* save_L, complex<double>* save_U)
  {
    if (block_L == 0 || block_U == 0)
    {
      std::cout << "[QMG-WARNING]: In expose_LU, block LU has not been computed.\n";
    }
    else
    {
      copy_vector(save_L, block_L, coarse_lat->get_size_cm());
      copy_vector(save_U, block_U, coarse_lat->get_size_cm());
    }
  }

  QMGDoublingType get_doubling()
  {
    return doubling;
  }


private:

  // Declare merge sort for build_mapping. Smallest int first.
  void merge_sort(int* sort_me, int* temp, int size);

  // Recursive function to learn every fine site corresponding
  // to a coarse site. Recursive b/c it's dimension-independent!
  void recursive_site_build(int* list, int* coarse_coord, int* coords, int step, int& count);

  // Special C function to take the inverse
  // square root of the real part of a site. 
  // Needed for block orthonormalize.
  static inline void inv_real_sqrt(int i, complex<double>& elem, void* extra_data)
  {
    double tmp = 1.0/sqrt(real(elem));
    elem = complex<double>(tmp, 0.0);
  }

  // Special C function to take the inverse magnitude square root
  // of a site. Needed for block bi-orthonormalize.
  static inline void inv_abs_sqrt(int i, complex<double>& elem, void* extra_data)
  {
    double tmp = 1.0/abs(elem);
    elem = complex<double>(tmp, 0.0);
  }

  // Special C function to take the inverse magnitude square root,
  // preserving the phase, of a site. Needed for block bi-orthonormalize.
  static inline void inv_phase_abs_sqrt(int i, complex<double>& elem, void* extra_data)
  {
    double tmp = 1.0/sqrt(abs(elem));
    elem = polar(tmp, arg(elem));
  }
  
};

// Recursive function to learn every fine site corresponding
// to a coarse site. Recursive b/c it's dimension-independent!
void TransferMG::recursive_site_build(int* list, int* coarse_coords, int* coords, int step, int& count)
{
  if (step < fine_lat->get_nd())
  {
    // Loop over every site in block in dimension step.
    for (int i = coarse_coords[step]*blocksizes[step]; i < (coarse_coords[step]+1)*blocksizes[step]; i++)
    {
      coords[step] = i;
      recursive_site_build(list, coarse_coords, coords, step+1, count);
    }
  }
  else // we're on the lowest level. coords is fully populated.
  {
    for (int i = 0; i < fine_lat->get_nc(); i++)
    {
      list[count++] = fine_lat->cv_coord_to_index(coords, i);
    }
  }
}

// Figure out the one-to-many mapping between coarse sites (volume)
//         and fine sites (full color vector).
//         There's no reason why this couldn't be many-to-many (for overlapping 
//         domains, for ex).
void TransferMG::build_mapping()
{
  // Get total number of coarse sites.
  const int coarse_volume = coarse_lat->get_volume();

  // Temporary memory used for merge sort.
  int* tmp_mem = new int[fine_sites_per_coarse];

  // Temporary memory for recursive fine site build.
  int* coarse_coord = new int[coarse_lat->get_nd()];
  int* fine_coords = new int[coarse_lat->get_nd()];

  // Counter.
  int count; 

  // Loop over coarse sites.
  // OPENMP/ACC WOULD GO HERE.
  for (int i = 0; i < coarse_volume; i++)
  {
    // Initialize coarse map.
    coarse_map[i] = new int[fine_sites_per_coarse];

    // Get coarse coordinates.
    coarse_lat->index_to_coord(i, coarse_coord);

    // Recursively build fine site list.
    count = 0;
    recursive_site_build(coarse_map[i], coarse_coord, fine_coords, 0, count);

    // Sort the fine indices.
    merge_sort(coarse_map[i], tmp_mem, fine_sites_per_coarse);
  }

  // Clean up.
  delete[] tmp_mem;
  delete[] coarse_coord;
  delete[] fine_coords; 

}

// More general 'prolong' operator (coarse_to_fine)
// Arg 1: coarse vector.
// Arg 2: fine vector.
// Arg 3: null vector(s).
// Arg 4: number of null vectors. (there are some cases we want to prolong w/ 1.)
void TransferMG::prolong_c2f(complex<double>* coarse_cv, complex<double>* fine_cv, complex<double>** in_null_vectors, int nvec)
{
  int i,i_dof,j;
  const int coarse_volume = coarse_lat->get_volume();

  // Loop over every coarse site.
  for (i = 0; i < coarse_volume; i++)
  {
    // Loop over each null vector (generically, coarse dof)
    for (i_dof = 0; i_dof < nvec; i_dof++)
    {
      // Get the coarse color vector index.
      int cv_index = coarse_lat->vol_index_dof_to_cv_index(i, i_dof);

      // We saved each coarse site's corresponding fine degrees of freedom.
      for (j = 0; j < fine_sites_per_coarse; j++)
      {
        // Do the prolong!
        // Update the fine vector site with the i_dof's null vector site times
        // the relevant coarse index. 
        fine_cv[coarse_map[i][j]] += in_null_vectors[i_dof][coarse_map[i][j]]*coarse_cv[cv_index];
      }

    }
  }
}

// Pre-declaring a 'restrict' operator (fine_to_coarse)
// Arg 1: fine vector.
// Arg 2: coarse vector.
// Arg 3: null vector(s).
// Arg 4: number of null vectors. (there are some cases we want to restrict w/ 1.)
void TransferMG::restrict_f2c(complex<double>* fine_cv, complex<double>* coarse_cv, complex<double>** in_null_vectors, int nvec)
{
  int i,i_dof,j;
  const int coarse_volume = coarse_lat->get_volume();

  // Loop over every coarse site.
  for (i = 0; i < coarse_volume; i++)
  {
    // Loop over each null vector (generically, coarse dof)
    for (i_dof = 0; i_dof < nvec; i_dof++)
    {
      // Get the coarse color vector index.
      int cv_index = coarse_lat->vol_index_dof_to_cv_index(i, i_dof);

      // We saved each coarse site's corresponding fine degrees of freedom.
      for (j = 0; j < fine_sites_per_coarse; j++)
      {
        // Do the restrict.
        // Update the coarse vector site with the conjugate i_dof's null
        // vector times the relevant fine index.
        coarse_cv[cv_index] += conj(in_null_vectors[i_dof][coarse_map[i][j]])*fine_cv[coarse_map[i][j]];
      }
    }
  }
}

// Perform the block orthonormalization.
void TransferMG::block_orthonormalize()
{
  // James Osborn has a smart way of doing it---you can write block
  // orthonormalization in terms of prolong-restrict operations. 
  // Ref: https://github.com/usqcd-software/qopqdp/blob/master/lib/mg/mg_p.c
  // Line 240, function QOPP(mgOrtho)(QDPN(ColorVector) *cv[], int nv, QOP_MgBlock *mgb)

  // Iterators.
  int i,j;

  // Values.
  const int fine_size_cv = fine_lat->get_size_cv();
  const int coarse_size_cv = coarse_lat->get_size_cv();
  const int coarse_vol = coarse_lat->get_volume();
  const int coarse_dof = coarse_lat->get_nc();

  // Temporary vectors. 

  // Fine...
  complex<double>* fine_cv_1 = allocate_vector<complex<double>>(fine_size_cv);
  zero_vector(fine_cv_1, fine_size_cv);

  // ...and coarse.
  complex<double>* coarse_cv_2 = allocate_vector<complex<double>>(coarse_size_cv);
  zero_vector(coarse_cv_2, coarse_size_cv);

  // Loop over all null vectors.
  for (i = 0; i < coarse_dof; i++)
  {
    // Block Orthogonalize null vector i against previous null vectors.
    for (j = 0; j < i; j++)
    {
      // Zero out vectors.
      zero_vector(fine_cv_1, fine_size_cv);
      zero_vector(coarse_cv_2, coarse_size_cv);

      // Restrict null vector i with (already ortho'd null vector) j.
      // This is the <\vec{i},\vec{j}> dot product in Gram-Schmidt.
      restrict_f2c(null_vectors[i], coarse_cv_2, &null_vectors[j], 1);

      // If we're saving a Cholesky decomp, copy it in here.
      if (block_cholesky != 0)
      {
        // Even if we only restrict with one vector, the restrict function
        // knows the real Nc: we always grab from the first color component
        // of coarse_cv_2.
        copy_vector_blas(block_cholesky + j*coarse_dof + i, coarse_dof*coarse_dof, coarse_cv_2, coarse_dof, coarse_vol);
      }

      // Prolong again with null vector i.
      // This forms <\vec{i},\vec{j}> \vec{j}.
      prolong_c2f(coarse_cv_2, fine_cv_1, &null_vectors[j], 1);

      // Subtract off.
      // This forms \vec{i} -= <\vec{i},\vec{j}> \vec{j}.
      caxpy(-1.0, fine_cv_1, null_vectors[i], fine_size_cv);
    }

    // Block normalize null vector i.
    // zero out vectors again.
    zero_vector(fine_cv_1, fine_size_cv);
    zero_vector(coarse_cv_2, coarse_size_cv);

    // Restrict null vector i with null vector i.
    // This is a block <\vec{i}, \vec{i}>.
    restrict_f2c(null_vectors[i], coarse_cv_2, &null_vectors[i], 1);

    // Prepare to normalize. This requires taking the inverse sqrt of
    // <\vec{i}, \vec{i}>. The '0' is because it doesn't need any info passed in.
    arb_local_function_vector(coarse_cv_2, inv_real_sqrt, 0, coarse_size_cv);

    // If we're saving a Cholesky decomp, copy it in here.
    if (block_cholesky != 0)
    {
      cinvx(coarse_cv_2, coarse_size_cv);
      // Even if we only restrict with one vector, the restrict function
      // knows the real Nc: we always grab from the first color component
      // of coarse_cv_2.
      copy_vector_blas(block_cholesky + i*(coarse_dof+1), coarse_dof*coarse_dof, coarse_cv_2, coarse_dof, coarse_vol);
      cinvx(coarse_cv_2, coarse_size_cv);
    }

    // Prolong the inverse norm with null vector i. 
    // This is block \vec{i}/sqrt(<\vec{i}, \vec{i}>).
    prolong_c2f(coarse_cv_2, fine_cv_1, &null_vectors[i], 1);

    // Set this back to null_vectors[i].
    copy_vector(null_vectors[i], fine_cv_1, fine_size_cv);
  }

  // Clean up.
  deallocate_vector(&coarse_cv_2);
  deallocate_vector(&fine_cv_1);
}

// Perform the block bi-orthonormalization.
void TransferMG::block_bi_orthonormalize()
{
  // Based on the smart block_orthonormalize.

  // Iterators.
  int i,j;

  // Values.
  const int fine_size_cv = fine_lat->get_size_cv();
  const int coarse_size_cv = coarse_lat->get_size_cv();
  const int coarse_vol = coarse_lat->get_volume();
  const int coarse_dof = coarse_lat->get_nc();

  // Temporary vectors. 

  // Fine...
  complex<double>* fine_cv_1 = allocate_vector<complex<double>>(fine_size_cv);
  zero_vector(fine_cv_1, fine_size_cv);

  // ...and coarse.
  complex<double>* coarse_cv_2 = allocate_vector<complex<double>>(coarse_size_cv);
  zero_vector(coarse_cv_2, coarse_size_cv);

  // Loop over all null vectors.
  for (i = 0; i < coarse_dof; i++)
  {
    // Block Orthogonalize null vector i against previous null vectors.
    for (j = 0; j < i; j++)
    {
      // Zero out vectors.
      zero_vector(fine_cv_1, fine_size_cv);
      zero_vector(coarse_cv_2, coarse_size_cv);

      // Take care of the prolong.

      // Restrict null vector i with (already ortho'd null vector) j.
      // This is the <\restrict{i},\prolong{j}> dot product.
      restrict_f2c(null_vectors[i], coarse_cv_2, &restrict_null_vectors[j], 1);

      // If we're saving an LU decomp, copy U in here.
      if (block_U != 0)
      {
        // Even if we only restrict with one vector, the restrict function
        // knows the real Nc: we always grab from the first color component
        // of coarse_cv_2.
        copy_vector_blas(block_U + j*coarse_dof + i, coarse_dof*coarse_dof, coarse_cv_2, coarse_dof, coarse_vol);
      }

      // Prolong again with null vector i.
      // This forms <\restrict{i},\prolong{j}> \prolong{j}.
      prolong_c2f(coarse_cv_2, fine_cv_1, &null_vectors[j], 1);

      // Subtract off.
      // This forms \prolong{i} -= <\restrict{i},\prolong{j}> \prolong{j}.
      caxpy(-1.0, fine_cv_1, null_vectors[i], fine_size_cv);

      // Take care of the restrict.
      zero_vector(fine_cv_1, fine_size_cv);
      zero_vector(coarse_cv_2, coarse_size_cv);

      // Restrict restricting null vector i with (already ortho'd null vector) j.
      // This is the <\prolong{i},\restrict{j}> dot product.
      restrict_f2c(restrict_null_vectors[i], coarse_cv_2, &null_vectors[j], 1);

      // If we're saving an LU decomp, copy L in here.
      if (block_L != 0)
      {
        // Even if we only restrict with one vector, the restrict function
        // knows the real Nc: we always grab from the first color component
        // of coarse_cv_2.
        // Remark: We really should be copying in the complex conjugate.
        // We do that in one swoop at the end.
        copy_vector_blas(block_L + i*coarse_dof + j, coarse_dof*coarse_dof, coarse_cv_2, coarse_dof, coarse_vol);
      }

      // Prolong again with null vector i.
      // This forms <\prolong{i},\restrict{j}> \restrict{j}.
      prolong_c2f(coarse_cv_2, fine_cv_1, &restrict_null_vectors[j], 1);

      // Subtract off.
      // This forms \restrict{i} -= <\prolong{i},\restrict{j}> \restrict{j}
      caxpy(-1.0, fine_cv_1, restrict_null_vectors[i], fine_size_cv);
    }

    // Normalize restrict. Preserve phase.

    // Block normalize null vector i.
    // zero out vectors again.
    zero_vector(fine_cv_1, fine_size_cv);
    zero_vector(coarse_cv_2, coarse_size_cv);

    // Restrict prolongator i with restrictor i.
    // This is a block <\restrict{i}, \prolng{i}>.
    restrict_f2c(null_vectors[i], coarse_cv_2, &restrict_null_vectors[i], 1);

    // Prepare to normalize. This requires taking the inverse abs sqrt of
    // <\restrict{i}, \prolong{i}>. The '0' is because it doesn't need any info passed in.
    arb_local_function_vector(coarse_cv_2, inv_phase_abs_sqrt, 0, coarse_size_cv);

    // If we're saving an LU, copy L in here.
    if (block_L != 0)
    {
      cinvx(coarse_cv_2, coarse_size_cv);
      // Even if we only restrict with one vector, the restrict function
      // knows the real Nc: we always grab from the first color component
      // of coarse_cv_2.

      // It's sort of goofy that this works. The above function takes the
      // inverse square root of the mag, but preserves the phase.
      // This function inverts that, which flips the phase...
      // and then we do an overall conj of L below, which flips the phase
      // back to what we want. I lol'd.
      copy_vector_blas(block_L + i*(coarse_dof+1), coarse_dof*coarse_dof, coarse_cv_2, coarse_dof, coarse_vol);
      cinvx(coarse_cv_2, coarse_size_cv);
    }

    // Prolong the inverse norm with null vector i. 
    // This is block phases \restrict{i}/(~sqrt(<\restrict{i}, \prolong{i}>)).
    prolong_c2f(coarse_cv_2, fine_cv_1, &restrict_null_vectors[i], 1);

    // Set this back to null_vectors[i].
    copy_vector(restrict_null_vectors[i], fine_cv_1, fine_size_cv);

    // Normalize prolong. Only requires abs-ing the norm.
    zero_vector(fine_cv_1, fine_size_cv);

    abs_vector(coarse_cv_2, coarse_size_cv);

    // If we're saving an LU, copy U in here.
    if (block_U != 0)
    {
      cinvx(coarse_cv_2, coarse_size_cv);
      // Even if we only restrict with one vector, the restrict function
      // knows the real Nc: we always grab from the first color component
      // of coarse_cv_2.
      copy_vector_blas(block_U + i*(coarse_dof+1), coarse_dof*coarse_dof, coarse_cv_2, coarse_dof, coarse_vol);
      cinvx(coarse_cv_2, coarse_size_cv);
    }

    // Prolong the inverse norm with prolong null vector i. 
    // This is block \prolong{i}/sqrt(abs(<\restrict{i}, \prolong{i}>)).
    prolong_c2f(coarse_cv_2, fine_cv_1, &null_vectors[i], 1);

    // Set this back to null_vectors[i].
    copy_vector(null_vectors[i], fine_cv_1, fine_size_cv);

    
  }

  // If we're doing an LU, conjugate L here.
  // This is really where all the magic happens.
  if (block_L != 0)
  {
    conj_vector(block_L, coarse_lat->get_size_cm());
  }

  // Clean up.
  deallocate_vector(&coarse_cv_2);
  deallocate_vector(&fine_cv_1);
}

////////////////////////
// AUXILARY FUNCTIONS //
////////////////////////

// Declare merge sort for build_mapping. Smallest int first.
void TransferMG::merge_sort(int* sort_me, int* temp, int size)
{
  if (size == 1) // trivial
  {
    return;
  }
  if (size == 2) // trivial
  {
    temp[0] = sort_me[0];
    if (sort_me[1] < sort_me[0])
    {
      sort_me[0] = sort_me[1];
      sort_me[1] = temp[0];
    }
    return;
  }

  // recurse.
  merge_sort(sort_me, temp, size/2);
  merge_sort(sort_me + size/2, temp + size/2, size - size/2);

  // merge
  int curr1 = 0;
  int curr2 = size/2;
  int currtmp = 0;

  while (curr1 < size/2 && curr2 < size)
  {
    if (sort_me[curr1] < sort_me[curr2])
      temp[currtmp++] = sort_me[curr1++];
    else
      temp[currtmp++] = sort_me[curr2++];
  }

  while (curr1 < size/2)
    temp[currtmp++] = sort_me[curr1++];

  while (curr2 < size)
    temp[currtmp++] = sort_me[curr2++];

  for (int i = 0; i < size; i++)
    sort_me[i] = temp[i];
}

#endif // QMG_TRANSFER_OBJECT
