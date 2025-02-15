// Copyright (c) 2017 Evan S Weinberg
// A bit of an exploration place for coding up a good
// prolong/restrict interface.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"

// QMG
#include "lattice/lattice.h"

// Merge sort wooo, smallest element first.
void merge_sort(int* sort_me, int* temp, int size)
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

// Pre-declaring a 'prolong' operator (coarse_to_fine)
// Arg 1: coarse vector.
// Arg 2: fine vector.
// Arg 3: null vector(s).
// Arg 4: number of null vectors. (there are some cases we want to prolong w/ 1.)
// Arg 5: One-to-many mapping between coarse site and fine site+dof.
// Arg 6: coarse lattice object.
// Arg 7: fine lattice object. 
void coarse_to_fine_cv(complex<double>* coarse_cv, complex<double>* fine_cv, complex<double>** null_vectors, int nvec, int** coarse_map, Lattice2D* coarse_lat, Lattice2D* fine_lat);

// Pre-declaring a 'restrict' operator (fine_to_coarse)
// Arg 1: fine vector.
// Arg 2: coarse vector.
// Arg 3: null vector(s).
// Arg 4: number of null vectors. (there are some cases we want to restrict w/ 1.)
// Arg 5: One-to-many mapping between coarse site and fine site+dof.
// Arg 6: fine lattice object.
// Arg 7: coarse lattice object.
void fine_to_coarse_cv(complex<double>* fine_cv, complex<double>* coarse_cv, complex<double>** null_vectors, int nvec, int** coarse_map, Lattice2D* fine_lat, Lattice2D* coarse_lat);



// Special C function to take the inverse
// square root of the real part of a site. 
inline void inv_real_sqrt(int i, complex<double>& elem, void* extra_data);

int main(int argc, char** argv)
{
  // Set output to not be that long.
  cout << setiosflags(ios::fixed) << setprecision(6);

  //Iterators and such.
  int i,j;
  int x_coarse,y_coarse,dof_coarse,x_fine,y_fine,dof_fine;
  int count; 

  // Basic information for fine level.
  const int x_len = 64;
  const int y_len = 64;
  const int dof = 1;

  // Blocking size.
  const int x_block = 4;
  const int y_block = 4;

  // Number of null vectors (equiv. degrees of freedom on coarse level)
  const int coarse_dof = 2;

  // Create a lattice object for the fine lattice.
  Lattice2D* lat = new Lattice2D(x_len, y_len, dof);

  // Create a lattice object for the coarse lattice.
  Lattice2D* coarse_lat = new Lattice2D(x_len/x_block, y_len/y_block, coarse_dof);

  // Step 1: Figure out the one-to-many mapping between coarse sites (volume)
  //         and fine sites (full color vector).
  //         There's no reason why this couldn't be many-to-many (for overlapping 
  //         domains, for ex).

  // Get total number of coarse sites.
  const int coarse_volume = coarse_lat->get_volume();

  // Allocate a coarse volume array of pointers to lists of fine sites...
  int** coarse_map = allocate_vector<int*>(coarse_volume);

  // Compute number of fine sites per coarse site.
  const int fine_sites_per_coarse = x_block*y_block*dof;

  // temporary memory needed for merge sorts.
  int* tmp_mem = new int[fine_sites_per_coarse];

  // I need some type of blas-like function for this...
  for (i = 0; i < coarse_volume; i++)
  {
    // We need to allocate each element in coarse_map as an array of length
    // 'fine_sites_per_coarse'.
    coarse_map[i] = new int[fine_sites_per_coarse];

    // Actually do the loop and find out which fine sites correspond to which
    // coarse sites.

    // Get coarse coordinates.
    coarse_lat->index_to_coord(i, x_coarse, y_coarse);

    // Loop over corresponding fine coordinates.
    // Thought: there must be some recursive way to do this for general
    // dimensions. 
    count = 0;
    for (y_fine = y_coarse*y_block; y_fine < (y_coarse+1)*y_block; y_fine++)
    {
      for (x_fine = x_coarse*x_block; x_fine < (x_coarse+1)*x_block; x_fine++)
      {
        for (dof_fine = 0; dof_fine < dof; dof_fine++)
        {
          coarse_map[i][count++] = lat->cv_coord_to_index(x_fine, y_fine, dof_fine);
        }
      }
    }

    // Sort the fine indices for efficiency. 
    merge_sort(coarse_map[i], tmp_mem, fine_sites_per_coarse);

  }
  // Clean up temporary memory.
  delete[] tmp_mem;

  // Good! Now we have the mapping between the coarse site at memory index
  // 'i' and the corresponding memory indices for the fine sites. We need this
  // to do a prolong and restrict.

  // Step 2: Create null vectors. For now, we'll just have two: one for
  //         even sites only, the other for all sites (intentionally 
  //         non-orthogonal).

  // First off, get length of colorvector. 
  const int fine_size_cv = lat->get_size_cv();

  // Allocate 2 null vectors.
  const int num_nvec = coarse_dof; // different names make sense in different contexts.
  complex<double>** null_vectors = new complex<double>*[num_nvec];
  null_vectors[0] = allocate_vector<complex<double>>(fine_size_cv);
  null_vectors[1] = allocate_vector<complex<double>>(fine_size_cv);

  // Set first vector to even only.
  constant_vector(null_vectors[0], 1.0, fine_size_cv/2);
  zero_vector(null_vectors[0] + fine_size_cv/2, fine_size_cv/2);

  // Set second vector to ones everywhere.
  constant_vector(null_vectors[1], 1.0, fine_size_cv);

  // Step 3: Prolong. We'll put something on a coarse site, and prolong it
  //         to the fine lattice.

  // We'll create a coarse vector and a fine vector.
  const int coarse_size_cv = coarse_lat->get_size_cv();
  complex<double>* coarse_cv_1 = allocate_vector<complex<double>>(coarse_size_cv);
  zero_vector(coarse_cv_1, coarse_size_cv);

  complex<double>* fine_cv_1 = allocate_vector<complex<double>>(fine_size_cv);
  zero_vector(fine_cv_1, fine_size_cv);

  // Put two points on the coarse vector.
  coarse_cv_1[coarse_lat->cv_coord_to_index(coarse_lat->get_dim_mu(0)/2, coarse_lat->get_dim_mu(1)/2, 0)] = 1.0;
  coarse_cv_1[coarse_lat->cv_coord_to_index(1, coarse_lat->get_dim_mu(1)/2, 1)] = 1.0;

  // Look for non-zero elements of the vector.
  cout << "Printing original coarse vector.\n";
  for (i = 0; i < coarse_size_cv; i++)
  {
    if (abs(coarse_cv_1[i]) > 1e-12)
    {
      coarse_lat->cv_index_to_coord(i, x_coarse, y_coarse, dof_coarse);
      cout << "Original coarse vector: Non-zero at (" << x_coarse << "," << y_coarse << "," << dof_coarse << ") = " << coarse_cv_1[i] << "\n";
    }
  }

  // Actually do prolong!
  // We're putting this in a function because we re-use it for block-
  // orthonormalization (but we discuss why later).
  coarse_to_fine_cv(coarse_cv_1, fine_cv_1, null_vectors, num_nvec, coarse_map, coarse_lat, lat);

  // Step 4: Restrict. We'll restrict the result from the fine lattice back
  //         down to the coarse lattice. This should match the original
  //         result, up to a normalization!

  // Create a coarse vector to restrict into.
  complex<double>* coarse_cv_2 = allocate_vector<complex<double>>(coarse_size_cv);
  zero_vector(coarse_cv_2, coarse_size_cv);

  // We're putting this in a function because we re-use it for block-
  // orthonormalization (but we discuss why later).
  fine_to_coarse_cv(fine_cv_1, coarse_cv_2, null_vectors, num_nvec, coarse_map, lat, coarse_lat);
  

  // Look for non-zero elements of the restricted vector.
  cout << "Printing prolong-restrict coarse vector: no block ortho.\n";
  for (i = 0; i < coarse_size_cv; i++)
  {
    if (abs(coarse_cv_2[i]) > 1e-12)
    {
      coarse_lat->cv_index_to_coord(i, x_coarse, y_coarse, dof_coarse);
      cout << "Prolong-Restrict Test 1: Non-zero at (" << x_coarse << "," << y_coarse << "," << dof_coarse << ") = " << coarse_cv_2[i] << "\n";
    }
  }

  // Step 5: Block orthonormalization. We need to block orthonormalize the
  //         vectors. As a test, we'll prolong and restrict again, and this
  //         time we should match the original coarse value.

  // James Osborn has a smart way of doing it---you can write block
  // orthonormalization in terms of prolong-restrict operations. 
  // Ref: https://github.com/usqcd-software/qopqdp/blob/master/lib/mg/mg_p.c
  // Line 240, function QOPP(mgOrtho)(QDPN(ColorVector) *cv[], int nv, QOP_MgBlock *mgb)

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
      fine_to_coarse_cv(null_vectors[i], coarse_cv_2, &null_vectors[j], 1, coarse_map, lat, coarse_lat);

      // Prolong again with null vector i.
      // This forms <\vec{i},\vec{j}> \vec{j}.
      coarse_to_fine_cv(coarse_cv_2, fine_cv_1, &null_vectors[j], 1, coarse_map, coarse_lat, lat);

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
    fine_to_coarse_cv(null_vectors[i], coarse_cv_2, &null_vectors[i], 1, coarse_map, lat, coarse_lat);

    // Prepare to normalize. This requires taking the inverse sqrt of
    // <\vec{i}, \vec{i}>. The '0' is because it doesn't need any info passed in.
    arb_local_function_vector(coarse_cv_2, inv_real_sqrt, 0, coarse_size_cv);

    // Prolong the inverse norm with null vector i. 
    // This is block \vec{i}/sqrt(<\vec{i}, \vec{i}>).
    coarse_to_fine_cv(coarse_cv_2, fine_cv_1, &null_vectors[i], 1, coarse_map, coarse_lat, lat);

    // Set this back to null_vectors[i].
    copy_vector(null_vectors[i], fine_cv_1, fine_size_cv);
  }

  // Test a prolong-restrict again.
  // Now, we should exactly reproduce the original vector. 

  zero_vector(fine_cv_1, fine_size_cv);
  zero_vector(coarse_cv_2, coarse_size_cv);
  coarse_to_fine_cv(coarse_cv_1, fine_cv_1, null_vectors, num_nvec, coarse_map, coarse_lat, lat);
  fine_to_coarse_cv(fine_cv_1, coarse_cv_2, null_vectors, num_nvec, coarse_map, lat, coarse_lat);

  cout << "Printing prolong-restrict coarse vector: yes block ortho.\n";
  for (i = 0; i < coarse_size_cv; i++)
  {
    if (abs(coarse_cv_2[i]) > 1e-12)
    {
      coarse_lat->cv_index_to_coord(i, x_coarse, y_coarse, dof_coarse);
      cout << "Prolong-Restrict Test 2: Non-zero at (" << x_coarse << "," << y_coarse << "," << dof_coarse << ") = " << coarse_cv_2[i] << "\n";
    }
  }

  // Last step: Clean up. 
  deallocate_vector(&coarse_cv_1);
  deallocate_vector(&coarse_cv_2);
  deallocate_vector(&fine_cv_1);
  for (i = 0; i < num_nvec; i++)
  {
    deallocate_vector(&null_vectors[i]);
  }
  delete[] null_vectors; 
  for (i = 0; i < coarse_volume; i++)
  {
    delete[] coarse_map[i];
  }
  deallocate_vector(&coarse_map);
  delete lat;
  delete coarse_lat;

  return 0;
}

// Pre-declaring a 'prolong' operator (coarse_to_fine)
// Arg 1: coarse vector.
// Arg 2: fine vector.
// Arg 3: null vector(s).
// Arg 4: number of null vectors. (there are some cases we want to prolong w/ 1.)
// Arg 5: One-to-many mapping between coarse site and fine site+dof.
// Arg 6: coarse lattice object.
// Arg 7: fine lattice object. 
void coarse_to_fine_cv(complex<double>* coarse_cv, complex<double>* fine_cv, complex<double>** null_vectors, int nvec, int** coarse_map, Lattice2D* coarse_lat, Lattice2D* fine_lat)
{
  int i,i_dof,j;
  const int coarse_volume = coarse_lat->get_volume();
  int fine_sites_per_coarse = fine_lat->get_nc();
  for (i = 0; i < fine_lat->get_nd(); i++)
  {
    fine_sites_per_coarse *= (fine_lat->get_dim_mu(i)/coarse_lat->get_dim_mu(i));
  }

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
        fine_cv[coarse_map[i][j]] += null_vectors[i_dof][coarse_map[i][j]]*coarse_cv[cv_index];
      }

    }
  }
}

// Pre-declaring a 'restrict' operator (fine_to_coarse)
// Arg 1: fine vector.
// Arg 2: coarse vector.
// Arg 3: null vector(s).
// Arg 4: number of null vectors. (there are some cases we want to restrict w/ 1.)
// Arg 5: One-to-many mapping between coarse site and fine site+dof.
// Arg 6: fine lattice object.
// Arg 7: coarse lattice object.
void fine_to_coarse_cv(complex<double>* fine_cv, complex<double>* coarse_cv, complex<double>** null_vectors, int nvec, int** coarse_map, Lattice2D* fine_lat, Lattice2D* coarse_lat)
{
  int i,i_dof,j;
  const int coarse_volume = coarse_lat->get_volume();
  int fine_sites_per_coarse = fine_lat->get_nc();
  for (i = 0; i < fine_lat->get_nd(); i++)
  {
    fine_sites_per_coarse *= (fine_lat->get_dim_mu(i)/coarse_lat->get_dim_mu(i));
  }

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
        coarse_cv[cv_index] += conj(null_vectors[i_dof][coarse_map[i][j]])*fine_cv[coarse_map[i][j]];
      }
    }
  }
}

// Special C function to take the inverse
// square root of the real part of a site. 
inline void inv_real_sqrt(int i, complex<double>& elem, void* extra_data)
{
  double tmp = 1.0/sqrt(real(elem));
  elem = complex<double>(tmp, 0.0);
}