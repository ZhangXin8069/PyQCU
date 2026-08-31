#pragma once

/**
   @file quda_define.h
   @brief Macros defined set by the cmake build system.  This file
   should not be edited manually.
 */

/**
 * @def   __COMPUTE_CAPABILITY__
 * @brief This macro sets the target GPU architecture, which is
 * defined on both host and device.
 */
#define __COMPUTE_CAPABILITY__ 700

/**
 * @def   MAX_MULTI_BLAS_N
 * @brief This macro sets the limit of blas fusion in the multi-blas
 * and multi-reduce kernels
 */
#define MAX_MULTI_BLAS_N 12

/**
 * @def   MAX_MULTI_BLAS_N
 * @brief This macro sets the limit of blas fusion in the multi-blas
 * and multi-reduce kernels
 */
#define MAX_MULTI_RHS 16

/**
 * @def   MAX_KERNEL_ARG_SIZE
 * @brief This macro sets the maximum static size of the kernel arguments
 * in bytes passed to a kernel on the target architecture
 */
#define MAX_KERNEL_ARG_SIZE 4096

/**
 * @def   MAX_MULTI_BLAS_N
 * @brief This macro sets the register tile size for MRHS kernels
 */
#define MAX_MULTI_RHS_TILE 1

/**
 * @def   QUDA_ALTERNATIVE_I_TO_F
 * @brief This sets the percentage of I2F conversions using the alternative path
 */
#define QUDA_ALTERNATIVE_I_TO_F 0

#define QUDA_HETEROGENEOUS_ATOMIC
#ifdef QUDA_HETEROGENEOUS_ATOMIC
/**
 * @def   HETEROGENEOUS_ATOMIC
 * @brief This macro sets whether we are compiling QUDA with heterogeneous atomic
 * support enabled or not
 */
#define HETEROGENEOUS_ATOMIC
#undef QUDA_HETEROGENEOUS_ATOMIC
#endif

#define QUDA_HETEROGENEOUS_ATOMIC_INF_INIT
#ifdef QUDA_HETEROGENEOUS_ATOMIC_INF_INIT
/**
 * @def   HETEROGENEOUS_ATOMIC_INF_INIT
 * @brief This macro sets whether we are using infinity for the signaling sentinel
 */
#define HETEROGENEOUS_ATOMIC_INF_INIT
#undef QUDA_HETEROGENEOUS_ATOMIC_INF_INIT
#endif

/* #undef QUDA_SHARED_MEMORY_SPILL */

#define QUDA_LARGE_KERNEL_ARG

/* #undef QUDA_DIRAC_CLOVER_HASENBUSCH */
#ifdef QUDA_DIRAC_CLOVER_HASENBUSCH
/**
 * @def   GPU_CLOVER_HASENBUSCH_TWIST
 * @brief This macro is set when we have clover Hasenbusch fermions enabled
 */
#define GPU_CLOVER_HASENBUSCH_TWIST
#endif

/* #undef QUDA_DIRAC_TWISTED_CLOVER */
#if defined(QUDA_DIRAC_TWISTED_CLOVER) || defined(QUDA_DIRAC_CLOVER_HASENBUSCH)
/**
 * @def   GPU_TWISTED_CLOVER_DIRAC
 * @brief This macro is set when we have TMC fermions enabled
 */
#define GPU_TWISTED_CLOVER_DIRAC
#endif

#define QUDA_DIRAC_CLOVER
#if defined(QUDA_DIRAC_CLOVER) || defined(QUDA_DIRAC_TWISTED_CLOVER) || defined(QUDA_DIRAC_CLOVER_HASENBUSCH)
/**
 * @def   GPU_CLOVER_DIRAC
 * @brief This macro is set when we have clover fermions enabled
 */
#define GPU_CLOVER_DIRAC
#endif

/* #undef QUDA_DIRAC_TWISTED_MASS */
#if defined(QUDA_DIRAC_TWISTED_MASS) || defined(QUDA_DIRAC_TWISTED_CLOVER)
/**
 * @def   GPU_TWISTED_MASS_DIRAC
 * @brief This macro is set when we have TM fermions enabled
 */
#define GPU_TWISTED_MASS_DIRAC
#endif

#define QUDA_DIRAC_WILSON
#if defined(QUDA_DIRAC_WILSON) || defined(QUDA_DIRAC_CLOVER) || defined(QUDA_DIRAC_TWISTED_MASS)
/**
 * @def   GPU_WILSON_DIRAC
 * @brief This macro is set when we kave Wilson fermions enabled
 */
#define GPU_WILSON_DIRAC
#endif

/* #undef QUDA_DIRAC_DOMAIN_WALL */
#ifdef QUDA_DIRAC_DOMAIN_WALL
/**
 * @def   GPU_DOMAIN_WALL_DIRAC
 * @brief This macro is set when we have DWF fermions enabled
 */
#define GPU_DOMAIN_WALL_DIRAC
#endif

/* #undef QUDA_DIRAC_STAGGERED */
#ifdef QUDA_DIRAC_STAGGERED
/**
 * @def   GPU_STAGGERED_DIRAC
 * @brief This macro is set when we have staggered fermions enabled
 */
#define GPU_STAGGERED_DIRAC
#endif

/* #undef QUDA_DIRAC_LAPLACE */
#ifdef QUDA_DIRAC_LAPLACE
/**
 * @def   GPU_LAPLACE
 * @brief This macro is set when we have the Laplace operator enabled
 */
#define GPU_LAPLACE
#endif

/* #undef QUDA_DIRAC_COVDEV */
#ifdef QUDA_DIRAC_COVDEV
/**
 * @def   GPU_COVDEV
 * @brief This macro is set when we have the covariant derivative enabled
 */
#define GPU_COVDEV
#endif

/**
 * @def   QUDA_DOMAIN_DECOMPOSITION
 * @brief This macro sets the type of Domain Decomposition (DD)-aware Dirac operator enabled
 */
#define QUDA_DOMAIN_DECOMPOSITION 0

/* #undef QUDA_DIRAC_DISTANCE_PRECONDITIONING */
#ifdef QUDA_DIRAC_DISTANCE_PRECONDITIONING
/**
 * @def GPU_DISTANCE_PRECONDITIONING
 * @brief This macro is set when we have distance preconditioned
 * Wilson/clover dslash enabled
 */
#define GPU_DISTANCE_PRECONDITIONING
#endif

/**
 * @def QUDA_DSLASH_DOUBLE_STORE
 * @brief This macro sets whether to use double storage of the gauge
 * field to simplify indexing in the Dslash kernels.
 */
/* #undef QUDA_DSLASH_DOUBLE_STORE */

/**
 * @def QUDA_DSLASH_PREFETCH_TYPE
 * @brief This macro sets whether to use
 * the TMA for L2 prefetching:
 * NONE - no prefetch
 * THREAD - per thread prefetch
 * BULK - TMA bulk prefetch
 * TENSOR - TMA tensor descriptor prefetch
 */
#define QUDA_DSLASH_PREFETCH_TYPE_NONE

/**
 * @def QUDA_DSLASH_PREFETCH_DISTANCE_WILSON
 * @brief This macro sets the prefetch distance for Wilson fermions
 */
#define QUDA_DSLASH_PREFETCH_DISTANCE_WILSON 0

/**
 * @def QUDA_DSLASH_PREFETCH_DISTANCE_STAGGERED
 * @brief This macro sets the prefetch distance for staggered fermions
 */
#define QUDA_DSLASH_PREFETCH_DISTANCE_STAGGERED 0

#define QUDA_MULTIGRID
#ifdef QUDA_MULTIGRID
/**
 * @def   GPU_MULTIGRID
 * @brief This macro is set when we have multigrid enabled
 */
#define GPU_MULTIGRID
#endif

/**
 * @def   QUDA_MULTIGRID
 * @brief This macro is set when we have MMA enabled for the CUDA targets
 */
#define QUDA_ENABLE_MMA

#ifdef QUDA_MULTIGRID

/**
 * @def   QUDA_MULTIGRID_SETUP_*
 * @brief This macro is used to set the MMA type used for multigrid setup
 */
#define QUDA_MULTIGRID_MMA_SETUP_HALF 0
#define QUDA_MULTIGRID_MMA_SETUP_SINGLE 0

/**
 * @def   QUDA_MULTIGRID_MMA_DSLASH_*
 * @brief This macro is used to set the MMA type used for coarse dslash
 */
#define QUDA_MULTIGRID_MMA_DSLASH_HALF 0
#define QUDA_MULTIGRID_MMA_DSLASH_SINGLE 0

/**
 * @def   QUDA_MULTIGRID_MMA_PROLONGATOR_*
 * @brief This macro is used to set the MMA type used for prolongator
 */
#define QUDA_MULTIGRID_MMA_PROLONGATOR_HALF 0
#define QUDA_MULTIGRID_MMA_PROLONGATOR_SINGLE 0

/**
 * @def   QUDA_MULTIGRID_MMA_RESTRICTOR_*
 * @brief This macro is used to set the MMA type used for restrictor
 */
#define QUDA_MULTIGRID_MMA_RESTRICTOR_HALF 0
#define QUDA_MULTIGRID_MMA_RESTRICTOR_SINGLE 0

#endif

#define QUDA_CLOVER_DYNAMIC
#ifdef QUDA_CLOVER_DYNAMIC
/**
 * @def   DYNAMIC_CLOVER
 * @brief This macro sets whether we are compiling QUDA with dynamic
 * clover inversion support enabled or not
 */
#define DYNAMIC_CLOVER
#undef QUDA_CLOVER_DYNAMIC
#endif

#define QUDA_CLOVER_RECONSTRUCT
#ifdef QUDA_CLOVER_RECONSTRUCT
/**
 * @def   RECONSTRUCT_CLOVER
 * @brief This macro sets whether we are compiling QUDA with
 * compressed clover storage or not
 */
#define RECONSTRUCT_CLOVER
#undef QUDA_CLOVER_RECONSTRUCT
#endif

#define QUDA_CLOVER_CHOLESKY_PROMOTE
#ifdef QUDA_CLOVER_CHOLESKY_PROMOTE
/**
 * @def   CLOVER_PROMOTE_CHOLESKY
 * @brief This macro sets whether we promote the internal precision of
 * Cholesky decomposition used to invert the clover term
 */
#define CLOVER_PROMOTE_CHOLESKY
#undef QUDA_CLOVER_CHOLESKY_PROMOTE
#endif

/* #undef QUDA_MULTIGRID_DSLASH_PROMOTE */
#ifdef QUDA_MULTIGRID_DSLASH_PROMOTE
/**
 * @def   MULTIGRID_DSLASH_PROMOTE
 * @brief This macro sets whether we promote the internal precision of
 * the coarse dslash used in multigrid.  This enables reproducibility
 * regardless of the thread granularity chosen
 */
#define MULTIGRID_DSLASH_PROMOTE
#undef QUDA_CLOVER_CHOLESKY_PROMOTE
#endif

/**
 * @def QUDA_ORDER_DOUBLE
 * @brief This macro sets the data ordering for double precision fields
 */
#define QUDA_ORDER_DOUBLE 2

/**
 * @def QUDA_ORDER_SINGLE
 * @brief This macro sets the data ordering for single precision fields
 */
#define QUDA_ORDER_SINGLE 4

/**
 * @def QUDA_ORDER_HALF
 * @brief This macro sets the data ordering for half precision fields
 */
#define QUDA_ORDER_HALF 8

/**
 * @def QUDA_ORDER_QUARTER
 * @brief This macro sets the data ordering for quarter precision fields
 */
#define QUDA_ORDER_QUARTER 8

/**
 * @def QUDA_VECTORIZE_SINGLE
 * @brief Whether to employ vectorized instruction for single precision (where supported)
 */
/* #undef QUDA_VECTORIZE_SINGLE */

/**
 * @def QUDA_BLAS_PREFETCH_TYPE
 * @brief BLAS prefetch mode set by CMake (\c QUDA_BLAS_PREFETCH_TYPE): expands to one of
 * \c QUDA_BLAS_PREFETCH_TYPE_NONE, \c QUDA_BLAS_PREFETCH_TYPE_THREAD, \c QUDA_BLAS_PREFETCH_TYPE_BULK.
 */
#define QUDA_BLAS_PREFETCH_TYPE_THREAD

/**
 * @def QUDA_BUILD_NATIVE_FFT
 * @brief This macro is set by CMake if the native FFT library is used
 */
/* #undef QUDA_BUILD_NATIVE_FFT */

/**
 * @def QUDA_TARGET_CUDA
 * @brief This macro is set by CMake if the CUDA Build Target is selected
 */
#define QUDA_TARGET_CUDA ON

/**
 * @def QUDA_TARGET_HIP
 * @brief This macro is set by CMake if the HIP Build target is selected
 */
/* #undef QUDA_TARGET_HIP */

/**
 * @def QUDA_TARGET_SYCL
 * @brief This macro is set by CMake if the SYCL Build target is selected
 */
/* #undef QUDA_TARGET_SYCL */

#if !defined(QUDA_TARGET_CUDA) && !defined(QUDA_TARGET_HIP) && !defined(QUDA_TARGET_SYCL)
#error "No QUDA_TARGET selected"
#endif
