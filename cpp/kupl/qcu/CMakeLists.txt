cmake_minimum_required(VERSION 3.20)

project(kupl_qcu LANGUAGES CXX)

option(KUPL_BUILD_TESTS "Build KUPL QCU correctness tests" ON)
option(KUPL_ENABLE_NATIVE "Compile for the local CPU architecture" ON)
set(KUPL_TEST_OMP_THREADS "2" CACHE STRING
    "OpenMP threads used by CTest correctness runs")

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(kupl_qcu SHARED
  src/apply_clover.cpp
  src/apply_clover_bistabcg.cpp
  src/apply_clover_bistabcg_dslash.cpp
  src/apply_clover_dslash.cpp
  src/apply_clover_multigrid.cpp
  src/apply_clover_multigrid_verify.cpp
  src/apply_clovers.cpp
  src/apply_dslash.cpp
  src/apply_end.cpp
  src/apply_gauss_gauge.cpp
  src/apply_init.cpp
  src/apply_laplacian.cpp
  src/apply_multigrid.cpp
  src/apply_multigrid_strict.cpp
  src/apply_test.cpp
  src/apply_wilson_bistabcg.cpp
  src/apply_wilson_bistabcg_dslash.cpp
  src/apply_wilson_cg.cpp
  src/apply_wilson_cg_dslash.cpp
  src/apply_wilson_dslash.cpp
  src/backend.cpp
  src/lattice_clover_dslash.cpp
  src/lattice_clover_multigrid.cpp
  src/lattice_cpu.cpp
  src/lattice_gauss_gauge.cpp
  src/lattice_geometry.cpp
  src/lattice_laplacian.cpp
  src/lattice_linalg.cpp
  src/lattice_multigrid.cpp
  src/lattice_solver.cpp
  src/lattice_strict.cpp
  src/lattice_wilson_dslash.cpp
)
add_library(kupl::qcu ALIAS kupl_qcu)

target_compile_features(kupl_qcu PUBLIC cxx_std_17)
target_include_directories(kupl_qcu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(kupl_qcu
  PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)
target_compile_options(kupl_qcu PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
)

if(KUPL_ENABLE_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" KUPL_HAS_MARCH_NATIVE)
  if(KUPL_HAS_MARCH_NATIVE)
    target_compile_options(kupl_qcu PRIVATE -march=native)
  endif()
endif()

set_target_properties(kupl_qcu PROPERTIES
  OUTPUT_NAME qcu
  VERSION 0.1.0
  SOVERSION 0
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)

if(UNIX AND NOT APPLE)
  target_link_options(kupl_qcu PRIVATE
    -Wl,--exclude-libs,ALL
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/qcu.map
  )
endif()

if(KUPL_BUILD_TESTS)
  enable_testing()

  add_executable(kupl_qcu_tests tests/test_kupl.cpp)
  target_compile_features(kupl_qcu_tests PRIVATE cxx_std_17)
  target_link_libraries(kupl_qcu_tests
    PRIVATE
      kupl::qcu
      MPI::MPI_CXX
      OpenMP::OpenMP_CXX
  )

  add_executable(kupl_qcu_bench tests/benchmark_kupl.cpp)
  target_compile_features(kupl_qcu_bench PRIVATE cxx_std_17)
  target_link_libraries(kupl_qcu_bench
    PRIVATE
      kupl::qcu
      MPI::MPI_CXX
      OpenMP::OpenMP_CXX
  )

  add_test(NAME kupl_qcu_1rank COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 1
    $<TARGET_FILE:kupl_qcu_tests>
    --lattice 4 2 2 4
  )
  add_test(NAME kupl_qcu_2rank COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    $<TARGET_FILE:kupl_qcu_tests>
    --lattice 2 2 2 4
    --grid 2 1 1 1
  )
  add_test(NAME kupl_qcu_2rank_y COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    $<TARGET_FILE:kupl_qcu_tests>
    --lattice 4 2 2 4
    --grid 1 2 1 1
  )
  add_test(NAME kupl_qcu_2rank_z COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    $<TARGET_FILE:kupl_qcu_tests>
    --lattice 4 2 2 4
    --grid 1 1 2 1
  )
  add_test(NAME kupl_qcu_2rank_t COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    $<TARGET_FILE:kupl_qcu_tests>
    --lattice 4 2 2 4
    --grid 1 1 1 2
  )
  add_test(NAME kupl_qcu_4rank_diagonal COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 4
    $<TARGET_FILE:kupl_qcu_tests>
    --lattice 2 2 2 4
    --grid 2 1 2 1
  )
  add_test(NAME kupl_qcu_laplacian_1rank COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 1
    $<TARGET_FILE:kupl_qcu_tests>
    --laplacian
    --lattice 3 2 2 1
    --grid 1 1 1 1
  )
  add_test(NAME kupl_qcu_laplacian_2rank COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    $<TARGET_FILE:kupl_qcu_tests>
    --laplacian
    --lattice 3 2 2 1
    --grid 2 1 1 1
  )
  add_test(NAME kupl_qcu_laplacian_2rank_y COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    $<TARGET_FILE:kupl_qcu_tests>
    --laplacian
    --lattice 3 2 2 1
    --grid 1 2 1 1
  )
  add_test(NAME kupl_qcu_laplacian_2rank_z COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 2
    $<TARGET_FILE:kupl_qcu_tests>
    --laplacian
    --lattice 3 2 2 1
    --grid 1 1 2 1
  )

  find_package(Python3 COMPONENTS Interpreter QUIET)
  set(KUPL_PYTHON_TEST_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../pyqcu/testing/kupl)
  set(KUPL_CTEST_NAMES
    kupl_qcu_1rank
    kupl_qcu_2rank
    kupl_qcu_2rank_y
    kupl_qcu_2rank_z
    kupl_qcu_2rank_t
    kupl_qcu_4rank_diagonal
    kupl_qcu_laplacian_1rank
    kupl_qcu_laplacian_2rank
    kupl_qcu_laplacian_2rank_y
    kupl_qcu_laplacian_2rank_z
  )
  if(Python3_Interpreter_FOUND)
    add_test(NAME kupl_qcu_abi_surface COMMAND
      ${Python3_EXECUTABLE}
      ${KUPL_PYTHON_TEST_DIR}/check_abi.py
      --library $<TARGET_FILE:kupl_qcu>
      --header ${CMAKE_CURRENT_SOURCE_DIR}/include/kupl/qcu.h
      --cuda-header
      ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/qcu/python/pyqcu.h
    )
    list(APPEND KUPL_CTEST_NAMES kupl_qcu_abi_surface)

    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import numpy, mpi4py"
      RESULT_VARIABLE KUPL_PYTHON_ABI_DEPS
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if(KUPL_PYTHON_ABI_DEPS EQUAL 0)
      add_test(NAME kupl_qcu_python_layout COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_canonical_layout.py
        --library $<TARGET_FILE:kupl_qcu>
      )
      add_test(NAME kupl_qcu_python_new_abi_1rank COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_new_abi.py
        --library $<TARGET_FILE:kupl_qcu>
        --lattice 2 2 2 4
        --grid 1 1 1 1
        --threads ${KUPL_TEST_OMP_THREADS}
      )
      add_test(NAME kupl_qcu_python_new_abi_2rank COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 2
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_new_abi.py
        --library $<TARGET_FILE:kupl_qcu>
        --lattice 2 2 2 4
        --grid 2 1 1 1
        --threads ${KUPL_TEST_OMP_THREADS}
      )
      add_test(NAME kupl_qcu_python_new_abi_rectangular COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_new_abi.py
        --library $<TARGET_FILE:kupl_qcu>
        --lattice 4 4 8 8
        --grid 1 1 1 1
        --threads ${KUPL_TEST_OMP_THREADS}
      )
      add_test(NAME kupl_qcu_python_new_abi_block_rectangular COMMAND
        ${CMAKE_COMMAND} -E env
        KUPL_BLOCK_DSLASH_MIN_HALF_SITES=1
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_new_abi.py
        --library $<TARGET_FILE:kupl_qcu>
        --lattice 4 4 8 8
        --grid 1 1 1 1
        --threads ${KUPL_TEST_OMP_THREADS}
      )
      add_test(NAME kupl_qcu_python_new_abi_block_2rank COMMAND
        ${CMAKE_COMMAND} -E env
        KUPL_BLOCK_DSLASH_MIN_HALF_SITES=1
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 2
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_new_abi.py
        --library $<TARGET_FILE:kupl_qcu>
        --lattice 4 4 8 8
        --grid 2 1 1 1
        --threads ${KUPL_TEST_OMP_THREADS}
      )
      add_test(NAME kupl_qcu_python_dslash_block_2rank_c128 COMMAND
        ${CMAKE_COMMAND} -E env
        KUPL_BLOCK_DSLASH_MIN_HALF_SITES=1
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 2
        ${Python3_EXECUTABLE}
        ${KUPL_PYTHON_TEST_DIR}/test_dslash_block_mpi.py
        --library $<TARGET_FILE:kupl_qcu>
        --lattice 4 4 8 8
        --grid 2 1 1 1
        --threads ${KUPL_TEST_OMP_THREADS}
        --precision c128
        --solve
      )
      add_test(NAME kupl_qcu_python_multigrid_1rank COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_multigrid.py
        --library $<TARGET_FILE:kupl_qcu>
      )
      add_test(NAME kupl_qcu_python_multigrid_2rank COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 2
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_multigrid.py
        --library $<TARGET_FILE:kupl_qcu>
      )
      add_test(NAME kupl_qcu_python_strict_1rank COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 1
        ${Python3_EXECUTABLE}
        ${KUPL_PYTHON_TEST_DIR}/test_strict.py
        --library $<TARGET_FILE:kupl_qcu>
      )
      add_test(NAME kupl_qcu_python_strict_2rank COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 2
        ${Python3_EXECUTABLE}
        ${KUPL_PYTHON_TEST_DIR}/test_strict.py
        --library $<TARGET_FILE:kupl_qcu>
      )
      add_test(NAME kupl_qcu_python_strict_4rank COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 4
        ${Python3_EXECUTABLE}
        ${KUPL_PYTHON_TEST_DIR}/test_strict.py
        --library $<TARGET_FILE:kupl_qcu>
      )
      add_test(NAME kupl_qcu_python_legacy_mg_1rank COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 1
        ${Python3_EXECUTABLE}
        ${KUPL_PYTHON_TEST_DIR}/test_legacy_mg.py
        --library $<TARGET_FILE:kupl_qcu>
      )
      add_test(NAME kupl_qcu_python_legacy_mg_2rank COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 2
        ${Python3_EXECUTABLE}
        ${KUPL_PYTHON_TEST_DIR}/test_legacy_mg.py
        --library $<TARGET_FILE:kupl_qcu>
      )
      add_test(NAME kupl_qcu_python_legacy_mg_4rank COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 4
        ${Python3_EXECUTABLE}
        ${KUPL_PYTHON_TEST_DIR}/test_legacy_mg.py
        --library $<TARGET_FILE:kupl_qcu>
      )
      add_test(NAME kupl_qcu_python_failure_modes COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_failure_modes.py
        --library $<TARGET_FILE:kupl_qcu>
      )
      list(APPEND KUPL_CTEST_NAMES
        kupl_qcu_python_layout
        kupl_qcu_python_new_abi_1rank
        kupl_qcu_python_new_abi_2rank
        kupl_qcu_python_new_abi_rectangular
        kupl_qcu_python_new_abi_block_rectangular
        kupl_qcu_python_new_abi_block_2rank
        kupl_qcu_python_dslash_block_2rank_c128
        kupl_qcu_python_multigrid_1rank
        kupl_qcu_python_multigrid_2rank
        kupl_qcu_python_strict_1rank
        kupl_qcu_python_strict_2rank
        kupl_qcu_python_strict_4rank
        kupl_qcu_python_legacy_mg_1rank
        kupl_qcu_python_legacy_mg_2rank
        kupl_qcu_python_legacy_mg_4rank
        kupl_qcu_python_failure_modes
      )
    endif()
  endif()

  add_test(NAME kupl_qcu_explicit_threads COMMAND
    ${CMAKE_COMMAND} -E env
    KUPL_OMP_THREADS=3
    OMP_NUM_THREADS=3
    $<TARGET_FILE:kupl_qcu_bench>
    --lattice 2 2 2 4
    --iterations 1
  )
  set_tests_properties(kupl_qcu_explicit_threads PROPERTIES
    PASS_REGULAR_EXPRESSION "threads_per_rank=3"
    TIMEOUT 60
  )
  list(APPEND KUPL_CTEST_NAMES kupl_qcu_explicit_threads)

  set_tests_properties(
    ${KUPL_CTEST_NAMES}
    PROPERTIES
    ENVIRONMENT
      "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;KUPL_OMP_THREADS=${KUPL_TEST_OMP_THREADS};OMP_NUM_THREADS=${KUPL_TEST_OMP_THREADS}"
    TIMEOUT 300
  )
endif()
