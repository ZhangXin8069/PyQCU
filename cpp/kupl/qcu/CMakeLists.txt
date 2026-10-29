cmake_minimum_required(VERSION 3.20)

project(kupl_qcu LANGUAGES CXX)

get_filename_component(KUPL_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

option(KUPL_BUILD_TESTS "Build KUPL QCU correctness tests" ON)
option(KUPL_ENABLE_NATIVE "Compile for the local CPU architecture" ON)
option(KUPL_ENABLE_LTO "Enable link-time optimization and local symbol binding" OFF)
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

if(KUPL_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT KUPL_IPO_SUPPORTED OUTPUT KUPL_IPO_ERROR
                      LANGUAGES CXX)
  if(NOT KUPL_IPO_SUPPORTED)
    message(FATAL_ERROR "KUPL_ENABLE_LTO is unsupported: ${KUPL_IPO_ERROR}")
  endif()
  set_property(TARGET kupl_qcu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fno-semantic-interposition"
                          KUPL_HAS_NO_SEMANTIC_INTERPOSITION)
  if(KUPL_HAS_NO_SEMANTIC_INTERPOSITION)
    target_compile_options(kupl_qcu PRIVATE -fno-semantic-interposition)
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
    add_test(NAME kupl_qcu_kunpeng_runtime_plan COMMAND
      ${Python3_EXECUTABLE}
      ${KUPL_PYTHON_TEST_DIR}/test_kunpeng_runtime.py
    )
    list(APPEND KUPL_CTEST_NAMES kupl_qcu_kunpeng_runtime_plan)

    add_test(NAME kupl_qcu_pure_kupl_contract COMMAND
      ${Python3_EXECUTABLE}
      ${KUPL_PYTHON_TEST_DIR}/test_pure_kupl_install.py
    )
    list(APPEND KUPL_CTEST_NAMES kupl_qcu_pure_kupl_contract)

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
      add_test(NAME kupl_qcu_python_dslash_block_2rank_y_c64 COMMAND
        ${CMAKE_COMMAND} -E env
        KUPL_BLOCK_DSLASH_MIN_HALF_SITES=1
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 2
        ${Python3_EXECUTABLE}
        ${KUPL_PYTHON_TEST_DIR}/test_dslash_block_mpi.py
        --library $<TARGET_FILE:kupl_qcu>
        --lattice 4 8 8 8
        --grid 1 2 1 1
        --threads ${KUPL_TEST_OMP_THREADS}
        --precision c64
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
      add_test(NAME kupl_qcu_runtime_benchmark_smoke COMMAND
        ${CMAKE_COMMAND} -E env
        KUPL_NUMA_BIND=0
        PYQCU_KUPL_LIB=$<TARGET_FILE:kupl_qcu>
        ${Python3_EXECUTABLE}
        ${KUPL_PYTHON_TEST_DIR}/benchmark_runtime.py
        --lat 4 4 4 8
        --threads ${KUPL_TEST_OMP_THREADS}
        --output ${CMAKE_CURRENT_BINARY_DIR}/runtime_benchmark_smoke.json
      )
      set_tests_properties(kupl_qcu_runtime_benchmark_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "\"status\": \"PASS\""
        TIMEOUT 120
      )
      list(APPEND KUPL_CTEST_NAMES
        kupl_qcu_python_layout
        kupl_qcu_python_new_abi_1rank
        kupl_qcu_python_new_abi_2rank
        kupl_qcu_python_new_abi_rectangular
        kupl_qcu_python_new_abi_block_rectangular
        kupl_qcu_python_new_abi_block_2rank
        kupl_qcu_python_dslash_block_2rank_c128
        kupl_qcu_python_dslash_block_2rank_y_c64
        kupl_qcu_python_multigrid_1rank
        kupl_qcu_python_multigrid_2rank
        kupl_qcu_python_strict_1rank
        kupl_qcu_python_strict_2rank
        kupl_qcu_python_strict_4rank
        kupl_qcu_python_legacy_mg_1rank
        kupl_qcu_python_legacy_mg_2rank
        kupl_qcu_python_legacy_mg_4rank
        kupl_qcu_python_failure_modes
        kupl_qcu_runtime_benchmark_smoke
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

  if(UNIX AND NOT APPLE)
    if(MPI_CXX_LIBRARY_VERSION_STRING MATCHES "Open MPI|OpenMPI|Hyper MPI")
      add_test(NAME kupl_qcu_numa_topology_report COMMAND
        ${CMAKE_COMMAND} -E env
        KUPL_NUMA_BIND=1
        KUPL_MEMORY_POLICY=preferred
        KUPL_TOPOLOGY_REPORT=1
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} 1
        --bind-to none
        $<TARGET_FILE:kupl_qcu_bench>
        --lattice 2 2 2 4
        --iterations 1
      )
      set_tests_properties(kupl_qcu_numa_topology_report PROPERTIES
        PASS_REGULAR_EXPRESSION "PYQCU::KUPL::TOPOLOGY:"
        TIMEOUT 60
      )
      list(APPEND KUPL_CTEST_NAMES kupl_qcu_numa_topology_report)
    endif()

    if(Python3_Interpreter_FOUND)
      add_test(NAME kupl_qcu_numa_disjoint_binding COMMAND
        ${Python3_EXECUTABLE}
        ${KUPL_PYTHON_TEST_DIR}/test_kunpeng_mpi_binding.py
        --binary $<TARGET_FILE:kupl_qcu_bench>
        --mpirun ${MPIEXEC_EXECUTABLE}
        --ranks 2
        --threads 2
        --synthetic-two-numa
      )
      set_tests_properties(kupl_qcu_numa_disjoint_binding PROPERTIES
        PASS_REGULAR_EXPRESSION "status=PASS ranks=2"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
      )
      list(APPEND KUPL_CTEST_NAMES kupl_qcu_numa_disjoint_binding)

      add_test(NAME kupl_qcu_numa_explicit_nodes COMMAND
        ${Python3_EXECUTABLE}
        ${KUPL_PYTHON_TEST_DIR}/test_kunpeng_mpi_binding.py
        --binary $<TARGET_FILE:kupl_qcu_bench>
        --mpirun ${MPIEXEC_EXECUTABLE}
        --ranks 4
        --threads 2
        --synthetic-two-numa
        --cpu-node-map 0,1,0,1
      )
      set_tests_properties(kupl_qcu_numa_explicit_nodes PROPERTIES
        PASS_REGULAR_EXPRESSION "status=PASS ranks=4"
        SKIP_RETURN_CODE 77
        TIMEOUT 120
      )
      list(APPEND KUPL_CTEST_NAMES kupl_qcu_numa_explicit_nodes)

      add_test(NAME kupl_qcu_kunpeng_profile_contract COMMAND
        ${Python3_EXECUTABLE}
        ${KUPL_PYTHON_TEST_DIR}/test_kunpeng_profile_contract.py
        --binary $<TARGET_FILE:kupl_qcu_bench>
        --mpirun ${MPIEXEC_EXECUTABLE}
      )
      set_tests_properties(kupl_qcu_kunpeng_profile_contract PROPERTIES
        PASS_REGULAR_EXPRESSION "status=PASS non_kunpeng_rejected=true"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
      )
      list(APPEND KUPL_CTEST_NAMES kupl_qcu_kunpeng_profile_contract)
    endif()
  endif()

  set_tests_properties(
    ${KUPL_CTEST_NAMES}
    PROPERTIES
    ENVIRONMENT
      "PYTHONPATH=${KUPL_REPO_ROOT};OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;KUPL_ALLOW_OVERSUBSCRIBE=1;KUPL_OMP_THREADS=${KUPL_TEST_OMP_THREADS};OMP_NUM_THREADS=${KUPL_TEST_OMP_THREADS}"
    TIMEOUT 300
  )
endif()
