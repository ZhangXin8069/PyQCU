
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was QUDAConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

set(QUDA_QMP ON)
set(QUDA_MPI OFF)
set(QUDA_QIO ON)
set(QUDA_OPENMP OFF)
set(QUDA_QDPJIT OFF)
set(QUDA_GITVERSION 1.1.0-80c5fae-dirty-sm_70)
set(QUDA_PRECISION 12)
set(QUDA_RECONSTRUCT 7)

set(QUDA_TARGET_CUDA ON)
set(QUDA_TARGET_HIP  )

set(QUDA_NVSHMEM  OFF)

if( QUDA_QMP AND QUDA_MPI )
  message(FATAL_ERROR "Cannot have both QMP and MPI configured")  
endif()

# Everyone needs this
find_dependency(Threads REQUIRED)

if( QUDA_QMP )
  find_dependency(QMP REQUIRED)
endif()

if( QUDA_MPI )
  find_dependency(MPI REQUIRED)
endif()

if( QUDA_QIO )
  find_dependency(QIO REQUIRED)
endif()

if( QUDA_OPENMP )
  find_dependency(OpenMP REQUIRED)
endif()

if( QUDA_TARGET_CUDA )
  include(${CMAKE_CURRENT_LIST_DIR}/find_target_cuda_dependencies.cmake)
elseif(QUDA_TARGET_HIP )
  include(${CMAKE_CURRENT_LIST_DIR}/find_target_hip_dependencies.cmake )
endif()

if( QUDA_QDPJIT )
  find_dependency( QDPXX REQUIRED )
endif()

include(${CMAKE_CURRENT_LIST_DIR}/QUDATargets.cmake)


