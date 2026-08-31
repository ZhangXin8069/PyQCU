
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was QMPConfig.cmake.in                            ########

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

# Boiler Plate Config file
#
#
# Include our targets
include(CMakeFindDependencyMacro)

# Include the imported targets
include(${CMAKE_CURRENT_LIST_DIR}/QMPTargets.cmake)

# Set the CMakeModule path to include the current (substituted) package 
# directory in order so we can find Dmalloc if we need it
list(APPEND CMAKE_MODULE_PATH ${PACKAGE_PREFIX_DIR}/lib/cmake/QMP)

# This will let us check if we need MPI
set(QMP_MPI ON)

# This will let us check if we need DMalloc
set(QMP_USE_DMALLOC OFF)

# Resolve dependencies if needed
if( QMP_MPI )
  find_dependency(MPI)
  if( NOT MPI_C_FOUND )
    message(ERROR "Could not find MPI_C")
  endif()
endif()

# Resolve dependencies if needed
if( QMP_USE_DMALLOC )
  find_dependency(Dmalloc REQUIRED)
endif()

# Boiler plate stuff may not be needed now
check_required_components(MPI)
check_required_components(Dmalloc)
check_required_components(QMP)
