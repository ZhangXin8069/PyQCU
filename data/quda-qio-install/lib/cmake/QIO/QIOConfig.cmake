
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was QIOConfig.cmake.in                            ########

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
include(CMakeFindDependencyMacro)

# If CLime was a subtree build. The
# package info is parallel to the QIO one
# Otherwise user must specify by setting 
# CMAKE_PREFIX_PATH or by pointing -DCLime_DIR=
# to the directory containing the CLimeConfig.cmake
# file
if(TRUE)
  list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_LIST_DIR}/../CLime)
endif()

find_dependency(CLime REQUIRED)
check_required_components(CLime)

set(QIO_ENABLE_PARALLEL_BUILD ON)
# If parallel build we need to check for QMP
# QMP is always assumed external 
# Set -DQMP_DIR to point to the directory containing
# QMPConfig.cmake file
if(QIO_ENABLE_PARALLEL_BUILD)
  find_dependency(QMP REQUIRED)
  check_required_components(QMP)
endif()

# Include the generated exported targets
include(${CMAKE_CURRENT_LIST_DIR}/QIOTargets.cmake)
check_required_components(QIO)
