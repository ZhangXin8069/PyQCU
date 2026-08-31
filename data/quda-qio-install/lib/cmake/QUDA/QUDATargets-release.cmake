#----------------------------------------------------------------
# Generated CMake target import file for configuration "RELEASE".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "QUDA::quda" for configuration "RELEASE"
set_property(TARGET QUDA::quda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QUDA::quda PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libquda.so"
  IMPORTED_SONAME_RELEASE "libquda.so"
  )

list(APPEND _cmake_import_check_targets QUDA::quda )
list(APPEND _cmake_import_check_files_for_QUDA::quda "${_IMPORT_PREFIX}/lib/libquda.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
