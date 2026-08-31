#----------------------------------------------------------------
# Generated CMake target import file for configuration "RELEASE".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "QIO::qio" for configuration "RELEASE"
set_property(TARGET QIO::qio APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QIO::qio PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libqio.so"
  IMPORTED_SONAME_RELEASE "libqio.so"
  )

list(APPEND _cmake_import_check_targets QIO::qio )
list(APPEND _cmake_import_check_files_for_QIO::qio "${_IMPORT_PREFIX}/lib/libqio.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
