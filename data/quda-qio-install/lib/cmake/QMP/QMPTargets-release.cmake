#----------------------------------------------------------------
# Generated CMake target import file for configuration "RELEASE".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "QMP::qmp" for configuration "RELEASE"
set_property(TARGET QMP::qmp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(QMP::qmp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libqmp.so"
  IMPORTED_SONAME_RELEASE "libqmp.so"
  )

list(APPEND _cmake_import_check_targets QMP::qmp )
list(APPEND _cmake_import_check_files_for_QMP::qmp "${_IMPORT_PREFIX}/lib/libqmp.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
