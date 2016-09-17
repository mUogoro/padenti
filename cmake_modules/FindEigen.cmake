# Try to find Eigen 3 headers
#
# Once done this will define
# EIGEN_FOUND - System has Eigen 3 headers installed
# EIGEN_INCLUDE_DIRS - The Eigen 3 include directories

if (WIN32)
  set(EIGEN_PATH_GUESS "C:/Eigen" "C:/Program Files/Eigen" "C:/Program Files (x86)/Eigen")
  find_path(EIGEN_INCLUDE_DIR Eigen/Dense
            PATHS ${EIGEN_PATH_GUESS}
	        PATH_SUFFIXES "include")
elseif (UNIX AND NOT APPLE)
  find_path(EIGEN_INCLUDE_DIR Eigen/Dense
            PATHS "/usr/include")
endif (WIN32)
			   
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen DEFAULT_MSG EIGEN_INCLUDE_DIR)
mark_as_advanced(EIGEN_INCLUDE_DIR)

set(EIGEN_INCLUDE_DIRS ${EIGEN_INCLUDE_DIR})
