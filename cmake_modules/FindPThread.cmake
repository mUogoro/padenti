# Try to find Pthread Windows wrappers
#
# Once done this will define
# PTHREAD_FOUND - System has Pthread wrappers installed
# PTHREAD_INCLUDE_DIRS - The Pthread wrappers include directories
# PTHREAD_LIBRARY_DIRS - The Pthread wrappers library directories
# PTHREAD_LIBRARIES - The libraries needed to use Pthread wrappers

if (WIN32)
  set(PTHREAD_PATH_GUESS "C:/pthread" "C:/Program Files/pthread" "C:/Program Files (x86)/pthread")
  find_path(PTHREAD_INCLUDE_DIR pthread.h semaphore.h
            PATHS ${PTHREAD_PATH_GUESS}
	        PATH_SUFFIXES "include")
  find_library(PTHREAD_LIBRARY NAMES pthreadVC2
               PATHS ${PTHREAD_PATH_GUESS}
               PATH_SUFFIXES "lib/x64")
elseif (UNIX AND NOT APPLE)
  find_path(PTHREAD_INCLUDE_DIR pthread.h semaphore.h
            PATHS "/usr/include")
  find_library(PTHREAD_LIBRARY NAMES pthread
               PATHS "/usr/lib")
endif (WIN32)
			   
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PThread DEFAULT_MSG
                                  PTHREAD_LIBRARY PTHREAD_INCLUDE_DIR)
mark_as_advanced(PTHREAD_INCLUDE_DIR PTHREAD_LIBRARY)

get_filename_component(PTHREAD_LIBRARIES ${PTHREAD_LIBRARY} NAME)
get_filename_component(PTHREAD_LIBRARY_DIRS ${PTHREAD_LIBRARY} DIRECTORY)
set(PTHREAD_INCLUDE_DIRS ${PTHREAD_INCLUDE_DIR})