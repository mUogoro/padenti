# Try to find OpenCV
#
# Note: simple "wrapper" that guess the OpenCV location, include the library using
# the provided cmake files and adapt the exported variable names
# Note: only core, highgui and imgproc modelus get imported
#
# Once done this will define
# OPENCV_FOUND - System has OpenCV installed
# OPENCV_INCLUDE_DIRS - The OpenCV include directories
# OPENCV_LIBRARY_DIRS - The OpenCV library directories
# OPENCV_LIBRARIES - The libraries needed to use OpenCV


if (WIN32)
  set(OPENCV_PATH_GUESS "c:/opencv" "c:/opencv/build" "c:/Program Files/opencv/" "c:/Program Files/opencv/build")
  find_package(OpenCV COMPONENTS core imgproc highgui PATHS ${OPENCV_PATH_GUESS} REQUIRED)
elseif (UNIX AND NOT APPLE)
  #On Linux we suppose no guess is needed
  find_package(OpenCV COMPONENTS core imgproc highgui REQUIRED)
endif (WIN32)

if (CMAKE_BUILD_TYPE)
  string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE_UP_NAME)
  if (BUILD_TYPE_UP_NAME MATCHES "DEBUG")
    set(LIB_DEBUG_SUFFIX "d")
  endif (BUILD_TYPE_UP_NAME MATCHES "DEBUG")
endif (CMAKE_BUILD_TYPE)

set(OPENCV_FOUND ${OpenCV_FOUND})
set(OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
set(OPENCV_LIBRARY_DIRS ${OpenCV_LIB_DIR})
set(OPENCV_LIBRARIES "")
# Build the list of library modules (.lib on win, .so on linux)
foreach(OPENCV_MODULE ${OpenCV_LIBS})
  if (WIN32)
    set(OPENCV_LIBRARIES "${CMAKE_STATIC_LIBRARY_PREFIX}${OPENCV_MODULE}${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}${LIB_DEBUG_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}" ${OPENCV_LIBRARIES})
  elseif (UNIX AND NOT APPLE)
    set(OPENCV_LIBRARIES "${CMAKE_SHARED_LIBRARY_PREFIX}${OPENCV_MODULE}${CMAKE_SHARED_LIBRARY_SUFFIX}" ${OPENCV_LIBRARIES})
  endif (WIN32)
endforeach(OPENCV_MODULE ${OpenCV_LIBS})
