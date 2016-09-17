###############################################################################
# Padenti Library
#
# Copyright (C) 2015  Daniele Pianu <daniele.pianu@ieiit.cnr.it>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
###############################################################################

cmake_minimum_required (VERSION 3.1)

project (Padenti)
set(Padenti_VERSION_MAJOR 0)
set(Padenti_VERSION_MINOR 2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")

configure_file("${PROJECT_SOURCE_DIR}/src/padenti/padenti.hpp.in"
               "${PROJECT_BINARY_DIR}/src/padenti/padenti.hpp")

add_definitions(-DBOOST_ALL_DYN_LINK)
find_package(OpenCL REQUIRED)
find_package(PThread REQUIRED)
find_package(Eigen REQUIRED)
find_package(OpenCVHighGuiImgProc REQUIRED)
find_package(Boost COMPONENTS random filesystem system timer chrono log REQUIRED)

if (WIN32)
  find_package(Cygwin REQUIRED)

  # Enable cmath macros in Visual Studio
  add_definitions(-D_USE_MATH_DEFINES)
endif (WIN32)

# Set output directories
# - binaries (e.g. tests) in /bin
# - libraries (static and shared) and debug info in /lib
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_BINARY_DIR}/lib")
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

# OpenCL kernels' processing
if (WIN32)
  string(SUBSTRING ${PROJECT_SOURCE_DIR} 0 1 DRIVE)
  string(TOLOWER ${DRIVE} DRIVE)
  string(SUBSTRING ${PROJECT_SOURCE_DIR}/script/ksrc_to_hex.sh 3 -1 SCRIPT_PATH)
  string(SUBSTRING ${PROJECT_SOURCE_DIR}/src/padenti/kernels 3 -1 KERNELS_SRC_PATH)
  string(SUBSTRING ${PROJECT_BINARY_DIR}/kernels/padenti 3 -1 KERNELS_HEX_PATH)
  execute_process(COMMAND cmd.exe /c mkdir kernels)
  execute_process(COMMAND cmd.exe /c mkdir kernels\\padenti)
  execute_process(COMMAND c:/cygwin64/bin/bash.exe --login /cygdrive/${DRIVE}/${SCRIPT_PATH} /cygdrive/${DRIVE}/${KERNELS_SRC_PATH} /cygdrive/${DRIVE}/${KERNELS_HEX_PATH})
else (WIN32)
  execute_process(COMMAND mkdir kernels)
  execute_process(COMMAND mkdir kernels/padenti)
  execute_process(COMMAND ${PROJECT_SOURCE_DIR}/script/ksrc_to_hex.sh ${PROJECT_SOURCE_DIR}/src/padenti/kernels ${PROJECT_BINARY_DIR}/kernels/padenti)
endif (WIN32)

#if (NVIDIA)
#  if (NVIDIA MATCHES "True")
#    set(Padenti_OPENCL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/padenti/nvidia_opencl_headers)
#    install(DIRECTORY nvidia_opencl_headers DESTINATION include/padenti)
#  endif (NVIDIA MATCHES "True")
#endif (NVIDIA)

# Installation directives
get_filename_component(Padenti_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} ABSOLUTE)
set(Padenti_EXT_INCLUDE_DIRS ${PTHREAD_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR})
set(Padenti_EXT_LIBRARY_DIRS ${PTHREAD_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS} ${OPENCV_LIBRARY_DIRS})
set(Padenti_EXT_LIBRARIES ${PTHREAD_LIBRARIES} ${OPENCV_LIBRARIES} ${Boost_RANDOM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_CHRONO_LIBRARY} ${Boost_LOG_LIBRARY} ${OpenCL_LIBRARY})
configure_file("${PROJECT_SOURCE_DIR}/PadentiConfig.cmake.in"
               "${PROJECT_BINARY_DIR}/PadentiConfig.cmake")
if (WIN32)
  install(FILES ${PROJECT_BINARY_DIR}/PadentiConfig.cmake DESTINATION /)
else (WIN32)
  install(FILES ${PROJECT_BINARY_DIR}/PadentiConfig.cmake DESTINATION share/padenti)
endif (WIN32)
file(GLOB HEADER_FILES src/padenti/*.hpp)
install(FILES ${HEADER_FILES} DESTINATION include/padenti)
file(GLOB KERNEL_FILES "${PROJECT_BINARY_DIR}/kernels/padenti/*.inc")
install(FILES ${KERNEL_FILES} DESTINATION include/padenti)

add_subdirectory(test)

find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file("${PROJECT_SOURCE_DIR}/doc/Doxyfile.in"
                 "${PROJECT_BINARY_DIR}/doc/Doxyfile")
  add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
                    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/doc)
  install(DIRECTORY ${PROJECT_BINARY_DIR}/doc DESTINATION share/padenti)
endif(DOXYGEN_FOUND)