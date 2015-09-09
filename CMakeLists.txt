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

cmake_minimum_required (VERSION 2.8)

project (Padenti)
set(Padenti_VERSION_MAJOR 0)
set(Padenti_VERSION_MINOR 1)

configure_file("${PROJECT_SOURCE_DIR}/src/padenti/padenti.hpp.in"
               "${PROJECT_BINARY_DIR}/src/padenti/padenti.hpp")

add_definitions(-DBOOST_ALL_DYN_LINK)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS random filesystem system timer chrono log REQUIRED)

execute_process(COMMAND mkdir kernels)
execute_process(COMMAND mkdir kernels/padenti)
execute_process(COMMAND ${PROJECT_SOURCE_DIR}/script/ksrc_to_hex.sh ${PROJECT_SOURCE_DIR}/src/padenti/kernels ${PROJECT_BINARY_DIR}/kernels/padenti)

file(GLOB HEADER_FILES src/padenti/*.hpp)
install(FILES ${HEADER_FILES} DESTINATION include/padenti)

if (NVIDIA)
  if (NVIDIA MATCHES "True")
    set(Padenti_OPENCL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/padenti/nvidia_opencl_headers)
    install(DIRECTORY nvidia_opencl_headers DESTINATION include/padenti)
  endif (NVIDIA MATCHES "True")
endif (NVIDIA)

file(GLOB KERNEL_FILES ${PROJECT_BINARY_DIR}/kernels/*.inc)
install(FILES ${KERNEL_FILES} DESTINATION include/padenti/kernels)

get_filename_component(Padenti_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} ABSOLUTE)
configure_file("${PROJECT_SOURCE_DIR}/PadentiConfig.cmake.in"
               "${PROJECT_BINARY_DIR}/PadentiConfig.cmake")
install(FILES ${PROJECT_BINARY_DIR}/PadentiConfig.cmake DESTINATION share/padenti)

add_subdirectory(test)

find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file("${PROJECT_SOURCE_DIR}/doc/Doxyfile.in"
                 "${PROJECT_BINARY_DIR}/doc/Doxyfile")
  add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
                    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/doc)
  install(DIRECTORY ${PROJECT_BINARY_DIR}/doc DESTINATION share/padenti)
endif(DOXYGEN_FOUND)