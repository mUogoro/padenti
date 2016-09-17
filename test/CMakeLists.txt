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

if (WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp /arch:SSE2")
else (WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native")
endif (WIN32)

include_directories(${PTHREAD_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR}
                    ${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/kernels)
link_directories(${PTHREAD_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS} ${OPENCV_LIBRARY_DIRS} ${PROJECT_SOURCE_DIR})

#add_executable(test_training_set test_training_set.cpp)
#target_link_libraries(test_training_set ${OpenCV_LIBS} ${Boost_RANDOM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})

add_executable(test_tree_trainer test_tree_trainer.cpp)
target_link_libraries(test_tree_trainer ${PTHREAD_LIBRARIES} ${OPENCV_LIBRARIES} ${Boost_RANDOM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_CHRONO_LIBRARY} ${Boost_LOG_LIBRARY} ${OpenCL_LIBRARY})

add_executable(test_classifier test_classifier.cpp)
target_link_libraries(test_classifier ${OPENCV_LIBRARIES} ${Boost_RANDOM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${OpenCL_LIBRARY})

add_executable(test_rtree_trainer test_rtree_trainer.cpp)
target_link_libraries(test_rtree_trainer ${PTHREAD_LIBRARIES} ${OPENCV_LIBRARIES} ${Boost_RANDOM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_CHRONO_LIBRARY} ${Boost_LOG_LIBRARY} ${OpenCL_LIBRARY})

add_executable(test_regressor test_regressor.cpp)
target_link_libraries(test_regressor ${OPENCV_LIBRARIES} ${Boost_RANDOM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${OpenCL_LIBRARY})


if (WIN32)
  install(TARGETS test_tree_trainer DESTINATION test)
  install(TARGETS test_classifier DESTINATION test)
  install(TARGETS test_rtree_trainer DESTINATION test)
  install(TARGETS test_regressor DESTINATION test)
  install(FILES ${PROJECT_SOURCE_DIR}/test/feature.cl DESTINATION test)
else (WIN32)
  install(TARGETS test_tree_trainer DESTINATION share/padenti/test)
  install(TARGETS test_classifier DESTINATION share/padenti/test)
  install(TARGETS test_rtree_trainer DESTINATION share/padenti/test)
  install(TARGETS test_regressor DESTINATION share/padenti/test)
  install(FILES ${PROJECT_SOURCE_DIR}/test/feature.cl DESTINATION share/padenti/test)
endif (WIN32)
