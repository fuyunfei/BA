cmake_minimum_required(VERSION 2.8)

project(onlyfocal)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

# helloworld
 
add_executable(BA BA.cpp)
target_link_libraries(BA ${CERES_LIBRARIES})
add_executable(onlyfocal onlyfocal.cpp)
target_link_libraries(onlyfocal ${CERES_LIBRARIES})

