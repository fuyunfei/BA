cmake_minimum_required(VERSION 2.8)

project(BA)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

# helloworld
add_executable(BA BA.cc)
target_link_libraries(BA ${CERES_LIBRARIES})
