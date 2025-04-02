cmake_minimum_required(VERSION 3.25)
project(AppleGEMM)

set(CMAKE_CXX_STANDARD 23)

find_package(Armadillo REQUIRED)
find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenBLAS REQUIRED)

add_subdirectory(tests)
