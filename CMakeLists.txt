cmake_minimum_required(VERSION 3.25)
project(AppleGEMM)

set(CMAKE_CXX_STANDARD 23)

find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)

add_subdirectory(tests)
