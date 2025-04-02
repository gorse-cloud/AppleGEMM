add_library(test_utils STATIC TestUtils.cpp)
target_link_libraries(test_utils Armadillo::Armadillo Eigen3::Eigen OpenBLAS::OpenBLAS)

add_executable(matmul_test MatMulTest.cpp)
target_link_libraries(matmul_test PRIVATE GTest::GTest GTest::Main test_utils)
target_include_directories(matmul_test PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(matmul_benchmark MatMulBenchmark.cpp)
target_link_libraries(matmul_benchmark PRIVATE benchmark::benchmark test_utils)
target_include_directories(matmul_benchmark PRIVATE ${CMAKE_SOURCE_DIR})
