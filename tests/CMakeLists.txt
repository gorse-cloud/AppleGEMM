add_executable(matmul_test MatMulTest.cpp TestUtils.cpp)
target_link_libraries(matmul_test PRIVATE GTest::GTest GTest::Main)
target_include_directories(matmul_test PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(matmul_benchmark MatMulBenchmark.cpp TestUtils.cpp)
target_link_libraries(matmul_benchmark PRIVATE benchmark::benchmark)
target_include_directories(matmul_benchmark PRIVATE ${CMAKE_SOURCE_DIR})
