add_executable(gemm_test GEMMTest.cpp)
target_link_libraries(gemm_test PRIVATE GTest::GTest)

add_executable(gemm_benchmark GEMMBenchmark.cpp)
target_link_libraries(gemm_benchmark PRIVATE benchmark::benchmark)
