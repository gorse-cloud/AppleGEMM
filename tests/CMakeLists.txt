add_executable(gemm_test GEMMTest.cpp)
target_link_libraries(gemm_test PRIVATE GTest::GTest GTest::Main)
target_include_directories(gemm_test PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(gemm_benchmark GEMMBenchmark.cpp)
target_link_libraries(gemm_benchmark PRIVATE benchmark::benchmark)
