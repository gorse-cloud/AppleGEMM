add_library(test_utils STATIC TestUtils.cpp)
target_link_libraries(test_utils Armadillo::Armadillo Eigen3::Eigen OpenBLAS::OpenBLAS)

add_executable(mm_test MMTest.cpp)
target_link_libraries(mm_test PRIVATE GTest::GTest GTest::Main test_utils)
target_include_directories(mm_test PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(mm_benchmark MMBenchmark.cpp)
target_link_libraries(mm_benchmark PRIVATE benchmark::benchmark test_utils)
target_include_directories(mm_benchmark PRIVATE ${CMAKE_SOURCE_DIR})
