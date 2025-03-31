#include "Eigen/Dense"
#include "TestUtils.h"
#include "apple_gemm.h"
#include "benchmark/benchmark.h"

constexpr uint64_t m = 48, n = 32, k = 64;

static void BenchmarkMatMul(benchmark::State &state) {
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    matmul(a.data(), b.data(), c.data(), m, n, k);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK(BenchmarkMatMul);

static void BenchmarkMatMulApple(benchmark::State &state) {
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    apple_matmul(a.data(), b.data(), c.data(), m, n, k);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK(BenchmarkMatMulApple);

static void BenchmarkMatMulXTensor(benchmark::State &state) {
  auto a = random_matrix_eigen(m, k);
  auto b = random_matrix_eigen(k, n);
  Eigen::MatrixXf c(m, n);
  for (auto _ : state) {
    c = a * b;
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK(BenchmarkMatMulXTensor);

BENCHMARK_MAIN();
