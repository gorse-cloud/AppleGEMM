#include "Eigen/Dense"
#include "TestUtils.h"
#include "apple_gemm.h"
#include "benchmark/benchmark.h"
#include "cblas.h"

constexpr uint64_t m = 16, n = 16, k = 16;

static void BenchmarkMatMulNaive(benchmark::State &state) {
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    matmul(a.data(), b.data(), c.data(), m, n, k);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK(BenchmarkMatMulNaive);

static void BenchmarkMatMulEigen(benchmark::State &state) {
  auto a = random_matrix_eigen(m, k);
  auto b = random_matrix_eigen(k, n);
  Eigen::MatrixXf c(m, n);
  for (auto _ : state) {
    c = a * b;
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK(BenchmarkMatMulEigen);

static void BenchmarkMatMulArmadillo(benchmark::State &state) {
  auto a = random_matrix_arma(m, k);
  auto b = random_matrix_arma(k, n);
  arma::mat c(m, n);
  for (auto _ : state) {
    c = a * b;
    benchmark::DoNotOptimize(&c);
  }
}

BENCHMARK(BenchmarkMatMulArmadillo);

static void BenchmarkMatMulOpenBLAS(benchmark::State &state) {
  auto a = random_matrix(m, k);
  auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                a.data(), k, b.data(), n, 0.0f, c.data(), n);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK(BenchmarkMatMulOpenBLAS);

static void BenchmarkMatMulAppleGEMM(benchmark::State &state) {
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    apple_matmul(a.data(), b.data(), c.data(), m, n, k);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK(BenchmarkMatMulAppleGEMM);

BENCHMARK_MAIN();
