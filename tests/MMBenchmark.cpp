#include "Eigen/Dense"
#include "TestUtils.h"
#include "apple_gemm.h"
#include "benchmark/benchmark.h"
#include "cblas.h"

constexpr uint64_t m = 64, n = 64, k = 64;

class MMBenchmark : public benchmark::Fixture {};

BENCHMARK_F(MMBenchmark, Naive)(benchmark::State& state) {
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    mm(a.data(), b.data(), c.data(), m, n, k, false, false);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_F(MMBenchmark, Eigen)(benchmark::State& state) {
  auto a = random_matrix_eigen(m, k);
  auto b = random_matrix_eigen(k, n);
  Eigen::MatrixXf c(m, n);
  for (auto _ : state) {
    c = a * b;
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_F(MMBenchmark, Armadillo)(benchmark::State& state) {
  auto a = random_matrix_arma(m, k);
  auto b = random_matrix_arma(k, n);
  arma::mat c(m, n);
  for (auto _ : state) {
    c = a * b;
    benchmark::DoNotOptimize(&c);
  }
}

BENCHMARK_F(MMBenchmark, OpenBLAS)(benchmark::State& state) {
  auto a = random_matrix(m, k);
  auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                a.data(), k, b.data(), n, 0.0f, c.data(), n);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_F(MMBenchmark, AppleGEMM)(benchmark::State& state) {
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    apple_mm(a.data(), b.data(), c.data(), m, n, k, false, false);
    benchmark::DoNotOptimize(c);
  }
}

class MMBenchmarkTransposeA : public benchmark::Fixture {};

BENCHMARK_F(MMBenchmarkTransposeA, Naive)(benchmark::State& state) {
  const auto a = random_matrix(k, m);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    mm(a.data(), b.data(), c.data(), m, n, k, true, false);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_F(MMBenchmarkTransposeA, Eigen)(benchmark::State& state) {
  auto a = random_matrix_eigen(k, m);
  auto b = random_matrix_eigen(k, n);
  Eigen::MatrixXf c(m, n);
  for (auto _ : state) {
    c = a.transpose() * b;
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_F(MMBenchmarkTransposeA, Armadillo)(benchmark::State& state) {
  auto a = random_matrix_arma(k, m);
  auto b = random_matrix_arma(k, n);
  arma::mat c(m, n);
  for (auto _ : state) {
    c = a.t() * b;
    benchmark::DoNotOptimize(&c);
  }
}

BENCHMARK_F(MMBenchmarkTransposeA, OpenBLAS)(benchmark::State& state) {
  auto a = random_matrix(k, m);
  auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0f,
                a.data(), m, b.data(), n, 0.0f, c.data(), n);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_F(MMBenchmarkTransposeA, AppleGEMM)(benchmark::State& state) {
  const auto a = random_matrix(k, m);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    apple_mm(a.data(), b.data(), c.data(), m, n, k, true, false);
    benchmark::DoNotOptimize(c);
  }
}

class MMBenchmarkTransposeB : public benchmark::Fixture {};

BENCHMARK_F(MMBenchmarkTransposeB, Naive)(benchmark::State& state) {
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(n, k);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    mm(a.data(), b.data(), c.data(), m, n, k, false, true);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_F(MMBenchmarkTransposeB, Eigen)(benchmark::State& state) {
  auto a = random_matrix_eigen(m, k);
  auto b = random_matrix_eigen(n, k);
  Eigen::MatrixXf c(m, n);
  for (auto _ : state) {
    c = a * b.transpose();
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_F(MMBenchmarkTransposeB, Armadillo)(benchmark::State& state) {
  auto a = random_matrix_arma(m, k);
  auto b = random_matrix_arma(n, k);
  arma::mat c(m, n);
  for (auto _ : state) {
    c = a * b.t();
    benchmark::DoNotOptimize(&c);
  }
}

BENCHMARK_F(MMBenchmarkTransposeB, OpenBLAS)(benchmark::State& state) {
  auto a = random_matrix(m, k);
  auto b = random_matrix(n, k);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f,
                a.data(), k, b.data(), k, 0.0f, c.data(), n);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_F(MMBenchmarkTransposeB, AppleGEMM)(benchmark::State& state) {
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(n, k);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    apple_mm(a.data(), b.data(), c.data(), m, n, k, false, true);
    benchmark::DoNotOptimize(c);
  }
}

class MMBenchmarkTransposeAB : public benchmark::Fixture {};

BENCHMARK_F(MMBenchmarkTransposeAB, Naive)(benchmark::State& state) {
  const auto a = random_matrix(k, m);
  const auto b = random_matrix(n, k);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    mm(a.data(), b.data(), c.data(), m, n, k, true, true);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_F(MMBenchmarkTransposeAB, Eigen)(benchmark::State& state) {
  auto a = random_matrix_eigen(k, m);
  auto b = random_matrix_eigen(n, k);
  Eigen::MatrixXf c(m, n);
  for (auto _ : state) {
    c = a.transpose() * b.transpose();
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_F(MMBenchmarkTransposeAB, Armadillo)(benchmark::State& state) {
  auto a = random_matrix_arma(k, m);
  auto b = random_matrix_arma(n, k);
  arma::mat c(m, n);
  for (auto _ : state) {
    c = a.t() * b.t();
    benchmark::DoNotOptimize(&c);
  }
}

BENCHMARK_F(MMBenchmarkTransposeAB, OpenBLAS)(benchmark::State& state) {
  auto a = random_matrix(k, m);
  auto b = random_matrix(n, k);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, m, n, k, 1.0f,
                a.data(), m, b.data(), k, 0.0f, c.data(), n);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_F(MMBenchmarkTransposeAB, AppleGEMM)(benchmark::State& state) {
  const auto a = random_matrix(k, m);
  const auto b = random_matrix(n, k);
  auto c = random_matrix(m, n);
  for (auto _ : state) {
    apple_mm(a.data(), b.data(), c.data(), m, n, k, true, true);
    benchmark::DoNotOptimize(c);
  }
}

BENCHMARK_MAIN();
