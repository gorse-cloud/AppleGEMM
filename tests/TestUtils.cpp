#include "TestUtils.h"

#include <random>

namespace {
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(0, 100);
} // namespace

std::vector<float> random_matrix(uint64_t m, uint64_t n) {
  std::vector<float> v(n * m);
  for (size_t i = 0; i < m * n; i++) {
    v[i] = dist(gen);
  }
  return v;
}

Eigen::MatrixXf random_matrix_eigen(uint64_t m, uint64_t n) {
  Eigen::MatrixXf v(m, n);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      v << dist(gen);
    }
  }
  return v;
}

void matmul(const float *a, const float *b, float *c, uint64_t m, uint64_t n,
            uint64_t k) {
  for (uint64_t i = 0; i < m; i++) {
    for (uint64_t j = 0; j < n; j++) {
      c[i * n + j] = 0;
      for (uint64_t l = 0; l < k; l++) {
        c[i * n + j] += a[i * k + l] * b[l * n + j];
      }
    }
  }
}
