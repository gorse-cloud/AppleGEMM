#include "TestUtils.h"

#include <random>

#include <arm_neon.h>
#include <stdint.h>

void vmul_const_add_to(float *a, float *b, float *c, long n) {
  int epoch = n / 4;
  int remain = n % 4;
  for (int i = 0; i < epoch; i++) {
    float32x4_t v1 = vld1q_f32(a);
    float32x4_t v3 = vld1q_f32(c);
    float32x4_t v = vmlaq_n_f32(v3, v1, *b);
    vst1q_f32(c, v);
    a += 4;
    c += 4;
  }
  for (int i = 0; i < remain; i++) {
    c[i] += a[i] * b[0];
  }
}

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

arma::mat random_matrix_arma(uint64_t m, uint64_t n) {
  arma::mat v(m, n);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      v(i, j) = dist(gen);
    }
  }
  return v;
}

void mm(const float *a, const float *b, float *c, uint64_t m, uint64_t n,
            uint64_t k, bool transA, bool transB) {
  for (uint64_t i = 0; i < m; i++) {
    for (uint64_t j = 0; j < n; j++) {
      c[i * n + j] = 0;
    }
  }
  if (!transA && !transB) {
    for (uint64_t i = 0; i < m; i++) {
      for (uint64_t l = 0; l < k; l++) {
        for (uint64_t j = 0; j < n; j++) {
          c[i * n + j] += a[i * k + l] * b[l * n + j];
        }
      }
    }
  } else if (!transA && transB) {
    for (uint64_t i = 0; i < m; i++) {
      for (uint64_t j = 0; j < n; j++) {
        c[i * n + j] = 0;
        for (uint64_t l = 0; l < k; l++) {
          c[i * n + j] += a[i * k + l] * b[j * k + l];
        }
      }
    }
  } else if (transA && !transB) {
    for (uint64_t i = 0; i < m; i++) {
      for (uint64_t j = 0; j < n; j++) {
        c[i * n + j] = 0;
        for (uint64_t l = 0; l < k; l++) {
          c[i * n + j] += a[l * m + i] * b[l * n + j];
        }
      }
    }
  } else {
    for (uint64_t i = 0; i < m; i++) {
      for (uint64_t j = 0; j < n; j++) {
        c[i * n + j] = 0;
        for (uint64_t l = 0; l < k; l++) {
          c[i * n + j] += a[l * m + i] * b[j * k + l];
        }
      }
    }
  }
}
