#include "apple_gemm.h"
#include "gtest/gtest.h"

#include <random>
#include <vector>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(0.0, 100);

std::vector<float> random_matrix(size_t m, size_t n) {
  std::vector<float> v(n * m);
  for (size_t i = 0; i < m * n; i++) {
    v[i] = dis(gen);
  }
  return v;
}

void transpose(const float *a, float *c, uint64_t m, uint64_t n) {
  for (uint64_t i = 0; i < m; i++) {
    for (uint64_t j = 0; j < n; j++) {
      c[j * m + i] = a[i * n + j];
    }
  }
}

TEST(GEMMTest, transpose) {
  auto a = random_matrix(32, 32);
  auto c = random_matrix(32, 32);
  auto d = random_matrix(32, 32);
  amx_transpose(a.data(), c.data(), 32, 32);
  transpose(a.data(), d.data(), 32, 32);
  GTEST_ASSERT_EQ(c, d);
}
