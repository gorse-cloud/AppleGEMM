#include "TestUtils.h"
#include "apple_gemm.h"
#include "cblas.h"
#include "gtest/gtest.h"

TEST(MatMulTest, matmul) {
  constexpr uint64_t m = 63, n = 47, k = 79;
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  auto d = random_matrix(m, n);
  matmul(a.data(), b.data(), c.data(), m, n, k);
  apple_matmul(a.data(), b.data(), d.data(), m, n, k);
  GTEST_ASSERT_EQ(c, d);
}

TEST(MatMulTest, OpenBLAS) {
  constexpr uint64_t m = 63, n = 47, k = 79;
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  auto d = random_matrix(m, n);
  matmul(a.data(), b.data(), c.data(), m, n, k);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
              a.data(), k, b.data(), n, 0.0f, d.data(), n);
  GTEST_ASSERT_EQ(c, d);
}
