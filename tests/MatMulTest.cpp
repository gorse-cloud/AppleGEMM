#include "TestUtils.h"
#include "apple_gemm.h"
#include "gtest/gtest.h"

TEST(MatMulTest, matmul) {
  constexpr uint64_t m = 48, n = 32, k = 64;
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  auto d = random_matrix(m, n);
  matmul(a.data(), b.data(), c.data(), m, n, k);
  apple_matmul(a.data(), b.data(), d.data(), m, n, k);
  GTEST_ASSERT_EQ(c, d);
}
