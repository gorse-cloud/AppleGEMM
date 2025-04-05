#include "TestUtils.h"
#include "apple_gemm.h"
#include "cblas.h"
#include "gtest/gtest.h"

#define EXPECT_FLOATS_EQ(a, b, m, n)                                           \
  for (size_t i = 0; i < m; i++) {                                             \
    for (size_t j = 0; j < n; j++) {                                           \
      EXPECT_FLOAT_EQ(a[i * n + j], b[i * n + j])                              \
          << "at (" << i << "," << j << ")";                                   \
    }                                                                          \
  }

constexpr uint64_t m = 64, n = 64, k = 64;

TEST(AppleMMTest, noTrans) {
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  auto d = random_matrix(m, n);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
              a.data(), k, b.data(), n, 0.0f, c.data(), n);
  apple_mm(a.data(), b.data(), d.data(), m, n, k, false, false);
  EXPECT_FLOATS_EQ(c, d, m, n);
}

TEST(AppleMMTest, transposeA) {
  const auto a = random_matrix(k, m);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  auto d = random_matrix(m, n);
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0f, a.data(),
              m, b.data(), n, 0.0f, c.data(), n);
  apple_mm(a.data(), b.data(), d.data(), m, n, k, true, false);
  EXPECT_FLOATS_EQ(c, d, m, n);
}

TEST(AppleMMTest, transposeB) {
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(n, k);
  auto c = random_matrix(m, n);
  auto d = random_matrix(m, n);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a.data(),
              k, b.data(), k, 0.0f, c.data(), n);
  apple_mm(a.data(), b.data(), d.data(), m, n, k, false, true);
  EXPECT_FLOATS_EQ(c, d, m, n);
}

TEST(AppleMMTest, transposeAB) {
  const auto a = random_matrix(k, m);
  const auto b = random_matrix(n, k);
  auto c = random_matrix(m, n);
  auto d = random_matrix(m, n);
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, m, n, k, 1.0f, a.data(), m,
              b.data(), k, 0.0f, c.data(), n);
  apple_mm(a.data(), b.data(), d.data(), m, n, k, true, true);
  EXPECT_FLOATS_EQ(c, d, m, n);
}

TEST(MMTest, noTrans) {
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  auto d = random_matrix(m, n);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
              a.data(), k, b.data(), n, 0.0f, c.data(), n);
  mm(a.data(), b.data(), d.data(), m, n, k, false, false);
  EXPECT_FLOATS_EQ(c, d, m, n);
}

TEST(MMTest, transposeA) {
  const auto a = random_matrix(k, m);
  const auto b = random_matrix(k, n);
  auto c = random_matrix(m, n);
  auto d = random_matrix(m, n);
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0f, a.data(),
              m, b.data(), n, 0.0f, c.data(), n);
  mm(a.data(), b.data(), d.data(), m, n, k, true, false);
  EXPECT_FLOATS_EQ(c, d, m, n);
}

TEST(MMTest, transposeB) {
  const auto a = random_matrix(m, k);
  const auto b = random_matrix(n, k);
  auto c = random_matrix(m, n);
  auto d = random_matrix(m, n);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a.data(),
              k, b.data(), k, 0.0f, c.data(), n);
  mm(a.data(), b.data(), d.data(), m, n, k, false, true);
  EXPECT_FLOATS_EQ(c, d, m, n);
}

TEST(MMTest, transposeAB) {
  const auto a = random_matrix(k, m);
  const auto b = random_matrix(n, k);
  auto c = random_matrix(m, n);
  auto d = random_matrix(m, n);
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, m, n, k, 1.0f, a.data(), m,
              b.data(), k, 0.0f, c.data(), n);
  mm(a.data(), b.data(), d.data(), m, n, k, true, true);
  EXPECT_FLOATS_EQ(c, d, m, n);
}
