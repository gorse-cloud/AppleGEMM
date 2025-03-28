#pragma once

#include "apple_amx.h"

inline void apple_matmul(const float *a, const float *b, float *c, uint64_t m,
                         uint64_t n, uint64_t k) {
  for (uint64_t ii = 0; ii < m; ii += 16) {
    for (uint64_t jj = 0; jj < n; jj += 16) {
      amx_set();
      for (uint64_t kk = 0; kk < k; kk += 16) {
        for (uint64_t i = 0; i < 16; i++) {
          amx_ldz_f32(i * 4 + 1, a + (ii + i) * k + kk);
        }
        for (uint64_t i = 0; i < 16; i++) {
          amx_extrv_f32(i * 4 + 1);
          amx_ldx_f32(b + (kk + i) * n + jj);
          amx_fma_f32();
        }
      }
      for (uint64_t i = 0; i < 16; i++) {
        amx_stz_f32(i * 4, c + (ii + i) * n + jj);
      }
      amx_clr();
    }
  }
}
