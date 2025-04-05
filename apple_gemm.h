#pragma once

#include "apple_amx.h"

inline void apple_mm(const float *a, const float *b, float *c, uint64_t m,
                     uint64_t n, uint64_t k, bool transA, bool transB) {
  if (!transA && !transB) { // (m, k) * (k, n) -> (m, n)
    uint64_t ii = 0;
    for (; ii + 15 < m; ii += 16) {
      uint64_t jj = 0;
      for (; jj + 15 < n; jj += 16) {
        uint64_t kk = 0;
        amx_set();
        for (; kk + 15 < k; kk += 16) {
          for (uint64_t l = 0; l < 16; l++) {
            amx_ldz_f32(l * 4 + 1, a + (ii + l) * k + kk);
          }
          for (uint64_t l = 0; l < 16; l++) {
            amx_extrv_f32(l * 4 + 1);
            amx_ldx_f32(b + (kk + l) * n + jj);
            amx_fma_f32();
          }
        }
        for (uint64_t i = 0; i < 16; i++) {
          amx_stz_f32(i * 4, c + (ii + i) * n + jj);
        }
        amx_clr();
        for (uint64_t l = kk; l < k; l++) {
          for (uint64_t i = ii; i < ii + 16; i++) {
            for (uint64_t j = jj; j < jj + 16; j++) {
              c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
          }
        }
      }
      for (uint64_t i = ii; i < ii + 16; i++) {
        for (uint64_t j = jj; j < n; j++) {
          c[i * n + j] = 0;
          for (uint64_t l = 0; l < k; l++) {
            c[i * n + j] += a[i * k + l] * b[l * n + j];
          }
        }
      }
    }
    for (uint64_t i = ii; i < m; i++) {
      for (uint64_t j = 0; j < n; j++) {
        c[i * n + j] = 0;
        for (uint64_t l = 0; l < k; l++) {
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
    for (uint64_t ii = 0; ii < m; ii += 16) {
      for (uint64_t jj = 0; jj < n; jj += 16) {
        amx_set();
        for (uint64_t l = 0; l < k; l++) {
          amx_ldy_f32(a + l * m + ii);
          amx_ldx_f32(b + l * n + jj);
          amx_fma_f32();
        }
        for (uint64_t i = 0; i < 16; i++) {
          amx_stz_f32(i * 4, c + (ii + i) * n + jj);
        }
        amx_clr();
      }
    }
  } else { // (k, m)^T * (n, k)^T -> (m, n)
    for (uint64_t ii = 0; ii < m; ii += 16) {
      for (uint64_t jj = 0; jj < n; jj += 16) {
        amx_set();
        for (uint64_t kk = 0; kk < k; kk += 16) {
          for (uint64_t l = 0; l < 16; l++) {
            amx_ldz_f32(l * 4 + 1, b + (jj + l) * n + kk);
          }
          for (uint64_t l = 0; l < 16; l++) {
            amx_extrv_f32(l * 4 + 1);
            amx_extrx();
            amx_ldy_f32(a + (kk + l) * m + ii);
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
}
