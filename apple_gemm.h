#pragma once

#include "apple_amx.h"

inline void apple_mm(const float *a, const float *b, float *c, uint64_t m,
                     uint64_t n, uint64_t k, bool transA, bool transB) {
  if (!transA && !transB) { // (m, k) * (k, n) -> (m, n)
    for (uint64_t ii = 0; ii < m; ii += 16) {
      for (uint64_t jj = 0; jj < n; jj += 32) {
        amx_set();
        for (uint64_t kk = 0; kk < k; kk += 16) {
          for (uint64_t l = 0; l < 16; l++) {
            amx_ldz_f32(l * 4 + 2, a + (ii + l) * k + kk);
          }
          for (uint64_t l = 0; l < 16; l++) {
            amx_extrv_f32(l * 4 + 2);
            amx_ldx_f32x2(b + (kk + l) * n + jj);
            amx_fma_f32(0, 0, 0);
            amx_fma_f32(1, 64, 0);
          }
        }
        for (uint64_t i = 0; i < 16; i++) {
          amx_stz_f32x2(i * 4, c + (ii + i) * n + jj);
        }
        amx_clr();
      }
    }
    // 16 x 16 version
    // for (uint64_t ii = 0; ii < m; ii += 16) {
    //   for (uint64_t jj = 0; jj < n; jj += 16) {
    //     amx_set();
    //     for (uint64_t kk = 0; kk < k; kk += 16) {
    //       for (uint64_t l = 0; l < 16; l++) {
    //         amx_ldz_f32(l * 4 + 1, a + (ii + l) * k + kk);
    //       }
    //       for (uint64_t l = 0; l < 16; l++) {
    //         amx_extrv_f32(l * 4 + 1);
    //         amx_ldx_f32(b + (kk + l) * n + jj);
    //         amx_fma_f32();
    //       }
    //     }
    //     for (uint64_t i = 0; i < 16; i++) {
    //       amx_stz_f32(i * 4, c + (ii + i) * n + jj);
    //     }
    //     amx_clr();
    //   }
    // }
  } else if (!transA && transB) {
    for (uint64_t ii = 0; ii < m; ii += 16) {
      for (uint64_t jj = 0; jj < n; jj += 16) {
        amx_set();
        for (uint64_t kk = 0; kk < k; kk += 16) {
          for (uint64_t l = 0; l < 16; l++) {
            amx_ldz_f32(l * 4 + 1, a + (ii + l) * k + kk);
            amx_ldz_f32(l * 4 + 2, b + (jj + l) * k + kk);
          }
          for (uint64_t l = 0; l < 16; l++) {
            amx_extrv_f32(l * 4 + 2);
            amx_extrx();
            amx_extrv_f32(l * 4 + 1);
            amx_fma_f32();
          }
        }
        for (uint64_t l = 0; l < 16; l++) {
          amx_stz_f32(l * 4, c + (ii + l) * n + jj);
        }
        amx_clr();
      }
    }
  } else if (transA && !transB) { // (k, m)^T * (k, n) -> (m, n)
    for (uint64_t ii = 0; ii < m; ii += 32) {
      for (uint64_t jj = 0; jj < n; jj += 32) {
        amx_set();
        for (uint64_t l = 0; l < k; l++) {
          amx_ldy_f32x2(a + l * m + ii);
          amx_ldx_f32x2(b + l * n + jj);
          amx_fma_f32(0, 0, 0);
          amx_fma_f32(1, 64, 0);
          amx_fma_f32(2, 0, 64);
          amx_fma_f32(3, 64, 64);
        }
        for (uint64_t i = 0; i < 16; i++) {
          amx_stz_f32x2(i * 4, c + (ii + i) * n + jj);
          amx_stz_f32x2(i * 4 + 2, c + (ii + 16 + i) * n + jj);
        }
        amx_clr();
      }
    }
    // 16 x 16 version
    // for (uint64_t ii = 0; ii < m; ii += 16) {
    //   for (uint64_t jj = 0; jj < n; jj += 16) {
    //     amx_set();
    //     for (uint64_t l = 0; l < k; l++) {
    //       amx_ldy_f32(a + l * m + ii);
    //       amx_ldx_f32(b + l * n + jj);
    //       amx_fma_f32();
    //     }
    //     for (uint64_t i = 0; i < 16; i++) {
    //       amx_stz_f32(i * 4, c + (ii + i) * n + jj);
    //     }
    //     amx_clr();
    //   }
    // }
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
