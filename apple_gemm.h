#pragma once

#include "apple_amx.h"

inline void amx_transpose(float *a, float *c, uint64_t m, uint64_t n)
{
  for (uint64_t i = 0; i < m; i++) {
    for (uint64_t j = 0; j < n; j++) {
      c[j * m + i] = a[i * n + j];
    }
  }
}
