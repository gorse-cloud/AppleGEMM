#pragma once

#include "aarch64.h"

inline uint64_t amx_operand_field(uint64_t offset, uint64_t width,
                                  uint64_t value) {
  return ((1ULL << width) - 1ULL & value) << offset;
}

inline void amx_set() { AMX_SET(); }

inline void amx_clr() { AMX_CLR(); }

inline void amx_ldx_f32(const float *x) {
  AMX_LDX(amx_operand_field(0, 56, (uint64_t)x));
}

inline void amx_ldx_f32x2(const float *x) {
  AMX_LDX(amx_operand_field(62, 1, 1) | amx_operand_field(0, 56, (uint64_t)x));
}

inline void amx_ldy_f32(const float *y) {
  AMX_LDY(amx_operand_field(0, 56, (uint64_t)y));
}

inline void amx_ldy_f32x2(const float *y) {
  AMX_LDY(amx_operand_field(62, 1, 1) | amx_operand_field(0, 56, (uint64_t)y));
}

inline void amx_ldz_f32(uint64_t index, const float *z) {
  AMX_LDZ(amx_operand_field(56, 6, index) |
          amx_operand_field(0, 56, (uint64_t)z));
}

inline void amx_stx_f32(float *x) {
  AMX_STX(amx_operand_field(0, 56, (uint64_t)x));
}

inline void amx_sty_f32(float *y) {
  AMX_STY(amx_operand_field(0, 56, (uint64_t)y));
}

inline void amx_stz_f32(uint64_t index, float *z) {
  AMX_STZ(amx_operand_field(56, 6, index) |
          amx_operand_field(0, 56, (uint64_t)z));
}

inline void amx_stz_f32x2(uint64_t index, float *z) {
  AMX_STZ(amx_operand_field(62, 1, 1) | amx_operand_field(56, 6, index) |
          amx_operand_field(0, 56, (uint64_t)z));
}

inline void amx_fma_f32(uint64_t z = 0, uint64_t x_offset = 0,
                        uint64_t y_offset = 0) {
  AMX_FMA32(amx_operand_field(20, 6, z) | amx_operand_field(10, 9, x_offset) |
            amx_operand_field(0, 9, y_offset));
}

inline void amx_extrx() {
  AMX_EXTRX(amx_operand_field(27, 1, 1) | amx_operand_field(26, 1, 0));
}

inline void amx_extrv_f32(uint64_t index) {
  AMX_EXTRY(amx_operand_field(28, 2, 1) | amx_operand_field(20, 6, index));
}
