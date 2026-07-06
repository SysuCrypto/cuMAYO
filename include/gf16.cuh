#pragma once
#include <stdint.h>

extern __constant__ uint8_t GF16_PAIR_MUL[4096];
extern __constant__ uint32_t GF16_MUL_COL_PACKED[16][4];
extern __constant__ uint8_t d_gf16_inv[16];

__device__ uint8_t gf16_add(uint8_t a, uint8_t b);

__device__ __forceinline__
uint8_t gf16_pair_mul(uint8_t packed_xy, uint8_t y) {
    uint32_t idx = (packed_xy << 4) | (y & 0xF);
    return GF16_PAIR_MUL[idx];
}

__device__ __forceinline__ uint8_t get_byte(uint32_t w, int idx) {
    return (uint8_t)((w >> (idx * 8)) & 0xFFu);
}

__device__ __forceinline__
uint8_t gf16_mul_shared(uint8_t a4, uint8_t b4,
                               const uint32_t (&tab)[16][4]) {
    a4 &= 0x0F;
    b4 &= 0x0F;
    uint32_t w = tab[b4][a4 >> 2];
    return (uint8_t)((w >> ((a4 & 3) * 8)) & 0xFFu);
}

__device__ __forceinline__
uint8_t gf16_mul2nibble(uint8_t a, uint8_t b4,
                        const uint32_t (&tab)[16][4]) {
    b4 &= 0x0F;

    uint8_t ahi = (uint8_t)(a >> 4);
    uint8_t alo = (uint8_t)(a & 0x0F);

    uint8_t rhi = gf16_mul_shared(ahi, b4, tab);
    uint8_t rlo = gf16_mul_shared(alo, b4, tab);

    return (uint8_t)((rhi << 4) | (rlo & 0x0F));
}