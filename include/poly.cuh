#include "params.h"

__device__ __forceinline__ uint8_t MAYO_POLY_C(int n) {
    return (n==0)? MAYO_POLY_C0 :
           (n==1)? MAYO_POLY_C1 :
           (n==2)? MAYO_POLY_C2 :
                   MAYO_POLY_C3;
}

__device__ __forceinline__
void E1_inplace_packed(uint8_t* s_y,
                       int M_enc,
                       const uint32_t (&s_mul)[16][4])
{
    const unsigned mask = 0xffffffffu;
    int lane = threadIdx.x & 31;

    uint8_t last_byte = 0;
    if (lane == 0) last_byte = s_y[M_enc - 1];
    last_byte = (uint8_t)__shfl_sync(mask, last_byte, 0);
    uint8_t last = (uint8_t)((last_byte >> 4) & 0x0F);

    uint8_t old_local[3];
    uint8_t prevhi_local[3];
    int     idx_local[3];
    int     cnt = 0;

    for (int l = lane; l < M_enc; l += 32) {
        uint8_t old = s_y[l];
        uint8_t prevhi = 0;
        if (l > 0) {
            uint8_t prev = s_y[l - 1]; 
            prevhi = (uint8_t)((prev >> 4) & 0x0F); 
        }
        idx_local[cnt]    = l;
        old_local[cnt]    = old;
        prevhi_local[cnt] = prevhi;
        cnt++;
    }

    __syncwarp(mask);

    for (int t = 0; t < cnt; ++t) {
        int l = idx_local[t];
        uint8_t old = old_local[t];
        uint8_t prevhi = prevhi_local[t];
        uint8_t newb = (uint8_t)(((old & 0x0F) << 4) | (prevhi & 0x0F));
        s_y[l] = newb;
    }

    __syncwarp(mask);

    if (lane == 0) {
        uint8_t c0 = MAYO_POLY_C(0);
        uint8_t c1 = MAYO_POLY_C(1);
        uint8_t e0 = gf16_mul_shared(last, c0, s_mul);
        uint8_t e1 = gf16_mul_shared(last, c1, s_mul);
        s_y[0] ^= (uint8_t)(((e1 & 0x0F) << 4) | (e0 & 0x0F));
    }
    if (lane == 1) {
        uint8_t c2 = MAYO_POLY_C(2);
        uint8_t c3 = MAYO_POLY_C(3);
        uint8_t e2 = gf16_mul_shared(last, c2, s_mul);
        uint8_t e3 = gf16_mul_shared(last, c3, s_mul);
        s_y[1] ^= (uint8_t)(((e3 & 0x0F) << 4) | (e2 & 0x0F));
    }

    __syncwarp(mask);
}
