#include "expandsk.cuh"

#include "fips202.cuh"
#include "params.h"
#include "aes128_ctr.cuh"
#include "gf16.cuh"
#include "code.h"

__global__ void gpu_expandsk(uint8_t *d_sk,uint8_t *d_O, uint8_t *d_P1,uint8_t *d_P2,size_t pitch){
    int bid = blockIdx.x;
    d_sk  += bid * pitch;
    d_O +=  bid * pitch;
    d_P1   += bid * pitch;
    d_P2  += bid * pitch;

    __shared__ __align__(16) uint8_t s_S[16 + MAYO_O_BYTES];
    __shared__ __align__(16) uint8_t s_keccak_tmp[SHAKE256_RATE];
    __shared__ uint32_t s_mul[16][4];

    for (int idx = threadIdx.x; idx < 64; idx += 32) {
        int b = idx >> 2;
        int g = idx & 3;
        s_mul[b][g] = GF16_MUL_COL_PACKED[b][g];
    }
    __syncwarp();

    gpu_keccak keccak;
    keccak.shake<SHAKE256_RATE>(s_S, 16 + MAYO_O_BYTES, d_sk, MAYO_SK_BYTES, s_keccak_tmp);

    uint8_t* s_O = s_S + 16;
    aes128_ctr(s_S, d_P1, MAYO_P_BYTES);
    __syncwarp(); 

    for(int i = threadIdx.x; i < MAYO_O_BYTES; i+=32)
        d_O[i] = s_O[i];

    int val = MAYO_M / 2;
    for(int i0 = 0 ;i0 < val; i0 += 32)
    {
        int i = i0 + threadIdx.x;
        if( i >= val) continue;
        for (int r = 0; r < MAYO_M; r++)
        {
            uint8_t tmp_row[MAYO_O];
            #pragma unroll
            for (int b = 0; b < MAYO_O; b++) tmp_row[b] = 0;

            for (int k = r + 1; k < MAYO_M; k++)
            {
                int idxP = idx_mattri(i, r, k, MAYO_M, val);
                uint8_t p = d_P1[idxP];

                #pragma unroll
                for (int b = 0; b < MAYO_O; b++)
                {
                    uint8_t o = GET_matrix(s_O, k * MAYO_O + b);
                    tmp_row[b] ^= gf16_mul2nibble(p, o, s_mul);
                }
            }

            for (int k = 0; k < r; k++)
            {
                int idxP = idx_mattri(i, k, r, MAYO_M, val);
                uint8_t p = d_P1[idxP];

                #pragma unroll
                for (int b = 0; b < MAYO_O; b++)
                {
                    uint8_t o = GET_matrix(s_O, k * MAYO_O + b);
                    tmp_row[b] ^= gf16_mul2nibble(p, o, s_mul);
                }
            }

            #pragma unroll
            for (int b = 0; b < MAYO_O; b++)
            {
                int idxP2 = idx_mat(i, r, b, MAYO_O, val);
                d_P2[idxP2] ^= tmp_row[b];
            }
        }
    }
}