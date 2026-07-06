#include "keypair.cuh"

#include "fips202.cuh"
#include "params.h"
#include "aes128_ctr.cuh"
#include "gf16.cuh"
#include "code.h"


__global__ void gpu_keypair(uint8_t *d_sk,uint8_t *d_pk,uint8_t *d_eP3,uint8_t *d_P,uint8_t *d_P2,size_t pitch,size_t rand_index){
    __shared__ __align__(16) uint8_t s_random_buf[8];
    __shared__ __align__(16) uint8_t s_buf_gens_out[MAYO_SK_BYTES];
    __shared__ __align__(16) uint8_t s_S[16 + MAYO_O_BYTES];
    __shared__ __align__(16) uint8_t s_keccak_tmp[SHAKE128_RATE];

    int bid = blockIdx.x;
    d_sk  += bid * pitch;
    d_pk  += bid * pitch;
    d_eP3 += bid * pitch;
    d_P   += bid * pitch;
    d_P2  += bid * pitch;

    __shared__ uint32_t s_mul[16][4];

    int t = threadIdx.x; 
    for (int idx = t; idx < 64; idx += 32) {
        int b = idx >> 2;
        int g = idx & 3;
        s_mul[b][g] = GF16_MUL_COL_PACKED[b][g];
    }
    __syncwarp();
    uint64_t random_ctr = rand_index + blockIdx.x * 2 + 1;
    if (threadIdx.x < 8)
        s_random_buf[threadIdx.x] = random_ctr >> 8 * threadIdx.x;
    __syncwarp();
    gpu_keccak keccak;
    keccak.shake<SHAKE128_RATE>(s_buf_gens_out, MAYO_SK_BYTES, s_random_buf, 8, s_keccak_tmp);
    __syncwarp();
    if(threadIdx.x < MAYO_SK_BYTES)
        d_sk[threadIdx.x] = s_buf_gens_out[threadIdx.x];
    if(MAYO_SK_BYTES > 32 && threadIdx.x < MAYO_SK_BYTES-32)
        d_sk[threadIdx.x+32] = s_buf_gens_out[threadIdx.x+32];
    __syncwarp();
    keccak.shake<SHAKE256_RATE>(s_S, 16 + MAYO_O_BYTES, d_sk, MAYO_SK_BYTES, s_keccak_tmp);
    uint8_t* s_O = s_S + 16;
    if(threadIdx.x < 16)
        d_pk[threadIdx.x] = s_S[threadIdx.x];
    __syncwarp(); 
    aes128_ctr(s_S, d_P, MAYO_P_BYTES);
    __syncwarp(); 

    for(int i0 = 0; i0 < MAYO_Menc ; i0 += 32)
    {
        int i = i0 + threadIdx.x;
        if (i >= MAYO_Menc){
            continue;
        }

        constexpr int OT = (MAYO_O * (MAYO_O + 1)) / 2;
        uint8_t acc[OT];
        #pragma unroll
        for (int t = 0; t < OT; t++) acc[t] = 0;

        for(int r = 0; r < MAYO_M ;r++)
        {
            uint8_t B_row[MAYO_O];

            #pragma unroll
            for (int b = 0; b < MAYO_O; b++) B_row[b] = 0;

            for (int k = r; k < MAYO_M; k++)
            {
                int idxP = idx_mattri(i, r, k, MAYO_M, MAYO_Menc);
                uint8_t p = d_P[idxP];

                #pragma unroll
                for (int b = 0; b < MAYO_O; b++)
                {
                    uint8_t o = GET_matrix(s_O, k * MAYO_O + b);
                    B_row[b] ^= gf16_mul2nibble(p, o, s_mul);
                }
            }

            #pragma unroll
            for (int b = 0; b < MAYO_O; b++)
            {
                int idx2 = idx_mat(i, r, b, MAYO_O, MAYO_Menc);
                B_row[b] ^= d_P2[idx2];
            }


            #pragma unroll   
            for(int a = 0; a < MAYO_O; a++)
            {
                uint8_t o = GET_matrix(s_O,r*MAYO_O+a);
                #pragma unroll
                for(int b = a; b < MAYO_O; b++)
                {
                    uint8_t tmp = gf16_mul2nibble(B_row[b],o,s_mul);
                    if(a != b)
                    {
                        uint8_t ob = GET_matrix(s_O, r * MAYO_O + b);
                        tmp ^= gf16_mul2nibble(B_row[a],ob,s_mul);
                    }
                    acc[tri_idx(a,b)] ^= tmp;           
                }
            }
        }
        #pragma unroll
        for (int a = 0; a < MAYO_O; a++) {
            #pragma unroll
            for (int b = a; b < MAYO_O; b++) {
                int idx = idx_mattri(i, a, b, MAYO_O, MAYO_Menc);
                d_eP3[idx] = acc[tri_idx(a,b)];
            }
        }
    }
}