#include "verify.cuh"

#include "fips202.cuh"
#include "params.h"
#include "gf16.cuh"
#include "code.h"
#include "poly.cuh"

__global__ void gpu_verify(uint8_t *d_P1,uint8_t *d_P2,uint8_t *d_P3, uint8_t *d_M,uint8_t *d_sig,uint8_t *d_verify,size_t Mlen,size_t pitch){

    int bid = blockIdx.x;
    d_P1 += bid * pitch;
    d_P2 += bid * pitch;
    d_P3 += bid * pitch;
    d_M += bid * pitch;
    d_sig += bid * pitch;
    d_verify += bid * pitch;

    __shared__ uint32_t s_mul[16][4];
    __shared__ uint8_t s_keccak_tmp[SHAKE256_RATE];

    int t = threadIdx.x;
    for (int idx = t; idx < 64; idx += 32) {
        int b = idx >> 2;
        int g = idx & 3;
        s_mul[b][g] = GF16_MUL_COL_PACKED[b][g];
    }
    __syncwarp();

    __shared__ uint8_t s_key[MAYO_DIGEST_BYTES + MAYO_SK_BYTES];
    __shared__ uint8_t s_t[MAYO_Menc];
    __shared__ uint8_t s_y[MAYO_Menc];
    __shared__ uint8_t s_s[MAYO_K*MAYO_N/2];
    uint8_t *s_salt = s_key + MAYO_DIGEST_BYTES;
    uint8_t *d_salt = d_sig + MAYO_K*MAYO_N/2;

    for(int i = threadIdx.x; i < MAYO_SK_BYTES; i += blockDim.x)
        s_salt[i] = d_salt[i];
    __syncwarp();

    for(int i = threadIdx.x; i < MAYO_K*MAYO_N/2; i += blockDim.x)
        s_s[i] = d_sig[i];
    __syncwarp();

    gpu_keccak keccak;
    keccak.shake<SHAKE256_RATE>(s_key, MAYO_DIGEST_BYTES, d_M, Mlen, s_keccak_tmp);
    
    keccak.shake<SHAKE256_RATE>(s_t, MAYO_Menc, s_key, MAYO_DIGEST_BYTES + MAYO_SK_BYTES, s_keccak_tmp);

    for (int l = threadIdx.x; l < MAYO_Menc; l += 32) {
        s_y[l] = 0;
    }
    __syncwarp();
    const unsigned mask = 0xffffffffu;
   
    __shared__ uint8_t V4[MAYO_K][MAYO_M];
    __shared__ uint8_t X4[MAYO_K][MAYO_M];

    constexpr int P_PAD = MAYO_Pair + 1;
    __shared__ uint8_t U[MAYO_Menc][P_PAD];

    for (int i = 0; i < MAYO_K; ++i) {
        for (int r = threadIdx.x; r < MAYO_M; r += 32) {
            V4[i][r] = GET_matrix(s_s, i * MAYO_N + r) & 0x0F;
        }
        for (int k = threadIdx.x; k < MAYO_O; k += 32) {
            X4[i][k] = GET_matrix(s_s, i * MAYO_N + MAYO_M + k) & 0x0F;
        }
    }
    __syncwarp(mask);

    for (int l = threadIdx.x; l < MAYO_Menc; l += 32)
    {
        for (int pid = 0; pid < MAYO_Pair; ++pid) {
            U[l][pid] = 0;
        }

        for (int r = 0; r < MAYO_M; ++r)
        {
            uint8_t T[MAYO_K];
            #pragma unroll
            for (int i = 0; i < MAYO_K; ++i) T[i] = 0;

            #pragma unroll 1
            for (int c = r; c < MAYO_M; ++c)
            {
                const int idx = idx_mattri(l, r, c, MAYO_M, MAYO_Menc);
                const uint8_t P1v = d_P1[idx];

                #pragma unroll
                for (int i = 0; i < MAYO_K; ++i) {
                    T[i] ^= gf16_mul2nibble(P1v, V4[i][c], s_mul);
                }
            }

            #pragma unroll
            for (int i = 0; i < MAYO_K; ++i)
            {
                const uint8_t vi_r = V4[i][r];
                #pragma unroll
                for (int j = i; j < MAYO_K; ++j)
                {
                    const int pid = pair_id(i, j);
                    U[l][pid] ^= gf16_mul2nibble(T[j], vi_r, s_mul);
                    if (i != j) {
                        const uint8_t vj_r = V4[j][r];
                        U[l][pid] ^= gf16_mul2nibble(T[i], vj_r, s_mul);
                    }
                }
            }
        }

        for (int r = 0; r < MAYO_M; ++r)
        {
            uint8_t T[MAYO_K];
            #pragma unroll
            for (int i = 0; i < MAYO_K; ++i) T[i] = 0;

            #pragma unroll 1
            for (int k = 0; k < MAYO_O; ++k)
            {
                const int idx = idx_mat(l, r, k, MAYO_O, MAYO_Menc);
                const uint8_t P2v = d_P2[idx];

                #pragma unroll
                for (int i = 0; i < MAYO_K; ++i) {
                    T[i] ^= gf16_mul2nibble(P2v, X4[i][k], s_mul);
                }
            }

            #pragma unroll
            for (int i = 0; i < MAYO_K; ++i)
            {
                const uint8_t vi_r = V4[i][r];
                #pragma unroll
                for (int j = i; j < MAYO_K; ++j)
                {
                    const int pid = pair_id(i, j);
                    U[l][pid] ^= gf16_mul2nibble(T[j], vi_r, s_mul);
                    if (i != j) {
                        const uint8_t vj_r = V4[j][r];
                        U[l][pid] ^= gf16_mul2nibble(T[i], vj_r, s_mul);
                    }
                }
            }
        }

        for (int r = 0; r < MAYO_O; ++r)
        {
            uint8_t T[MAYO_K];
            #pragma unroll
            for (int i = 0; i < MAYO_K; ++i) T[i] = 0;

            #pragma unroll 1
            for (int c = r; c < MAYO_O; ++c)
            {
                const int idx = idx_mattri(l, r, c, MAYO_O, MAYO_Menc);
                const uint8_t P3v = d_P3[idx];

                #pragma unroll
                for (int i = 0; i < MAYO_K; ++i) {
                    T[i] ^= gf16_mul2nibble(P3v, X4[i][c], s_mul);
                }
            }

            #pragma unroll
            for (int i = 0; i < MAYO_K; ++i)
            {
                const uint8_t xi_r = X4[i][r];
                #pragma unroll
                for (int j = i; j < MAYO_K; ++j)
                {
                    const int pid = pair_id(i, j);
                    U[l][pid] ^= gf16_mul2nibble(T[j], xi_r, s_mul);
                    if (i != j) {
                        const uint8_t xj_r = X4[j][r];
                        U[l][pid] ^= gf16_mul2nibble(T[i], xj_r, s_mul);
                    }
                }
            }
        }
    }
    __syncwarp(mask);

    for (int l = threadIdx.x; l < MAYO_Menc; l += 32) s_y[l] = 0;
    __syncwarp(mask);

    for (int i = MAYO_K - 1; i >= 0; --i)
    {
        for (int j = i; j < MAYO_K; ++j)
        {
            E1_inplace_packed(s_y, MAYO_Menc, s_mul);

            const int pid = pair_id(i, j);
            for (int l = threadIdx.x; l < MAYO_Menc; l += 32) {
                s_y[l] ^= U[l][pid];
            }
            __syncwarp(mask);
        }
    }


    int eq = 1;
    for (int l = threadIdx.x; l < MAYO_Menc; l += 32) {
        if (s_y[l] != s_t[l]) eq = 0;
    }
    eq = __all_sync(mask, eq);

    if ((threadIdx.x & 31) == 0) {
        d_verify[0] = (uint8_t)(eq ? 0 : 1);
    }
}