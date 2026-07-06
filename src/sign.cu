#include "sign.cuh"

#include "fips202.cuh"
#include "params.h"
#include "aes128_ctr.cuh"
#include "gf16.cuh"
#include "code.h"
#include "poly.cuh"

__device__ __forceinline__
int sample_solution(uint8_t* A_row0,
                          const uint8_t* s_y_packed,
                          const uint8_t* s_r_packed,
                          uint8_t* s_x_packed,
                          const uint32_t (&s_mul)[16][4])
{
    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;

    const int m = MAYO_M;
    const int KO = MAYO_K * MAYO_O;
    const int KO_BYTES = (KO + 1) / 2;
    const int o_bytes  = (MAYO_O + 1) / 2;
    const int ROW_BYTES = MAYO_K * o_bytes;

    __shared__ uint8_t s_y4[MAYO_M];
    for (int r = lane; r < m; r += 32) {
        s_y4[r] = GET_matrix(s_y_packed, r);
    }

    for (int b = lane; b < KO_BYTES; b += 32) {
        s_x_packed[b] = s_r_packed[b];
    }
    __syncwarp(mask);

    for (int r = lane; r < m; r += 32) {
        uint8_t acc = 0;
        #pragma unroll 1
        for (int c = 0; c < KO; ++c) {
            uint8_t a = A_get4(A_row0, r, c, o_bytes);
            uint8_t rv= GET_matrix(s_r_packed, c);
            acc ^= (gf16_mul_shared(a, rv, s_mul) & 0x0F);
        }
        s_y4[r] ^= acc;
    }
    __syncwarp(mask);

    int piv_row = 0;
    for (int piv_col = 0; piv_col < KO && piv_row < m; ++piv_col)
    {
        int p = -1;
        if (lane == 0) {
            for (int rr = piv_row; rr < m; ++rr) {
                if (A_get4(A_row0, rr, piv_col, o_bytes) != 0) { p = rr; break; }
            }
        }
        p = __shfl_sync(mask, p, 0);
        if (p < 0) continue;

        if (p != piv_row) {
            if (lane == 0) {
                uint8_t ty = s_y4[piv_row];
                s_y4[piv_row] = s_y4[p];
                s_y4[p] = ty;
            }

            for (int by = lane; by < ROW_BYTES; by += 32) {
                uint8_t* rp = &A_row0[piv_row * ROW_BYTES + by];
                uint8_t* rq = &A_row0[p * ROW_BYTES + by];
                uint8_t t  = *rp;
                *rp = *rq;
                *rq = t;
            }
            __syncwarp(mask);
        }

        uint8_t piv = A_get4(A_row0, piv_row, piv_col, o_bytes) & 0x0F;
        uint8_t inv = d_gf16_inv[piv];
        if (lane == 0) {
            s_y4[piv_row] = gf16_mul_shared(s_y4[piv_row], inv, s_mul) & 0x0F;
        }
        for (int by = lane; by < ROW_BYTES; by += 32) {
            uint8_t* rp = &A_row0[piv_row * ROW_BYTES + by];
            *rp = gf16_mul2nibble(*rp, inv, s_mul);
        }
        __syncwarp(mask);

        for (int rr = piv_row + 1; rr < m; ++rr) {
            uint8_t f = A_get4(A_row0, rr, piv_col, o_bytes) & 0x0F;
            if (f == 0) continue;

            if (lane == 0) {
                uint8_t t = gf16_mul_shared(s_y4[piv_row], f, s_mul) & 0x0F;
                s_y4[rr] ^= t;
            }
            for (int by = lane; by < ROW_BYTES; by += 32) {
                uint8_t* rrp = &A_row0[rr * ROW_BYTES + by];
                uint8_t  pv  = A_row0[piv_row * ROW_BYTES + by];
                *rrp ^= gf16_mul2nibble(pv, f, s_mul);
            }
            __syncwarp(mask);
        }

        piv_row++;
    }

    int last_zero = 1;
    if (lane == 0) {
        last_zero = 1;
        for (int c = 0; c < KO; ++c) {
            if (A_get4(A_row0, m - 1, c, o_bytes) != 0) { last_zero = 0; break; }
        }
    }
    last_zero = __shfl_sync(mask, last_zero, 0);
    if (last_zero) return 0;

    for (int rr = m - 1; rr >= 0; --rr)
    {
        int c = -1;
        if (lane == 0) {
            for (int cc = 0; cc < KO; ++cc) {
                if (A_get4(A_row0, rr, cc, o_bytes) != 0) { c = cc; break; }
            }
        }
        c = __shfl_sync(mask, c, 0);
        if (c < 0) continue;

        uint8_t alpha = s_y4[rr] & 0x0F;
        if (lane == 0 && alpha != 0) {
            vec_xor4(s_x_packed, c, alpha);
        }
        __syncwarp(mask);

        if (alpha != 0) {
            for (int r = lane; r < m; r += 32) {
                uint8_t a = A_get4(A_row0, r, c, o_bytes) & 0x0F;
                uint8_t t = gf16_mul_shared(alpha, a, s_mul) & 0x0F;
                s_y4[r] ^= t;
            }
        }
        __syncwarp(mask);
    }

    return 1;
}

__device__ __forceinline__
uint8_t compute_s_elem(int e, const uint8_t* s_V, int M_enc, const uint8_t* d_O, const uint8_t* s_x, const uint32_t (&s_mul)[16][4])
{
    const int n = MAYO_N;
    const int m = MAYO_M;
    const int o = MAYO_O;

    int i = e / n;
    int t = e - i * n;

    if (t < m) {
        uint8_t acc = 0;
        #pragma unroll
        for (int k = 0; k < MAYO_O; ++k) {
            uint8_t o4 = GET_matrix(d_O, t * MAYO_O + k) & 0x0F;
            uint8_t x4 = GET_matrix(s_x, i * MAYO_O + k) & 0x0F;
            acc ^= (gf16_mul_shared(o4, x4, s_mul) & 0x0F);
        }
        uint8_t v4 = GET_matrix(s_V + i * M_enc, t) & 0x0F;
        return (uint8_t)((v4 ^ acc) & 0x0F);
    } else {
        int k = t - m;
        if (k < o) return (uint8_t)(GET_matrix(s_x, i * MAYO_O + k) & 0x0F);
        return 0;
    }
}
__device__ __forceinline__
void s_packed(uint8_t* d_sig, const uint8_t* s_V, int M_enc, const uint8_t* d_O, const uint8_t* s_x, const uint32_t (&s_mul)[16][4])
{
    const int lane = threadIdx.x & 31;
    const int total_elems  = MAYO_K * MAYO_N;
    const int total_bytes  = (total_elems + 1) / 2;

    for (int b = lane; b < total_bytes; b += 32)
    {
        int e0 = (b << 1);
        int e1 = e0 + 1;

        uint8_t v0 = 0, v1 = 0;
        if (e0 < total_elems) v0 = compute_s_elem(e0, s_V, M_enc, d_O, s_x, s_mul);
        if (e1 < total_elems) v1 = compute_s_elem(e1, s_V, M_enc, d_O, s_x, s_mul);

        d_sig[b] = (uint8_t)(((v1 & 0x0F) << 4) | (v0 & 0x0F));
    }
}


__global__ void gpu_sign(uint8_t *d_esk,uint8_t *d_Mess,size_t Mlen,uint8_t *d_P1,uint8_t *d_L,uint8_t *d_O, uint8_t *d_sig, uint8_t *d_M, size_t pitch){
    const int keylen = MAYO_DIGEST_BYTES+MAYO_SK_BYTES+MAYO_SK_BYTES;
    __shared__ uint8_t s_key[keylen+1];
    __shared__ uint8_t s_salt[MAYO_SK_BYTES];
    __shared__ uint8_t s_keccak_tmp[SHAKE256_RATE];
    __shared__ uint8_t s_t[MAYO_Menc];
    __shared__ uint8_t s_y[MAYO_Menc];
    __shared__ uint8_t s_V[MAYO_N*MAYO_K/2];
    __shared__ uint8_t s_A[MAYO_M][MAYO_K][(MAYO_O+1)/2];
    __shared__ uint8_t s_x[MAYO_K * MAYO_O/2]; 


    __shared__ uint32_t s_mul[16][4];

    int t = threadIdx.x;
    for (int idx = t; idx < 64; idx += 32) {
        int b = idx >> 2;
        int g = idx & 3;
        s_mul[b][g] = GF16_MUL_COL_PACKED[b][g];
    }

    int bid = blockIdx.x;
    d_esk  += bid * pitch;
    d_Mess += bid * pitch;
    d_P1 += bid * pitch;
    d_L += bid * pitch;
    d_O += bid * pitch;
    d_sig += bid * pitch;
    d_M += bid * pitch;
   
    gpu_keccak keccak;
    keccak.shake<SHAKE256_RATE>(s_key, MAYO_DIGEST_BYTES, d_Mess, Mlen, s_keccak_tmp);
    uint8_t* s_R = s_key + MAYO_DIGEST_BYTES;
    uint8_t* s_seedsk = s_R + MAYO_SK_BYTES;
    uint8_t* s_ctr = s_seedsk + MAYO_SK_BYTES;

    for(int i = threadIdx.x; i < MAYO_SK_BYTES; i+=blockDim.x)
        s_R[i] = 0;

    for(int i = threadIdx.x; i < MAYO_SK_BYTES; i+=blockDim.x)
        s_seedsk[i] = d_esk[i];

    __syncwarp();

    keccak.shake<SHAKE256_RATE>(s_salt, MAYO_SK_BYTES, s_key, keylen, s_keccak_tmp);

    for(int i = threadIdx.x; i < MAYO_SK_BYTES; i+=blockDim.x)
        s_R[i] = s_salt[i];

    keccak.shake<SHAKE256_RATE>(s_t, MAYO_Menc, s_key, MAYO_DIGEST_BYTES+MAYO_SK_BYTES, s_keccak_tmp);
    for(int ctr = 0;ctr < 255; ctr++)
    {
        for (int idx = threadIdx.x; idx < MAYO_K * MAYO_M * ((MAYO_O + 1) / 2); idx += blockDim.x) {
            reinterpret_cast<uint8_t*>(s_A)[idx] = 0;
        }
        __syncwarp();
        uint8_t* s_r = s_V + MAYO_K * MAYO_Menc;
        s_ctr[0] = (uint8_t) ctr;
        keccak.shake<SHAKE256_RATE>(s_V,MAYO_N*MAYO_K/2, s_key, keylen+1, s_keccak_tmp);

       const int o_bytes = (MAYO_O + 1) / 2;

        for (int j = threadIdx.x; j < MAYO_Menc; j += 32)
        {
            const int row0 = 2 * j;
            const int row1 = row0 + 1;

            for (int kbyte = 0; kbyte < o_bytes; ++kbyte)
            {
                const int k0 = (kbyte << 1);
                const int k1 = k0 + 1;

                uint8_t tmp1[MAYO_K],tmp2[MAYO_K];

                #pragma unroll
                for (int ii = 0; ii < MAYO_K; ++ii) {
                    tmp1[ii] = 0;
                    tmp2[ii] = 0;
                }

                for (int l = 0; l < MAYO_Menc; ++l)
                {
                    const int low0  = idx_mat(j, 2*l,   k0, MAYO_O, MAYO_Menc);
                    const int high0 = idx_mat(j, 2*l+1, k0, MAYO_O, MAYO_Menc);
                    const uint8_t L0_low  = d_L[low0];
                    const uint8_t L0_high = d_L[high0];

                    const uint8_t L0_low_low   =  L0_low       & 0x0F;
                    const uint8_t L0_low_high  = (L0_low >> 4) & 0x0F;
                    const uint8_t L0_high_low  =  L0_high      & 0x0F;
                    const uint8_t L0_high_high = (L0_high>> 4) & 0x0F;

                    uint8_t L1_low_low=0, L1_low_high=0, L1_high_low=0, L1_high_high=0;
                    if (k1 < MAYO_O)
                    {
                        const int low1  = idx_mat(j, 2*l,   k1, MAYO_O, MAYO_Menc);
                        const int high1 = idx_mat(j, 2*l+1, k1, MAYO_O, MAYO_Menc);
                        const uint8_t L1_low  = d_L[low1];
                        const uint8_t L1_high = d_L[high1];

                        L1_low_low   =  L1_low       & 0x0F;
                        L1_low_high  = (L1_low >> 4) & 0x0F;
                        L1_high_low  =  L1_high      & 0x0F;
                        L1_high_high = (L1_high>> 4) & 0x0F;
                    }

                    #pragma unroll
                    for (int ii = 0; ii < MAYO_K; ++ii)
                    {
                        const uint8_t v      = s_V[ii * MAYO_Menc + l];
                        const uint8_t v_high = (v >> 4) & 0x0F;
                        const uint8_t v_low  =  v       & 0x0F;

                        tmp1[ii] ^= gf16_mul_shared(v_high, L0_high_low,  s_mul);
                        tmp1[ii] ^= gf16_mul_shared(v_low,  L0_low_low,   s_mul);
                        tmp2[ii] ^= gf16_mul_shared(v_high, L0_high_high, s_mul);
                        tmp2[ii] ^= gf16_mul_shared(v_low,  L0_low_high,  s_mul);  

                        if (k1 < MAYO_O) {
                            tmp1[ii] ^= (gf16_mul_shared(v_high, L1_high_low,  s_mul)<<4);
                            tmp1[ii] ^= (gf16_mul_shared(v_low,  L1_low_low,   s_mul)<<4);
                            tmp2[ii] ^= (gf16_mul_shared(v_high, L1_high_high, s_mul)<<4);
                            tmp2[ii] ^= (gf16_mul_shared(v_low,  L1_low_high,  s_mul)<<4);
                        }
                    }
                }

                #pragma unroll
                for (int ii = 0; ii < MAYO_K; ++ii)
                {
                    const int base0 = (ii * MAYO_M + row0) * o_bytes;
                    const int base1 = (ii * MAYO_M + row1) * o_bytes;
                    d_M[base0 + kbyte] = tmp1[ii];
                    d_M[base1 + kbyte] = tmp2[ii];
                }
            }
        }


        int ell = 0;
        const int lane    = threadIdx.x;

        for (int i = 0; i < MAYO_K; ++i)
        {
            for (int j = MAYO_K - 1; j >= i; --j)  
            {
                int deback = MAYO_M - ell;

                for (int k = lane; k < deback; k += blockDim.x)
                {
                    int r = k + ell;
                    #pragma unroll
                    for (int m = 0; m < o_bytes; ++m)
                    {
                        s_A[r][i][m] ^= d_M[(j * MAYO_M + k) * o_bytes + m];
                        if (i != j) {
                            s_A[r][j][m] ^= d_M[(i * MAYO_M + k) * o_bytes + m];
                        }
                    }
                }

                if (ell > 0)
                {
                    int r_end = ell + 3;              

                    for (int k = lane; k < r_end; k += blockDim.x)
                    {
                        int r = k;
                        #pragma unroll
                        for (int m = 0; m < o_bytes; ++m)
                        {
                            uint8_t acc_i = 0;
                            uint8_t acc_j = 0;

                            #pragma unroll
                            for (int n = 0; n < 4; ++n)
                            {
                                int t = r - n;
                                if ((unsigned)t < (unsigned)ell)
                                {
                                    int src = (MAYO_M - ell) + t;
                                    acc_i ^= gf16_mul2nibble(d_M[(j * MAYO_M + src) * o_bytes + m], MAYO_POLY_C(n), s_mul);
                                    if (i != j) {
                                        acc_j ^= gf16_mul2nibble(d_M[(i * MAYO_M + src) * o_bytes + m], MAYO_POLY_C(n), s_mul);
                                    }
                                }
                            }

                            s_A[r][i][m] ^= acc_i;
                            if (i != j) {
                                s_A[r][j][m] ^= acc_j;
                            }
                        }
                    }
                }

            ++ell;
            __syncwarp(); 
            }
        }
        
        const unsigned mask = 0xffffffffu;
        __shared__ uint8_t s_U[MAYO_Menc][MAYO_Pair];

        for (int l = threadIdx.x; l < MAYO_Menc; l += 32)
        {
            for (int p = 0; p < MAYO_Pair; ++p) s_U[l][p] = 0;

            for (int m = 0; m < MAYO_M; ++m)
            {
                uint8_t T[MAYO_K];
                #pragma unroll
                for (int i = 0; i < MAYO_K; ++i) T[i] = 0;

                #pragma unroll 1
                for (int n = m; n < MAYO_M; ++n)
                {
                    const int idx = idx_mattri(l, m, n, MAYO_M, MAYO_Menc);
                    const uint8_t P1 = d_P1[idx]; 

                    #pragma unroll
                    for (int i = 0; i < MAYO_K; ++i) {
                        T[i] ^= gf16_mul2nibble(P1, GET_matrix(&s_V[i * MAYO_Menc], n), s_mul);
                    }
                }

                #pragma unroll
                for (int i = 0; i < MAYO_K; ++i)
                {
                    const uint8_t vi_m = GET_matrix(&s_V[i * MAYO_Menc], m);
                    #pragma unroll
                    for (int j = i; j < MAYO_K; ++j)
                    {
                        const int pid = pair_id(i, j);
                        s_U[l][pid] ^= gf16_mul2nibble(T[j], vi_m, s_mul);

                        if (i != j) {
                            const uint8_t vj_m = GET_matrix(&s_V[j * MAYO_Menc], m);
                            s_U[l][pid] ^= gf16_mul2nibble(T[i], vj_m, s_mul);
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
                    s_y[l] ^= s_U[l][pid];
                }
                __syncwarp(mask);
            }
        }

        for (int l = threadIdx.x; l < MAYO_Menc; l += 32) {
            s_y[l] ^= s_t[l];
        }
        __syncwarp(mask);



        int ok = 0;
        ok = sample_solution((uint8_t*)&s_A[0][0][0],s_y,s_r,s_x,s_mul);
        
        ok = __shfl_sync(0xffffffffu, ok, 0);
        if(ok){
        break;}
    }

    __syncwarp();

    s_packed(d_sig,s_V,MAYO_Menc,d_O,s_x,s_mul);
    uint8_t *d_salt = d_sig + (MAYO_N*MAYO_K)/2;
    for(int i=threadIdx.x;i<MAYO_SK_BYTES;i+=32)
        d_salt[i] = s_salt[i];
}