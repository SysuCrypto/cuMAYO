#pragma once

#define ALIGN_TO_128_BYTES(x) ((((x) + 127) / 128) * 128)


__device__ __forceinline__
uint8_t GET_matrix(const uint8_t* packed,int k) { 
    int idx = (k) ^ 1; 
    uint8_t b = packed[idx >> 1];
    return (idx & 1) ? (b & 0xF) : ((b >> 4) & 0xF);
}


__device__ __forceinline__
int idx_mattri(int i, int j, int k, int m, int val){
    int a = k - j;
    a += (m * j);
    a -= (j*(j-1) / 2);
    return i + a * val; 
}

__device__ __forceinline__
int idx_mat(int i, int j, int k, int m, int val){
    int a = m * j;
    a += k;
    return i + a*val; 
}

__device__ __forceinline__
int tri_idx(int a, int b) {
    return a * MAYO_O - (a * (a - 1)) / 2 + (b - a);
}


__device__ __forceinline__
int pair_id(int i, int j) {
    return i * MAYO_K - (i * (i - 1)) / 2 + (j - i);
}

__device__ __forceinline__
void vec_xor4(uint8_t* v_packed, int idx, uint8_t val4)
{
    val4 &= 0x0F;
    int by = idx >> 1;
    uint8_t b = v_packed[by];
    if (idx & 1) b ^= (uint8_t)(val4 << 4);
    else         b ^= val4;
    v_packed[by] = b;
}

__device__ __forceinline__
uint8_t A_get4(const uint8_t* A_row0, int row, int col, int o_bytes)
{
    const int KO_stride = MAYO_K * o_bytes;

    int blk = col / MAYO_O;
    int off = col - blk * MAYO_O;     // 0..O-1
    int by  = blk * o_bytes + (off >> 1);
    uint8_t b = A_row0[row * KO_stride + by];
    return (off & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
}

__device__ __forceinline__
void A_set4(uint8_t* A_row0, int row, int col, int o_bytes, uint8_t v4)
{
    v4 &= 0x0F;
    const int KO_stride = MAYO_K * o_bytes;

    int blk = col / MAYO_O;
    int off = col - blk * MAYO_O;
    int by  = blk * o_bytes + (off >> 1);

    uint8_t* p = &A_row0[row * KO_stride + by];
    uint8_t  b = *p;

    if (off & 1) {
        b = (uint8_t)((b & 0x0F) | (v4 << 4));
    } else {
        b = (uint8_t)((b & 0xF0) | v4);
    }
    *p = b;
}