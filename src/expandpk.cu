#include "expandpk.cuh"

#include "fips202.cuh"
#include "params.h"
#include "aes128_ctr.cuh"
#include "gf16.cuh"
#include "code.h"

__global__ void gpu_expandpk(uint8_t *d_epk,uint8_t *d_pk, uint8_t *d_cpk, size_t pitch){
    __shared__ __align__(16) uint8_t s_pk[16];

    int bid = blockIdx.x;
    d_pk  += bid * pitch;
    d_epk += bid * pitch;
    d_cpk += bid * pitch;

    if(threadIdx.x < 16)
    {   
        s_pk[threadIdx.x] = d_pk[threadIdx.x];
    }
    __syncwarp(); 
    aes128_ctr_warp(s_pk, d_epk, MAYO_P_BYTES);
    __syncwarp(); 
}