// aes128_ctr.cuh
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

using u32 = uint32_t;
using u64 = uint64_t;
using u8  = uint8_t;


extern __device__ __constant__ u32 d_T0_base[256];
extern __device__ __constant__ u8  d_SBOX_base[256];
extern __device__ __constant__ u32 d_RCON32[10];

void aes128_tables_init();

__device__ void aes128_ctr(const u8* key16, u8* out, size_t out_len);

__device__ void aes128_ctr_warp(const u8* key16, u8* out, size_t out_len);