#pragma once

#include <cstdint>

__global__ void gpu_keypair(uint8_t *d_sk,uint8_t *d_pk,uint8_t *d_eP3,uint8_t *d_P,uint8_t *d_P2,size_t keypair_mem_pool_pitch,size_t rand_index = 0);