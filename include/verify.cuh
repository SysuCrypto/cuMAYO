#pragma once

#include <cstdint>
#include "params.h"

__global__ void gpu_verify(uint8_t *d_P1,uint8_t *d_P2,uint8_t *d_P3, uint8_t *d_M,uint8_t *d_sig,uint8_t *d_verify,size_t Mlen,size_t pitch);