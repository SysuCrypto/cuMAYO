#pragma once

#include <cstdint>

__global__ void gpu_expandsk(uint8_t *d_sk,uint8_t *d_O,uint8_t *d_P1,uint8_t *d_P2,size_t pitch);