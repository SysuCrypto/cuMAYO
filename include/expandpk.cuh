#pragma once

#include <cstdint>

__global__ void gpu_expandpk(uint8_t *d_epk,uint8_t *d_pk, uint8_t *cpk, size_t pitch);