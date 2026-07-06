#pragma once

#include <cstdint>
#include "params.h"

__global__ void gpu_sign(uint8_t *d_esk,uint8_t *d_Mess,size_t Mlen,uint8_t *d_P1,uint8_t *d_L,uint8_t *d_O,uint8_t *d_sig,uint8_t *d_M, size_t pitch);