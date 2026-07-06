#pragma once

#include <cstddef>
#include <cstdint>

#include "params.h"


void crypto_sign_keypair(uint8_t *d_keypair_mem_pool, size_t keypair_mem_pool_pitch, size_t batch_size = 1, cudaStream_t stream = nullptr, size_t rand_index = 0);

void crypto_sign_keypair_DtH(uint8_t *pk, uint8_t *sk, uint8_t *d_keypair_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_expandsk_HtD(uint8_t *csk, uint8_t *d_expandsk_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_expandsk(uint8_t *d_expandsk_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_expandsk_DtH(uint8_t *esk ,uint8_t *d_expandsk_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_expandpk_HtD(uint8_t *cpk, uint8_t *d_expandpk_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_expandpk(uint8_t *d_expandpk_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_expandpk_DtH(uint8_t *epk ,uint8_t *d_expandpk_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_sign_HtD(uint8_t *cesk,uint8_t *cMess,size_t mlen,uint8_t *d_sign_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_sign(uint8_t *d_sign_mem_pool,size_t Mlen, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_sign_DtH(uint8_t *sig ,uint8_t *d_sig_mem_pool,size_t Mlen,size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_signWithesk_HtD(uint8_t *cesk,uint8_t *cMess,size_t mlen,uint8_t *d_sign_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_signWithesk(uint8_t *d_sign_mem_pool,size_t Mlen, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_signWithesk_DtH(uint8_t *sig ,uint8_t *d_sig_mem_pool,size_t Mlen,size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_verify_HtD(uint8_t *epk,uint8_t *Mess,uint8_t *sig,size_t mlen,uint8_t *d_verify_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_verify(uint8_t *d_verify_mem_pool,size_t Mlen, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_verify_DtH(uint8_t *verify ,size_t Mlen,uint8_t *d_verify_mem_pool,size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_verifyWithepk_HtD(uint8_t *epk,uint8_t *Mess,uint8_t *sig,size_t mlen,uint8_t *d_verify_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_verifyWithepk(uint8_t *d_verify_mem_pool,size_t Mlen, size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);

void crypto_sign_verifyWithepk_DtH(uint8_t *verify ,size_t Mlen,uint8_t *d_verify_mem_pool,size_t pitch, size_t batch_size, cudaStream_t stream = nullptr);