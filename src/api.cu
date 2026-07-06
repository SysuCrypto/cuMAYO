#include "api.cuh"
#include "fips202.cuh"
#include "params.h"
#include "keypair.cuh"
#include "expandsk.cuh"
#include "expandpk.cuh"
#include "sign.cuh"
#include "verify.cuh"
#include <cuda_runtime.h>
#include "code.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

void crypto_sign_keypair(uint8_t *d_keypair_mem_pool, size_t pitch,
                        size_t batch_size, cudaStream_t stream, size_t rand_index) {
    uint8_t *d_sk = d_keypair_mem_pool;
    uint8_t *d_pk = d_keypair_mem_pool + ALIGN_TO_128_BYTES(MAYO_SK_BYTES);
    uint8_t *d_eP3 = d_pk + 16;
    uint8_t *d_P = d_pk + ALIGN_TO_128_BYTES(MAYO_PK_BYTES);
    uint8_t *d_P2 = d_P + MAYO_P1_BYTES;

    gpu_keypair<<<batch_size, 32, 0, stream>>>(d_sk,d_pk,d_eP3,d_P,d_P2,pitch,rand_index);
}


void crypto_sign_keypair_DtH(uint8_t *pk, uint8_t *sk, uint8_t *d_keypair_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream) {
    uint8_t *d_sk = d_keypair_mem_pool;
    uint8_t *d_pk = d_keypair_mem_pool + ALIGN_TO_128_BYTES(MAYO_SK_BYTES);

        
    cudaMemcpy2DAsync(pk, MAYO_PK_BYTES, d_pk, pitch, MAYO_PK_BYTES, batch_size, cudaMemcpyDeviceToHost, stream);

    cudaMemcpy2DAsync(sk, MAYO_SK_BYTES, d_sk, pitch, MAYO_SK_BYTES, batch_size, cudaMemcpyDeviceToHost, stream);

}

void crypto_sign_expandsk_HtD(uint8_t *csk, uint8_t *d_expandsk_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_sk = d_expandsk_mem_pool;

    cudaMemcpy2DAsync(d_sk, pitch, csk, MAYO_SK_BYTES, MAYO_SK_BYTES, batch_size, cudaMemcpyHostToDevice, stream);
}


void crypto_sign_expandsk(uint8_t *d_expandsk_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_sk = d_expandsk_mem_pool;
    uint8_t *d_O = d_sk + MAYO_SK_BYTES;
    uint8_t *d_P1 = d_O + MAYO_O_BYTES;
    uint8_t *d_P2 = d_P1 + MAYO_P1_BYTES;

    gpu_expandsk<<<batch_size,32,0,stream>>>(d_sk,d_O,d_P1,d_P2,pitch);
}


void crypto_sign_expandsk_DtH(uint8_t *esk ,uint8_t *d_expandsk_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_esk = d_expandsk_mem_pool;
    int len = MAYO_SK_BYTES + MAYO_O_BYTES + MAYO_P_BYTES ;

    cudaMemcpy2DAsync(esk, len, d_esk, pitch, len, batch_size, cudaMemcpyDeviceToHost, stream);
}

void crypto_sign_expandpk_HtD(uint8_t *cpk, uint8_t *d_expandpk_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_pk = d_expandpk_mem_pool + MAYO_P_BYTES - 16;

    cudaMemcpy2DAsync(d_pk, pitch, cpk, MAYO_PK_BYTES, MAYO_PK_BYTES, batch_size, cudaMemcpyHostToDevice, stream);
}


void crypto_sign_expandpk(uint8_t *d_expandpk_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_epk = d_expandpk_mem_pool;
    uint8_t *d_pk = d_expandpk_mem_pool + MAYO_P_BYTES - 16;
    uint8_t *d_cpk = d_epk + MAYO_P_BYTES;

    gpu_expandpk<<<batch_size,128,0,stream>>>(d_epk,d_pk,d_cpk,pitch);
}


void crypto_sign_expandpk_DtH(uint8_t *epk ,uint8_t *d_expandpk_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_epk = d_expandpk_mem_pool;
    int len = MAYO_P_BYTES + MAYO_PK_BYTES - 16;

    cudaMemcpy2DAsync(epk, len, d_epk, pitch, len, batch_size, cudaMemcpyDeviceToHost, stream);
}


void crypto_sign_sign_HtD(uint8_t *cesk,uint8_t *cMess,size_t mlen,uint8_t *d_sign_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_esk = d_sign_mem_pool;
    int len = MAYO_SK_BYTES + MAYO_O_BYTES + MAYO_P_BYTES ;
    uint8_t *d_M = d_sign_mem_pool + len;

    cudaMemcpy2DAsync(d_esk, pitch, cesk, len, len, batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_M, pitch, cMess, mlen, mlen, batch_size, cudaMemcpyHostToDevice, stream);
}

void crypto_sign_sign(uint8_t *d_sign_mem_pool,size_t Mlen, size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_esk = d_sign_mem_pool;
    uint8_t *d_O = d_esk + MAYO_SK_BYTES;
    uint8_t *d_P1 = d_O + MAYO_O_BYTES;
    uint8_t *d_L = d_P1 + MAYO_P1_BYTES;
    uint8_t *d_Mess = d_L + MAYO_P2_BYTES;
    uint8_t *d_sig = d_Mess + Mlen;
    uint8_t *d_M = d_sig + MAYO_K*MAYO_N/2 + MAYO_SK_BYTES ;

    gpu_sign<<<batch_size,32,0,stream>>>(d_esk,d_Mess,Mlen,d_P1,d_L,d_O,d_sig,d_M,pitch);
}

void crypto_sign_sign_DtH(uint8_t *sig ,uint8_t *d_sig_mem_pool,size_t Mlen,size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_sig = d_sig_mem_pool + MAYO_SK_BYTES + MAYO_O_BYTES + MAYO_P_BYTES + Mlen;
    int len = (MAYO_K*MAYO_N)/2+MAYO_SK_BYTES;

    cudaMemcpy2DAsync(sig, len, d_sig, pitch, len, batch_size, cudaMemcpyDeviceToHost, stream);
}

void crypto_sign_signWithesk_HtD(uint8_t *csk,uint8_t *cMess,size_t mlen,uint8_t *d_sign_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_esk = d_sign_mem_pool;
    int len = ALIGN_TO_128_BYTES(MAYO_SK_BYTES + MAYO_O_BYTES + MAYO_P_BYTES) ;
    uint8_t *d_M = d_sign_mem_pool + len;

    cudaMemcpy2DAsync(d_esk, pitch, csk, MAYO_SK_BYTES, MAYO_SK_BYTES, batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_M, pitch, cMess, mlen, mlen, batch_size, cudaMemcpyHostToDevice, stream);
}

void crypto_sign_signWithesk(uint8_t *d_sign_mem_pool,size_t Mlen, size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_esk = d_sign_mem_pool;
    uint8_t *d_O = d_esk + MAYO_SK_BYTES;
    uint8_t *d_P1 = d_O + MAYO_O_BYTES;
    uint8_t *d_L = d_P1 + MAYO_P1_BYTES;
    uint8_t *d_Mess = d_sign_mem_pool + ALIGN_TO_128_BYTES(MAYO_SK_BYTES + MAYO_O_BYTES + MAYO_P_BYTES);
    uint8_t *d_sig = d_Mess + ALIGN_TO_128_BYTES(Mlen);
    uint8_t *d_M = d_sig + ALIGN_TO_128_BYTES(MAYO_K*MAYO_N/2 + MAYO_SK_BYTES) ;

    gpu_expandsk<<<batch_size,32,0,stream>>>(d_esk,d_O,d_P1,d_L,pitch);

    gpu_sign<<<batch_size,32,0,stream>>>(d_esk,d_Mess,Mlen,d_P1,d_L,d_O,d_sig,d_M,pitch);
}

void crypto_sign_signWithesk_DtH(uint8_t *sig ,uint8_t *d_sig_mem_pool,size_t Mlen,size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_sig = d_sig_mem_pool + ALIGN_TO_128_BYTES(MAYO_SK_BYTES + MAYO_O_BYTES + MAYO_P_BYTES) + ALIGN_TO_128_BYTES(Mlen);
    int len = (MAYO_K*MAYO_N)/2+MAYO_SK_BYTES;

    cudaMemcpy2DAsync(sig, len, d_sig, pitch, len, batch_size, cudaMemcpyDeviceToHost, stream);
}

void crypto_sign_verify_HtD(uint8_t *epk,uint8_t *Mess,uint8_t *sig,size_t mlen,uint8_t *d_verify_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_epk = d_verify_mem_pool;
    int epklen = ALIGN_TO_128_BYTES(MAYO_PK_BYTES + MAYO_P_BYTES - 16);
    int siglen = ALIGN_TO_128_BYTES((MAYO_K*MAYO_N)/2+MAYO_SK_BYTES);
    uint8_t *d_M = d_epk + epklen;
    uint8_t *d_sig = d_M + ALIGN_TO_128_BYTES(mlen);

    cudaMemcpy2DAsync(d_epk, pitch, epk, epklen, epklen, batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_M, pitch, Mess, mlen, mlen, batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_sig, pitch, sig, siglen, siglen, batch_size, cudaMemcpyHostToDevice, stream);
}

void crypto_sign_verify(uint8_t *d_verify_mem_pool,size_t Mlen, size_t pitch, size_t batch_size, cudaStream_t stream){
    int epklen = ALIGN_TO_128_BYTES(MAYO_PK_BYTES + MAYO_P_BYTES - 16);
    int siglen = ALIGN_TO_128_BYTES((MAYO_K*MAYO_N)/2+MAYO_SK_BYTES);
    uint8_t *d_P1 = d_verify_mem_pool;
    uint8_t *d_P2 = d_P1 + MAYO_P1_BYTES;
    uint8_t *d_P3 = d_P2 + MAYO_P2_BYTES;
    uint8_t *d_M = d_verify_mem_pool + epklen;
    uint8_t *d_sig = d_M + ALIGN_TO_128_BYTES(Mlen);
    uint8_t *d_verify = d_sig + siglen;

    gpu_verify<<<batch_size,32,0,stream>>>(d_P1,d_P2,d_P3,d_M,d_sig,d_verify,Mlen,pitch);
}

void crypto_sign_verify_DtH(uint8_t *verify ,size_t Mlen,uint8_t *d_verify_mem_pool,size_t pitch, size_t batch_size, cudaStream_t stream){
    int epklen = ALIGN_TO_128_BYTES(MAYO_PK_BYTES + MAYO_P_BYTES - 16);
    int siglen = ALIGN_TO_128_BYTES((MAYO_K*MAYO_N)/2+MAYO_SK_BYTES);
    uint8_t *d_verify = d_verify_mem_pool + epklen + ALIGN_TO_128_BYTES(Mlen) + siglen;
    int len = 1;

    cudaMemcpy2DAsync(verify, len, d_verify, pitch, len, batch_size, cudaMemcpyDeviceToHost, stream);
}

void crypto_sign_verifyWithepk_HtD(uint8_t *cpk,uint8_t *Mess,uint8_t *sig,size_t mlen,uint8_t *d_verify_mem_pool, size_t pitch, size_t batch_size, cudaStream_t stream){
    uint8_t *d_epk = d_verify_mem_pool;
    int epklen = ALIGN_TO_128_BYTES(MAYO_PK_BYTES + MAYO_P_BYTES - 16);
    int siglen = (MAYO_K*MAYO_N)/2+MAYO_SK_BYTES;
    uint8_t *d_M = d_epk + epklen;
    uint8_t *d_sig = d_M + ALIGN_TO_128_BYTES(mlen);
    uint8_t *d_pk = d_verify_mem_pool + MAYO_P_BYTES - 16;

    cudaMemcpy2DAsync(d_pk, pitch, cpk, MAYO_PK_BYTES, MAYO_PK_BYTES, batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_M, pitch, Mess, mlen, mlen, batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_sig, pitch, sig, siglen, siglen, batch_size, cudaMemcpyHostToDevice, stream);
}

void crypto_sign_verifyWithepk(uint8_t *d_verify_mem_pool,size_t Mlen, size_t pitch, size_t batch_size, cudaStream_t stream){
    int epklen = ALIGN_TO_128_BYTES(MAYO_PK_BYTES + MAYO_P_BYTES - 16);
    int siglen = ALIGN_TO_128_BYTES((MAYO_K*MAYO_N)/2+MAYO_SK_BYTES);
    uint8_t *d_P1 = d_verify_mem_pool;
    uint8_t *d_P2 = d_P1 + MAYO_P1_BYTES;
    uint8_t *d_P3 = d_P2 + MAYO_P2_BYTES;
    uint8_t *d_M = d_verify_mem_pool + epklen;
    uint8_t *d_sig = d_M + ALIGN_TO_128_BYTES(Mlen);
    uint8_t *d_verify = d_sig + siglen;
    uint8_t *d_epk = d_verify_mem_pool;
    uint8_t *d_pk = d_verify_mem_pool + MAYO_P_BYTES - 16;
    uint8_t *d_cpk = d_epk + MAYO_P_BYTES;

    gpu_expandpk<<<batch_size,128,0,stream>>>(d_epk,d_pk,d_cpk,pitch);

    gpu_verify<<<batch_size,32,0,stream>>>(d_P1,d_P2,d_P3,d_M,d_sig,d_verify,Mlen,pitch);
}

void crypto_sign_verifyWithepk_DtH(uint8_t *verify ,size_t Mlen,uint8_t *d_verify_mem_pool,size_t pitch, size_t batch_size, cudaStream_t stream){
    int epklen = ALIGN_TO_128_BYTES(MAYO_PK_BYTES + MAYO_P_BYTES - 16);
    int siglen = ALIGN_TO_128_BYTES((MAYO_K*MAYO_N)/2+MAYO_SK_BYTES);
    uint8_t *d_verify = d_verify_mem_pool + epklen + ALIGN_TO_128_BYTES(Mlen) + siglen;
    int len = 1;

    cudaMemcpy2DAsync(verify, len, d_verify, pitch, len, batch_size, cudaMemcpyDeviceToHost, stream);
}