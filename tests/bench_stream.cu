#include <iostream>
#include <vector>
#include "aes128_ctr.cuh"
#include "params.h"
#include "api.cuh"
#include "time.cuh"
#include "code.h"
using namespace std;
#define Mlen 32

void bench_cudamayo(int batch_size, int n_streams)
{
    aes128_tables_init();
    const int measure_iters = 10;
    std::vector<cudaStream_t> streams_vec(n_streams);
    for (int i = 0; i < n_streams; ++i) {
        cudaStreamCreateWithFlags(&streams_vec[i], cudaStreamNonBlocking);
    }
    uint8_t *d_keypair_mem_pool;
    size_t d_keypair_mem_pool_pitch;
    uint8_t *h_pk, *h_sk;
    cudaMallocHost(&h_pk, batch_size * MAYO_PK_BYTES);
    cudaMallocHost(&h_sk, batch_size * MAYO_SK_BYTES);

    size_t byte_size_per_keypair = ALIGN_TO_128_BYTES(MAYO_PK_BYTES) + ALIGN_TO_128_BYTES(MAYO_SK_BYTES) ;

    const size_t P1_bytes = MAYO_P1_BYTES;
    const size_t P2_bytes = MAYO_P2_BYTES;
    const size_t P_bytes  = ALIGN_TO_128_BYTES(P1_bytes + P2_bytes);
    size_t mem_size_per_keypair = byte_size_per_keypair + P_bytes;
    cudaMallocPitch(&d_keypair_mem_pool, &d_keypair_mem_pool_pitch, mem_size_per_keypair, batch_size);

    for(int i = 0; i < n_streams; i++)
    {
        int sub_batch = batch_size / n_streams;
        int last_sub = sub_batch + ((i == n_streams - 1)?(batch_size % n_streams):0);
        if(last_sub == 0) continue;
        int base_idx = i * sub_batch;
        uint8_t *h_pk_i = h_pk + (size_t)base_idx * MAYO_PK_BYTES;
        uint8_t *h_sk_i = h_sk + (size_t)base_idx * MAYO_SK_BYTES;
        uint8_t *d_pool_i = d_keypair_mem_pool + (size_t)base_idx * d_keypair_mem_pool_pitch;
        crypto_sign_keypair(d_pool_i, d_keypair_mem_pool_pitch, last_sub, streams_vec[i]);
        crypto_sign_keypair_DtH(h_pk_i, h_sk_i, d_pool_i, d_keypair_mem_pool_pitch, last_sub, streams_vec[i]);
    } 
    cudaDeviceSynchronize();
    {
        ChronoTimer timer_keypair_stream("keypair stream",batch_size);
        for(int it = 0; it < measure_iters; it++)
        {
            timer_keypair_stream.start();
            for(int i = 0; i < n_streams; i++)
            {
                int sub_batch = batch_size / n_streams;
                int last_sub = sub_batch + ((i == n_streams - 1)?(batch_size % n_streams):0);
                if(last_sub == 0) continue;
                int base_idx = i * sub_batch;
                uint8_t *h_pk_i = h_pk + (size_t)base_idx * MAYO_PK_BYTES;
                uint8_t *h_sk_i = h_sk + (size_t)base_idx * MAYO_SK_BYTES;
                uint8_t *d_pool_i = d_keypair_mem_pool + (size_t)base_idx * d_keypair_mem_pool_pitch;
                crypto_sign_keypair(d_pool_i, d_keypair_mem_pool_pitch, last_sub, streams_vec[i]);
                crypto_sign_keypair_DtH(h_pk_i, h_sk_i, d_pool_i, d_keypair_mem_pool_pitch, last_sub, streams_vec[i]);
            } 
            cudaDeviceSynchronize();
            timer_keypair_stream.stop();
        }
    }
    cudaFree(d_keypair_mem_pool);

    //esk+sign
    uint8_t *d_sign_mem_pool;
    size_t d_sign_mem_pool_pitch;
    uint8_t *h_sig, *h_MESS;

    cudaMallocHost(&h_MESS, batch_size * Mlen);
    srand(time(NULL));

    for (size_t i = 0; i < (size_t)batch_size * Mlen; i++) {
        h_MESS[i] = rand() & 0xFF;
    }
    int siglen = (MAYO_K*MAYO_N)/2+MAYO_SK_BYTES;
    cudaMallocHost(&h_sig, batch_size * siglen);
    size_t mem_size_per_sign = ALIGN_TO_128_BYTES(MAYO_SK_BYTES + MAYO_O_BYTES + MAYO_P_BYTES) + ALIGN_TO_128_BYTES(Mlen) + ALIGN_TO_128_BYTES(MAYO_K*MAYO_N/2 + MAYO_SK_BYTES) + ALIGN_TO_128_BYTES(MAYO_K * MAYO_M * ((MAYO_O+1)/2));

    cudaMallocPitch(&d_sign_mem_pool, &d_sign_mem_pool_pitch, mem_size_per_sign, batch_size);
    for(int i = 0; i < n_streams; i++)
    {
        int sub_batch = batch_size / n_streams;
        int last_sub = sub_batch + ((i == n_streams - 1)?(batch_size % n_streams):0);
        if(last_sub == 0) continue;
        int base_idx = i * sub_batch;
        uint8_t *d_pool_i = d_sign_mem_pool + (size_t)base_idx * d_sign_mem_pool_pitch;
        uint8_t *h_sk_i = h_sk + (size_t)base_idx * MAYO_SK_BYTES;
        uint8_t *h_MESS_i = h_MESS + (size_t)base_idx * Mlen;
        uint8_t *h_sig_i = h_sig + (size_t)base_idx * siglen;
        crypto_sign_signWithesk_HtD(h_sk_i, h_MESS_i, Mlen, d_pool_i, d_sign_mem_pool_pitch, last_sub, streams_vec[i]);
        crypto_sign_signWithesk(d_pool_i, Mlen, d_sign_mem_pool_pitch, last_sub, streams_vec[i]);
        crypto_sign_signWithesk_DtH(h_sig_i, d_pool_i, Mlen, d_sign_mem_pool_pitch, last_sub, streams_vec[i]);
    } 
    cudaDeviceSynchronize();
    {
        ChronoTimer timer_sign_stream("sign stream",batch_size);
        for(int it = 0; it < measure_iters; it++)
        {
            timer_sign_stream.start();
            for(int i = 0; i < n_streams; i++)
            {
                int sub_batch = batch_size / n_streams;
                int last_sub = sub_batch + ((i == n_streams - 1)?(batch_size % n_streams):0);
                if(last_sub == 0) continue;
                int base_idx = i * sub_batch;
                uint8_t *d_pool_i = d_sign_mem_pool + (size_t)base_idx * d_sign_mem_pool_pitch;
                uint8_t *h_sk_i = h_sk + (size_t)base_idx * MAYO_SK_BYTES;
                uint8_t *h_MESS_i = h_MESS + (size_t)base_idx * Mlen;
                uint8_t *h_sig_i = h_sig + (size_t)base_idx * siglen;
                crypto_sign_signWithesk_HtD(h_sk_i, h_MESS_i, Mlen, d_pool_i, d_sign_mem_pool_pitch, last_sub, streams_vec[i]);
                crypto_sign_signWithesk(d_pool_i, Mlen, d_sign_mem_pool_pitch, last_sub, streams_vec[i]);
                crypto_sign_signWithesk_DtH(h_sig_i, d_pool_i, Mlen, d_sign_mem_pool_pitch, last_sub, streams_vec[i]);
            } 
            cudaDeviceSynchronize();
            timer_sign_stream.stop();
        }
    }
    cudaFreeHost(h_sk);
    cudaFree(d_sign_mem_pool);

    //epk+verify
    int epklen = MAYO_PK_BYTES + MAYO_P_BYTES - 16;
    uint8_t *h_verify;
    cudaMallocHost(&h_verify, batch_size);
    uint8_t *d_verify_mem_pool;
    size_t d_verify_mem_pool_pitch;

    size_t mem_size_per_verify = ALIGN_TO_128_BYTES(epklen) + ALIGN_TO_128_BYTES(Mlen) + ALIGN_TO_128_BYTES(siglen) + 1;

    cudaMallocPitch(&d_verify_mem_pool, &d_verify_mem_pool_pitch, mem_size_per_verify, batch_size);
    for(int i = 0; i < n_streams; i++)
    {
        int sub_batch = batch_size / n_streams;
        int last_sub = sub_batch + ((i == n_streams - 1)?(batch_size % n_streams):0);
        if(last_sub == 0) continue;
        int base_idx = i * sub_batch;

        uint8_t *h_pk_i = h_pk +  base_idx * MAYO_PK_BYTES;
        uint8_t *h_MESS_i = h_MESS + base_idx * Mlen;
        uint8_t *h_sig_i = h_sig + base_idx * siglen;
        uint8_t *h_verify_i = h_verify + base_idx;
        uint8_t *d_pool_i = d_verify_mem_pool + (size_t)base_idx * d_verify_mem_pool_pitch;
        crypto_sign_verifyWithepk_HtD(h_pk_i,h_MESS_i,h_sig_i,Mlen, d_pool_i, d_verify_mem_pool_pitch, last_sub, streams_vec[i]);
        crypto_sign_verifyWithepk(d_pool_i,Mlen,d_verify_mem_pool_pitch,last_sub,streams_vec[i]);
        crypto_sign_verifyWithepk_DtH(h_verify_i,Mlen,d_pool_i,d_verify_mem_pool_pitch, last_sub, streams_vec[i]);
    }
    cudaDeviceSynchronize();
    {
        ChronoTimer timer_verify_stream("verify stream",batch_size);
        for(int it = 0; it < measure_iters; it++)
        {
            timer_verify_stream.start();
            for(int i = 0; i < n_streams; i++)
            {
                int sub_batch = batch_size / n_streams;
                int last_sub = sub_batch + ((i == n_streams - 1)?(batch_size % n_streams):0);
                if(last_sub == 0) continue;
                int base_idx = i * sub_batch;

                uint8_t *h_pk_i = h_pk +  base_idx * MAYO_PK_BYTES;
                uint8_t *h_MESS_i = h_MESS + base_idx * Mlen;
                uint8_t *h_sig_i = h_sig + base_idx * siglen;
                uint8_t *h_verify_i = h_verify + base_idx;
                uint8_t *d_pool_i = d_verify_mem_pool + (size_t)base_idx * d_verify_mem_pool_pitch;
                crypto_sign_verifyWithepk_HtD(h_pk_i,h_MESS_i,h_sig_i,Mlen, d_pool_i, d_verify_mem_pool_pitch, last_sub, streams_vec[i]);
                crypto_sign_verifyWithepk(d_pool_i,Mlen,d_verify_mem_pool_pitch,last_sub,streams_vec[i]);
                crypto_sign_verifyWithepk_DtH(h_verify_i,Mlen,d_pool_i,d_verify_mem_pool_pitch, last_sub, streams_vec[i]);
            }
            cudaDeviceSynchronize();
            timer_verify_stream.stop();
        }
    }

    for(int i = 0; i < batch_size; i++)
    {
        if(h_verify[i] != 0)
        {
            std::cout << "signature verification failed at index " << i << std::endl;
            exit(1);
        }
    }
    cudaFreeHost(h_sig);
    cudaFreeHost(h_MESS);
    cudaFreeHost(h_pk);
    cudaFreeHost(h_verify);
    cudaFree(d_verify_mem_pool);

    std::cout << "bench_cudamayo done, batch_size = " << batch_size
              << ", MAYO_VERSION = " << MAYO_VERSION << std::endl;

    for (auto &s : streams_vec) cudaStreamDestroy(s);
}

int main()
{
    int batch   = 20000;
    int streams = 10;

    bench_cudamayo(batch, streams);

    return 0;
}
