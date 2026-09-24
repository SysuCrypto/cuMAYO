# cuMAYO

## Description

A CUDA implementation of MAYO, a post-quantum digital signature scheme.

## Prerequisites

We have tested this project on NVIDIA GeForce RTX 4090, NVIDIA Tesla A100, and NVIDIA GeForce RTX 3090. Other hardware and software configurations may also work, but they have not been tested in the current project. Below we provide the best-performing configuration on the RTX 4090 platform.

### Hardware

- CPU: 13th Gen Intel(R) Core(TM) i7-13700K (16 cores, 24 threads, up to 5.4 GHz)
- GPU: NVIDIA GeForce RTX 4090 (24 GB VRAM)

### Software

- GCC/G++ 11.4.0
- CMake 3.22.1
- CUDA 12.4

## Build

```bash
cd tests
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

## Run

```bash
cd tests
./build/bench_stream_mayo1
```

The current CMake configuration builds benchmark binaries for `MAYO_1`, `MAYO_2`, `MAYO_3`, and `MAYO_5`.

## License

The cuMAYO project is released under the Apache License 2.0. See the `LICENSE` file for details.

The CUDA implementation of FIPS202 includes code adapted from third-party MIT-licensed code. The original copyright and MIT License notices are retained in the corresponding source files.

## How to Cite

If you use cuMAYO in your work, please cite the following paper:

Chen, H., Huang, J., Jin, S., Cheung, R. C. C., Chen, D., & Dai, W. (2026). High-Throughput GPU Design and Implementation of MAYO with Matrix Computation Reordering. *IACR Transactions on Cryptographic Hardware and Embedded Systems*, *2026*(4), 496–521. [https://doi.org/10.46586/tches.v2026.i4.496-521](https://doi.org/10.46586/tches.v2026.i4.496-521)

```bibtex
@article{chen2026highthroughput,
  author  = {Haoyang Chen and
             Junhao Huang and
             Shutong Jin and
             Ray C. C. Cheung and
             Donglong Chen and
             Wangchen Dai},
  title   = {High-Throughput GPU Design and Implementation of {MAYO} with Matrix Computation Reordering},
  journal = {IACR Transactions on Cryptographic Hardware and Embedded Systems},
  volume  = {2026},
  number  = {4},
  pages   = {496--521},
  year    = {2026},
  doi     = {10.46586/tches.v2026.i4.496-521}
}
```
