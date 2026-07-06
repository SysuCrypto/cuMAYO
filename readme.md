# MAYO-GPU

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

This code is submitted as supplementary material for peer review only.
Copyright © 2026 The authors. Sun Yat-sen University (SYSU). All rights reserved.
No public license is granted at this stage.
The code will be released publicly after paper acceptance.