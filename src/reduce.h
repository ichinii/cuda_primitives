#pragma once

#include <cassert>
#include "misc.h"

extern __global__ void sum_0(int *data_dst, int *data_src);
extern __global__ void sum_1(int *data_dst, int *data_src);
extern __global__ void sum_2(int *data_dst, int *data_src);
extern __global__ void sum_3(int *data_dst, int *data_src);
extern __global__ void sum_4(int *data_dst, int *data_src);
extern __global__ void sum_5(int *data_dst, int *data_src);

template <unsigned int blockSize>
__device__ void sum_6_warp(volatile int *sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >=  8) sdata[tid] += sdata[tid + 4];
    if (blockSize >=  4) sdata[tid] += sdata[tid + 2];
    if (blockSize >=  2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void sum_6(int *data_dst, int *data_src, int N) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockDim.x;

    extern __shared__ int sdata[];
    sdata[tid] = data_src[gid]; // TODO: half of the threads do nothing but this. do an addition here
    // sdata[tid] = 0;

    for (int i = gid + gridSize; i < N; i += gridSize)
        sdata[tid] += data_src[i];
    __syncthreads();

    // sdata[tid] = data_src[gid] + data_src[gid + blockDim.x];
    // __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
        sum_6_warp<blockSize>(sdata, tid);

    if (tid == 0)
        data_dst[bid] = sdata[0];
}

inline void sum_6(int *data_dst, int *data_src, int B, int T, int N) {
    assert(T <= 512);
    assert(is_power_of_2(T));

#define sum_case(t) case t: sum_6<t><<<B, t, t*sizeof(int)>>>(data_dst, data_src, N); break;
    switch (T) {
        sum_case(512)
        sum_case(256)
        sum_case(128)
        sum_case( 64)
        sum_case( 32)
        sum_case( 16)
        sum_case(  8)
        sum_case(  4)
        sum_case(  2)
        sum_case(  1)
    }
#undef sum_case
}
