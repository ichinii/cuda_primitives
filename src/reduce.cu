#include "reduce.h"

__global__ void sum_0(int *data_dst, int *data_src) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    for (int s = 1; s < blockDim.x; s*=2) {
        if (gid % (2*s) == 0)
            data_src[gid] += data_src[gid + s];

        __syncthreads();
    }

    if (tid == 0)
        data_dst[bid] = data_src[gid];
}

__global__ void sum_1(int *data_dst, int *data_src) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    extern __shared__ int sdata[];

    sdata[tid] = data_src[gid];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s*=2) {
        if (tid % (2*s) == 0)
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    if (tid == 0)
        data_dst[bid] = sdata[0];
}

__global__ void sum_2(int *data_dst, int *data_src) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    extern __shared__ int sdata[];

    sdata[tid] = data_src[gid];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s*=2) {
        int index = 2*s*tid;
        if (index < blockDim.x)
            sdata[index] += sdata[index + s];

        __syncthreads();
    }

    if (tid == 0)
        data_dst[bid] = sdata[0];
}

__global__ void sum_3(int *data_dst, int *data_src) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    extern __shared__ int sdata[];

    sdata[tid] = data_src[gid];
    __syncthreads();

    for (int s = blockDim.x/2; 0 < s; s>>=1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    if (tid == 0)
        data_dst[bid] = sdata[0];
}

__global__ void sum_4(int *data_dst, int *data_src) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x*2 + threadIdx.x;

    extern __shared__ int sdata[];

    sdata[tid] = data_src[gid] + data_src[gid + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x/2; 0 < s; s>>=1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    if (tid == 0)
        data_dst[bid] = sdata[0];
}

__device__ void sum_5_warp(volatile int *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void sum_5(int *data_dst, int *data_src) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x*2 + threadIdx.x;

    extern __shared__ int sdata[];

    sdata[tid] = data_src[gid] + data_src[gid + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x/2; 32 < s; s>>=1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    if (tid < 32)
        sum_5_warp(sdata, tid);

    if (tid == 0)
        data_dst[bid] = sdata[0];
}
