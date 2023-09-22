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
        data_dst[bid] = sdata[tid];
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
        data_dst[bid] = sdata[tid];
}
