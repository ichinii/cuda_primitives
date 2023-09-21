#include "compute.h"

__global__ void sum(int *data_dst, int *data_src) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    for (int i = 1; i < blockDim.x; i*=2) {
        if (gid % (2*i) == 0)
            data_src[gid] += data_src[gid + i];

        __syncthreads();
    }

    if (tid == 0)
        data_dst[bid] = data_src[gid];
}
