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
__device__ void sum_6_block(int *sdata, int bid, int tid) {
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
        sum_6_warp<blockSize>(sdata, tid);
}

template <unsigned int blockSize>
__global__ void sum_6(int *data_dst, int *data_src, int N) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gid = blockIdx.x*blockSize + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockSize;

    extern __shared__ int sdata[];
    sdata[tid] = data_src[gid];

    for (int i = gid + gridSize; i < N; i += gridSize)
        sdata[tid] += data_src[i];
    __syncthreads();

    sum_6_block<blockSize>(sdata, bid, tid);

    if (tid == 0)
        data_dst[bid] = sdata[0];
}

void do_sum_6_impl(int *data_dst, int *data_src, int B, int T, int N) {
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

void do_sum_6(int *&data_dst, int *&data_src, int b, int t, int n) {
    t = std::min(t, n);
    b = std::min(b, n/t);
    b = floor_power_of_2(b);
    while (n > 1) {
        do_sum_6_impl(data_dst, data_src, b, t, n);
        std::swap(data_dst, data_src);
        n = b;
        t = std::min(t, n);
        b /= t;
    }
    std::swap(data_dst, data_src);
};

template <unsigned int U>
__device__ inline int sum_7_gridstep_unroll(int *data_src) {
    unsigned int gridSize = gridDim.x*blockDim.x;

    if constexpr (1 < U) {
        return data_src[(U-1)*gridSize] + sum_7_gridstep_unroll<U - 1>(data_src);
    }
    return data_src[0];
}

__device__ inline int sum_7_gridstep(int *sdata, int *data_src, int gridSize, int N) {
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    while (i+gridSize*8 < N) {
        sdata[tid] += sum_7_gridstep_unroll<8>(data_src+i);
        i += gridSize*8;
    }
    if (i+gridSize*4 < N) {
        sdata[tid] += sum_7_gridstep_unroll<4>(data_src+i);
        i += gridSize*4;
    }
    if (i+gridSize*2 < N) {
        sdata[tid] += sum_7_gridstep_unroll<2>(data_src+i);
        i += gridSize*2;
    }
    if (i+gridSize < N) {
        sdata[tid] += sum_7_gridstep_unroll<1>(data_src+i);
        i += gridSize;
    }
    if (i < N) {
        sdata[tid] += data_src[i];
    }

    __syncthreads();
}

template <unsigned int blockSize>
__global__ void sum_7(int *data_dst, int *data_src, int N) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gid = blockIdx.x*blockSize + threadIdx.x;
    unsigned int gridSize = gridDim.x*blockSize;

    extern __shared__ int sdata[];
    sdata[tid] = 0;

    sum_7_gridstep(sdata, data_src, gridSize, N);
    sum_6_block<blockSize>(sdata, bid, tid);

    if (tid == 0)
        data_dst[bid] = sdata[0];
}

void do_sum_7_impl(int *data_dst, int *data_src, int B, int T, int N) {
    assert(T <= 512);
    assert(is_power_of_2(T));

#define sum_case(t) case t: sum_7<t><<<B, t, t*sizeof(int)>>>(data_dst, data_src, N); break;
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

void do_sum_7(int *&data_dst, int *&data_src, int b, int t, int n) {
    t = std::min(t, n);
    b = std::min(b, n/t);
    b = floor_power_of_2(b);
    while (n > 1) {
        do_sum_7_impl(data_dst, data_src, b, t, n);
        std::swap(data_dst, data_src);
        n = b;
        t = std::min(t, n);
        b /= t;
    }
    std::swap(data_dst, data_src);
};

/* begin
 * Mark Harris NVIDIA Developer Technology
 */
template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
extern __shared__ int sdata[];
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockSize*2) + tid;
unsigned int gridSize = blockSize*2*gridDim.x;
sdata[tid] = 0;
while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
__syncthreads();
if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
if (tid < 32) warpReduce<blockSize>(sdata, tid);
if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void do_reduce6_impl(int *data_dst, int *data_src, int B, int T, int N) {
    assert(T <= 512);
    assert(is_power_of_2(T));

#define sum_case(t) case t: reduce6<t><<<B, t, t*sizeof(int)>>>(data_dst, data_src, N); break;
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
/* end
 * Mark Harris NVIDIA Developer Technology
 */

void do_reduce6(int *data_dst, int *data_src, int B, int T, int N) {
    int r = T*2;
    int n = N;

    while (1 < n) {
        int b = std::min(n/r, B);
        int t = std::min(T, n/2);
        do_reduce6_impl(data_dst, data_src, b, t, n);
        n /= r;
    }
}
