#include <iostream>
#include "compute.h"

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    constexpr const std::size_t T = 512;
    constexpr const std::size_t B = T;
    constexpr const std::size_t N = B*T;

    static_assert(B % T == 0);

    int *data_src;
    int *data_dst;
    cudaMallocManaged(&data_src, N * sizeof(int));
    cudaMallocManaged(&data_dst, N * sizeof(int));

    for (std::size_t i = 0; i < N; ++i)
        data_src[i] = 1;

    cudaDeviceSynchronize();

    for (auto b = B; 0 < b; b/=T) {
        sum<<<b, T>>>(data_dst, data_src);
        std::swap(data_dst, data_src);
    }

    cudaDeviceSynchronize();

    std::cout << "expected: " << N << std::endl;
    std::cout << "got: " << data_src[0] << std::endl;
    std::cout << "got: " << data_dst[0] << std::endl;
    std::cout << "got: " << data_dst << " " << data_src << std::endl;
    std::cout << "test: " << (N == data_src[0]) << std::endl;

    return 0;
}
