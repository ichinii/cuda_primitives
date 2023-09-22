#include <iostream>
#include <vector>
#include "reduce.h"

int main()
{
    constexpr const std::size_t T = 512;
    constexpr const std::size_t B = T*T;
    constexpr const std::size_t N = B*T;
    constexpr const std::size_t S = N * sizeof(int);

    static_assert(B % T == 0);

    std::cout << std::boolalpha
        << "num elements: " << N << std::endl
        << "num threads per block: " << T << std::endl
        << std::endl;

    int *data_src;
    int *data_dst;
    cudaMalloc(&data_src, S);
    cudaMalloc(&data_dst, S);

    auto init_data = std::vector<int>(N);
    for (std::size_t i = 0; i < N; ++i)
        init_data[i] = 1;

    auto test = [&] (const char *name, auto f, int shmem_size) {
        cudaMemcpy(data_src, init_data.data(), S, cudaMemcpyHostToDevice);

        for (auto b = B; 0 < b; b/=T) {
            f<<<b, T, shmem_size>>>(data_dst, data_src);
            std::swap(data_dst, data_src);
        }
        std::swap(data_dst, data_src);

        int result;
        cudaMemcpy(&result, data_dst, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "running test: '" << name << "'" << std::endl
            << "expected: " << N << std::endl
            << "got: " << result << std::endl
            << "ok: " << (N == result) << std::endl
            << std::endl;
    };

    test("sum_0", sum_0, 0);
    test("sum_1", sum_1, T*sizeof(int));
    test("sum_2", sum_2, T*sizeof(int));

    cudaFree(data_src);
    cudaFree(data_dst);

    return 0;
}
