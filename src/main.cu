#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include "reduce.h"

int main()
{
    constexpr const std::size_t N = 1<<22;
    constexpr const std::size_t T = 128;
    // constexpr const std::size_t B = N/T;
    constexpr const std::size_t S = N * sizeof(int);

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

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    auto test = [&] (const char *name, auto test_fn) {
        using namespace std::chrono;

        cudaMemcpy(data_src, init_data.data(), S, cudaMemcpyHostToDevice);

        auto start_time = steady_clock::now();
        cudaEventRecord(start_event);
        test_fn();
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        auto stop_time = steady_clock::now();

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

        auto elapsed_time2 = duration_cast<microseconds>(stop_time - start_time).count() / 1000.0f;

        int result;
        cudaMemcpy(&result, data_dst, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout
            << "running test: '" << name << "'" << std::endl
            // << "expected: " << N << std::endl
            // << "got:      " << result << std::endl
            << "ok:       " << (N == result) << std::endl
            << "duration: " << elapsed_time << std::endl
            << "duration: " << elapsed_time2 << std::endl
            << std::endl;
    };

    auto test_cpu_0 = [&] {
        using namespace std::chrono;

        auto start_time = steady_clock::now();

        int result = 0;
        for (int i = 0; i < N; ++i)
            result += init_data[i];

        auto stop_time = steady_clock::now();
        auto elapsed_time = duration_cast<microseconds>(stop_time - start_time).count() / 1000.0f;
        std::cout
            << "running cpu test" << std::endl
            << "ok:       " << (N == result) << std::endl
            << "duration: " << elapsed_time << std::endl
            << std::endl;
    };

    auto test_cpu_1 = [&] {
        using namespace std::chrono;

        auto start_time = steady_clock::now();

        int result = std::accumulate(init_data.begin(), init_data.end(), 0);

        auto stop_time = steady_clock::now();
        auto elapsed_time = duration_cast<microseconds>(stop_time - start_time).count() / 1000.0f;
        std::cout
            << "running cpu test" << std::endl
            << "ok:       " << (N == result) << std::endl
            << "duration: " << elapsed_time << std::endl
            << std::endl;
    };

    // auto test_fn_0 = [&] {
    //     for (auto b = B; 0 < b; b/=T) {
    //         sum_0<<<b, T>>>(data_dst, data_src);
    //         std::swap(data_dst, data_src);
    //     }
    //     std::swap(data_dst, data_src);
    // };

    // auto test_fn_1 = [&] {
    //     for (auto b = B; 0 < b; b/=T) {
    //         sum_1<<<b, T, T*sizeof(int)>>>(data_dst, data_src);
    //         std::swap(data_dst, data_src);
    //     }
    //     std::swap(data_dst, data_src);
    // };

    // auto test_fn_2 = [&] {
    //     for (auto b = B; 0 < b; b/=T) {
    //         sum_2<<<b, T, T*sizeof(int)>>>(data_dst, data_src);
    //         std::swap(data_dst, data_src);
    //     }
    //     std::swap(data_dst, data_src);
    // };

    // auto test_fn_3 = [&] {
    //     for (auto b = B; 0 < b; b/=T) {
    //         sum_3<<<b, T, T*sizeof(int)>>>(data_dst, data_src);
    //         std::swap(data_dst, data_src);
    //     }
    //     std::swap(data_dst, data_src);
    // };

    // auto test_fn_4 = [&] {
    //     int t = T;
    //     for (auto b = B/2; 1 < b; b/=T*2) {
    //         sum_4<<<b, T, T*sizeof(int)>>>(data_dst, data_src);
    //         std::swap(data_dst, data_src);
    //         t /= 2;
    //     }
    //     sum_4<<<1, t/2, t*sizeof(int)>>>(data_dst, data_src);
    // };

    // auto test_fn_5 = [&] {
    //     int t = T;
    //     for (auto b = B/2; 1 < b; b/=T*2) {
    //         sum_5<<<b, T, T*sizeof(int)>>>(data_dst, data_src);
    //         std::swap(data_dst, data_src);
    //         t /= 2;
    //     }
    //     sum_3<<<1, t/2, t*sizeof(int)>>>(data_dst, data_src);
    // };

    auto test_fn_6 = [&] (std::size_t maxBlocks) {
        do_sum_6(data_dst, data_src, maxBlocks, T, N);
    };

    auto test_fn_7 = [&] (std::size_t maxBlocks) {
        do_sum_7(data_dst, data_src, maxBlocks, T, N);
    };

    auto test_reduce6 = [&] (std::size_t maxBlocks) {
        do_reduce6(data_dst, data_src, maxBlocks, T, N);
    };

    test_cpu_0();
    test_cpu_1();
    // test("sum_0", test_fn_0);
    // test("sum_1", test_fn_1);
    // test("sum_2", test_fn_2);
    // test("sum_3", test_fn_3);
    // test("sum_4", test_fn_4);
    // test("sum_5", test_fn_5);

    // for (int i = 0; i < 20; ++i) {
        int i = 12;
        int maxBlocks = 1<<i;
        std::cout << "\tthread ratio: " << (T*maxBlocks / 2816.0f) << std::endl;
        std::cout << "\tmax blocks: " << maxBlocks << std::endl << std::endl;
        // test("sum_6", [=] { test_fn_6(maxBlocks); });
        test("reduce6", [=] { test_reduce6(maxBlocks); });
        test("sum_7", [=] { test_fn_7(maxBlocks); });
    // }

    cudaFree(data_src);
    cudaFree(data_dst);

    return 0;
}
