#pragma once

#include <cassert>
#include "misc.h"

// TODO: write description of functions and requirements of the parameters

extern __global__ void sum_0(int *data_dst, int *data_src);
extern __global__ void sum_1(int *data_dst, int *data_src);
extern __global__ void sum_2(int *data_dst, int *data_src);
extern __global__ void sum_3(int *data_dst, int *data_src);
extern __global__ void sum_4(int *data_dst, int *data_src);
extern __global__ void sum_5(int *data_dst, int *data_src);

extern void do_sum_6(int *&data_dst, int *&data_src, int b, int t, int n);
extern void do_sum_7(int *&data_dst, int *&data_src, int b, int t, int n);

extern void do_reduce6(int *data_dst, int *data_src, int B, int T, int N);

// TODO: test and profile and compare
// somehow runtime mesurements are wrong. depends on order of kernel execution
// extern void test_reduce();
// extern void prof_reduce();
