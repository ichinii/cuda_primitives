#pragma once

constexpr inline bool is_power_of_2(int n) {
    return (n & (n-1)) == 0;
}


template <typename T>
constexpr inline T floor_power_of_2(T n) {
    int i = 0;
    while (n != 1) {
        n >>= 1;
        ++i;
    }
    return 1 << i;
};
