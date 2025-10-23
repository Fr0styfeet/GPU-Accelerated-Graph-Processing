#pragma once
#include <chrono>

inline double getTime() {
    using namespace std::chrono;
    auto now = high_resolution_clock::now();
    auto ms = time_point_cast<microseconds>(now).time_since_epoch().count();
    return ms / 1000.0;  // milliseconds
}
