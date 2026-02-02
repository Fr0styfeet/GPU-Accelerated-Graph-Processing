#pragma once
#include "graph_weighted.h"

class DeltaStepping {
private:
    CSRGraphWeighted graph;
    float delta; // Delta parameter for bucketing

public:
    double cpuTime;
    double gpuTime;

    DeltaStepping(const CSRGraphWeighted& g, float delta_param = 1.0f);  // constructor

    void runCPU(int source);  // CPU Delta-Stepping
    void runGPU(int source);  // GPU Delta-Stepping
};
