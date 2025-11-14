#pragma once
#include "graph_weighted.h"

class bellman {
private:
    CSRGraphWeighted graph;

public:
    double cpuTime; 
    double gpuTime;

    bellman(const CSRGraphWeighted& g);   // constructor

    void runCPU(int source);             // CPU Bellman–Ford
    void runGPU(int source);             // GPU Bellman–Ford
};
