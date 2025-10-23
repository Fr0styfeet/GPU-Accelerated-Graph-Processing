#pragma once
#include "graph.h"

// Declaration only (no definitions here)
__global__ void bfsKernel(int *rowPtr, int *colInd, int *frontier,
                          int *nextFrontier, int *visited, int *dist,
                          int V, bool *done);

void bfsGPU(const CSRGraph &g, int source);