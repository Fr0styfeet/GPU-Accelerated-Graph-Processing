// bfs.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "graph.h"
#include "utils.h"
#include "bfs.h"
using namespace std;

// -----------------------------------------
// BFS kernel: one thread per vertex
// -----------------------------------------
__global__ void bfsKernel(int *rowPtr, int *colInd, int *frontier, int *nextFrontier, 
                          int *visited, int V, int *foundNew)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;

    if (frontier[tid] == 1) {
        int start = rowPtr[tid];
        int end = rowPtr[tid + 1];
        for (int i = start; i < end; i++) {
            int nei = colInd[i];
            if (atomicCAS(&visited[nei], 0, 1) == 0) {
                nextFrontier[nei] = 1;
                atomicAdd(foundNew, 1);
            }
        }
    }
}

// -----------------------------------------
// BFS GPU function
// -----------------------------------------
void bfsGPU(const CSRGraph &g, int source) {
    int V = g.rowPtr.size() - 1;

    // GPU memory
    int *d_rowPtr, *d_colInd, *d_frontier, *d_nextFrontier, *d_visited, *d_foundNew;

    cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int));
    cudaMalloc(&d_colInd, g.colInd.size() * sizeof(int));
    cudaMalloc(&d_frontier, V * sizeof(int));
    cudaMalloc(&d_nextFrontier, V * sizeof(int));
    cudaMalloc(&d_visited, V * sizeof(int));
    cudaMalloc(&d_foundNew, sizeof(int));

    // Initialize host arrays
    vector<int> frontier(V, 0), visited(V, 0);
    frontier[source] = 1;
    visited[source] = 1;

    // Copy to GPU
    cudaMemcpy(d_rowPtr, g.rowPtr.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, g.colInd.data(), g.colInd.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier, frontier.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, visited.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_nextFrontier, 0, V * sizeof(int));

    int threads = 256;
    int blocks = (V + threads - 1) / threads;

    vector<int> traversal;
    traversal.push_back(source);
    
    while (true) {
        int foundNew = 0;
        cudaMemcpy(d_foundNew, &foundNew, sizeof(int), cudaMemcpyHostToDevice);

        bfsKernel<<<blocks, threads>>>(d_rowPtr, d_colInd, d_frontier, d_nextFrontier, 
                                       d_visited, V, d_foundNew);
        cudaDeviceSynchronize();

        cudaMemcpy(&foundNew, d_foundNew, sizeof(int), cudaMemcpyDeviceToHost);

        if (foundNew == 0) break;

        // Get newly discovered nodes
        cudaMemcpy(frontier.data(), d_nextFrontier, V * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < V; i++) {
            if (frontier[i] == 1) {
                traversal.push_back(i);
            }
        }

        // Swap frontiers
        cudaMemcpy(d_frontier, d_nextFrontier, V * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(d_nextFrontier, 0, V * sizeof(int));
    }

    // Print BFS traversal order
    cout << "\nBFS Traversal starting from node " << source << ":\n";
    for (int node : traversal)
        cout << node << " ";
    cout << endl;

    // Free GPU memory
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_frontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_visited);
    cudaFree(d_foundNew);
}