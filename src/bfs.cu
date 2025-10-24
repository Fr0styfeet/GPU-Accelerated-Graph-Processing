#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "graph.h"
#include "bfs.h"

using namespace std;

// ------------------------------
// BFS kernel: one thread per vertex
// ------------------------------
__global__ void bfsKernel(int *rowPtr, int *colInd, int *frontier, int *nextFrontier, int *visited, int V, bool *done)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;

    if (frontier[tid]) {
        frontier[tid] = 0;
        int start = rowPtr[tid];
        int end = rowPtr[tid + 1];
        for (int i = start; i < end; i++) {
            int nei = colInd[i];
            if (atomicCAS(&visited[nei], 0, 1) == 0) {
                nextFrontier[nei] = 1;
                *done = false;
            }
        }
    }
}

// ------------------------------
// BFS class implementation
// ------------------------------
BFS::BFS(const CSRGraph& g) : graph(g) {}

void BFS::run(int source) {
    int V = graph.rowPtr.size() - 1;

    // GPU memory
    int *d_rowPtr, *d_colInd, *d_frontier, *d_nextFrontier, *d_visited;
    bool *d_done;

    cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int));
    cudaMalloc(&d_colInd, graph.colInd.size() * sizeof(int));
    cudaMalloc(&d_frontier, V * sizeof(int));
    cudaMalloc(&d_nextFrontier, V * sizeof(int));
    cudaMalloc(&d_visited, V * sizeof(int));
    cudaMalloc(&d_done, sizeof(bool));

    vector<int> frontier(V, 0), visited(V, 0);
    frontier[source] = 1;
    visited[source] = 1;

    cudaMemcpy(d_rowPtr, graph.rowPtr.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, graph.colInd.data(), graph.colInd.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier, frontier.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, visited.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_nextFrontier, 0, V * sizeof(int));

    int threads = 256;
    int blocks = (V + threads - 1) / threads;

    vector<int> traversal;
    traversal.push_back(source);

    while (true) {
        bool done = true;
        cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);

        bfsKernel<<<blocks, threads>>>(d_rowPtr, d_colInd, d_frontier, d_nextFrontier, d_visited, V, d_done);
        cudaDeviceSynchronize();

        cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);

        cudaMemcpy(frontier.data(), d_nextFrontier, V * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < V; i++) {
            if (frontier[i] == 1)
                traversal.push_back(i);
        }

        if (done) break;

        cudaMemcpy(d_frontier, d_nextFrontier, V * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(d_nextFrontier, 0, V * sizeof(int));
    }

    cout << "\nBFS Traversal starting from node " << source << ":\n";
    for (int node : traversal)
        cout << node << " ";
    cout << endl;

    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_frontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_visited);
    cudaFree(d_done);
}
