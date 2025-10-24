// dijkstra.cu
#include "dijkstra.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>

using namespace std;

// ------------------------------
// GPU kernel for relaxation
// ------------------------------
__global__ void relaxKernel(int *rowPtr, int *colInd, float *weights,
                            float *dist, char *visited, int V, bool *done) 
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V || visited[u] || dist[u] == FLT_MAX) return;

    int start = rowPtr[u];
    int end = rowPtr[u + 1];

    for (int i = start; i < end; i++) {
        int v = colInd[i];
        float w = weights[i];

        // Atomic relaxation
        float oldDist = __int_as_float(atomicMin((int*)&dist[v], __float_as_int(dist[u] + w)));
        if (oldDist > dist[u] + w) {
            *done = false;
        }
    }

    visited[u] = 1;
}

// ------------------------------
// Dijkstra Class Implementation
// ------------------------------
Dijkstra::Dijkstra(const CSRGraphWeighted &g) : graph(g) {}

void Dijkstra::run(int source) {
    int V = graph.rowPtr.size() - 1;

    // GPU memory
    int *d_rowPtr, *d_colInd;
    float *d_weights, *d_dist;
    char *d_visited;
    bool *d_done;

    cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int));
    cudaMalloc(&d_colInd, graph.colInd.size() * sizeof(int));
    cudaMalloc(&d_weights, graph.weights.size() * sizeof(float));
    cudaMalloc(&d_dist, V * sizeof(float));
    cudaMalloc(&d_visited, V * sizeof(char));
    cudaMalloc(&d_done, sizeof(bool));

    // Host arrays
    vector<float> dist(V, FLT_MAX);
    vector<char> visited(V, 0);
    dist[source] = 0.0f;

    // Copy to GPU
    cudaMemcpy(d_rowPtr, graph.rowPtr.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, graph.colInd.data(), graph.colInd.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, graph.weights.data(), graph.weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist.data(), V * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, visited.data(), V * sizeof(char), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (V + threads - 1) / threads;

    while (true) {
        bool done = true;
        cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);

        relaxKernel<<<blocks, threads>>>(d_rowPtr, d_colInd, d_weights, d_dist, d_visited, V, d_done);
        cudaDeviceSynchronize();

        cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
        if (done) break;
    }

    // Copy distances back to host
    cudaMemcpy(dist.data(), d_dist, V * sizeof(float), cudaMemcpyDeviceToHost);

    // Print shortest distances
    cout << "\nDijkstra shortest distances from node " << source << ":\n";
    for (int i = 0; i < V; i++) {
        cout << "Node " << i << " -> " << dist[i] << "\n";
    }

    // Free GPU memory
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_weights);
    cudaFree(d_dist);
    cudaFree(d_visited);
    cudaFree(d_done);
}
