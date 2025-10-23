#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>
#include "graph_weighted.h"

#define INF 1e9

__global__ void dijkstraKernel(
    const int *rowPtr, const int *colInd, const float *weights,
    float *dist, int *visited, int V, bool *done)
{
    int u = -1;
    float minDist = INF;

    // Find the unvisited vertex with the smallest distance
    for (int i = 0; i < V; i++) {
        if (!visited[i] && dist[i] < minDist) {
            minDist = dist[i];
            u = i;
        }
    }

    if (u == -1) return; // all done
    visited[u] = 1;

    int start = rowPtr[u];
    int end = rowPtr[u + 1];
    for (int i = start; i < end; i++) {
        int v = colInd[i];
        float w = weights[i];
        if (!visited[v] && dist[u] + w < dist[v]) {
            dist[v] = dist[u] + w;
            *done = false;
        }
    }
}

// --------------------------------------
// Dijkstra GPU driver
// --------------------------------------
void dijkstraGPU(const CSRGraphWeighted &g, int source)
{
    int V = g.rowPtr.size() - 1;

    // Host arrays
    std::vector<float> dist(V, INF);
    std::vector<int> visited(V, 0); // ✅ changed from bool to int
    dist[source] = 0;

    // Device arrays
    int *d_rowPtr, *d_colInd, *d_visited;
    float *d_weights, *d_dist;
    bool *d_done;

    cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int));
    cudaMalloc(&d_colInd, g.colInd.size() * sizeof(int));
    cudaMalloc(&d_weights, g.weights.size() * sizeof(float));
    cudaMalloc(&d_visited, V * sizeof(int));
    cudaMalloc(&d_dist, V * sizeof(float));
    cudaMalloc(&d_done, sizeof(bool));

    cudaMemcpy(d_rowPtr, g.rowPtr.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, g.colInd.data(), g.colInd.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, g.weights.data(), g.weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, visited.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist.data(), V * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 1, blocks = 1; // Single-threaded kernel (Dijkstra is sequential)
    while (true) {
        bool done = true;
        cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);

        dijkstraKernel<<<blocks, threads>>>(d_rowPtr, d_colInd, d_weights, d_dist, d_visited, V, d_done);
        cudaDeviceSynchronize();

        cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
        if (done) break;
    }

    cudaMemcpy(dist.data(), d_dist, V * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nDijkstra's shortest distances from source " << source << ":\n";
    for (int i = 0; i < V; i++) {
        if (dist[i] >= INF) std::cout << i << ": INF\n";
        else std::cout << i << ": " << dist[i] << "\n";
    }

    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_weights);
    cudaFree(d_visited);
    cudaFree(d_dist);
    cudaFree(d_done);
}
