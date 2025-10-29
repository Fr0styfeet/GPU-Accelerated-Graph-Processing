#include "dijkstra.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>
#include <float.h>

using namespace std;

// Macro for error checking every CUDA API call
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - "      \
                 << #call << " failed" << endl;                                   \
            cerr << "Error: " << (err == cudaSuccess ? "No Error" : cudaGetErrorString(err)) << endl; \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)


// GPU kernel
// *d_done is now an int: 1 means done (no relaxation occurred), 0 means not done.
__global__ void relaxKernel(int *rowPtr, int *colInd, float *weights, float *dist, int V, int *d_done) 
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V || dist[u] == FLT_MAX) return;

    int start = rowPtr[u];
    int end = rowPtr[u + 1];

    for (int i = start; i < end; i++) {
        int v = colInd[i];
        float w = weights[i];
        float new_dist = dist[u] + w;

        // Atomic relaxation loop
        // We use atomicCAS on the integer representation of the float
        float old_dist = dist[v];
        while (new_dist < old_dist) {
            // Attempt to update dist[v] with new_dist
            // Cast float to int for atomicCAS comparison (standard CUDA trick)
            float result = __int_as_float(atomicCAS((int*)&dist[v], __float_as_int(old_dist), __float_as_int(new_dist)));
            
            if (result == old_dist) {
                // Successful update! Signal that work was done.
                // Set *d_done to 0 (false)
                atomicExch(d_done, 0); 
                break; // Exit inner while loop
            }
            // Another thread updated dist[v] before us; re-read the new value
            old_dist = result;
        }
    }
}

// Dijkstra Class Implementation
Dijkstra::Dijkstra(const CSRGraphWeighted &g) : graph(g) {}

void Dijkstra::run(int source) {
    int V = graph.rowPtr.size() - 1;
    if (V <= 0 || source >= V || source < 0) {
        cerr << "Graph is empty or source node is invalid." << endl;
        return;
    }

    // Set device and check for errors
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        cerr << "--- CUDA ERROR: No devices found. ---" << endl;
        return;
    }
    CUDA_CHECK(cudaSetDevice(0));

    // GPU memory
    int *d_rowPtr, *d_colInd;
    float *d_weights, *d_dist;
    int *d_done;

    // Host arrays
    vector<float> dist_h(V, FLT_MAX);
    dist_h[source] = 0.0f;
    int done_h;

    
    vector<float> weights_f(graph.weights.begin(), graph.weights.end());

    // Allocation Block
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colInd, graph.colInd.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights, weights_f.size() * sizeof(float))); 
    CUDA_CHECK(cudaMalloc(&d_dist, V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_done, sizeof(int)));

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_rowPtr, graph.rowPtr.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colInd, graph.colInd.data(), graph.colInd.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights_f.data(), weights_f.size() * sizeof(float), cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(d_dist, dist_h.data(), V * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (V + threads - 1) / threads;

    cout << "Running CUDA Parallel Relaxation from source " << source << "...\n";

    // Relaxation loop (Parallel Bellman-Ford)
    while (true) {
        done_h = 1; // Assume finished (1 = true)
        CUDA_CHECK(cudaMemcpy(d_done, &done_h, sizeof(int), cudaMemcpyHostToDevice));

        relaxKernel<<<blocks, threads>>>(d_rowPtr, d_colInd, d_weights, d_dist, V, d_done);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&done_h, d_done, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Loop breaks if d_done remains 1 (true)
        if (done_h == 1) break;
    }

    // Copy distances back to host
    CUDA_CHECK(cudaMemcpy(dist_h.data(), d_dist, V * sizeof(float), cudaMemcpyDeviceToHost));

    // Print shortest distances
    cout << "\nDijkstra shortest distances from node " << source << ":\n";
    for (int i = 0; i < V; i++) {
        if (dist_h[i] == FLT_MAX) {
             cout << "Node " << i << " -> unreachable\n";
        } else {
             cout << "Node " << i << " -> " << dist_h[i] << "\n";
        }
    }
    cout << endl;

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colInd));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_done));
    
    CUDA_CHECK(cudaDeviceReset());
}
