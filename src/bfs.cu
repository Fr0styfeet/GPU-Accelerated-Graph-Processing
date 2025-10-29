#include "bfs.h"
#include "graph.h" 
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <queue>

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



// BFS kernel: one thread per vertex
__global__ void bfsKernel(int *rowPtr, int *colInd, int *frontier, int *nextFrontier, int *visited, int V, int *done)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;

    if (frontier[tid]) {
        int start = rowPtr[tid];
        int end = rowPtr[tid + 1];
        
        for (int i = start; i < end; i++) {
            int nei = colInd[i];
            
            // Try to mark the neighbor as visited atomically.
            if (atomicCAS(&visited[nei], 0, 1) == 0) {
                nextFrontier[nei] = 1;
                
                // CRITICAL FIX: Use atomicMin to safely set the 'done' flag to 0 (false).
                atomicMin(done, 0); 
            }
        }
    }
}


// BFS class implementation
BFS::BFS(const CSRGraph& g) : graph(g) {} //constructor


void BFS::run(int source) {
    int V = graph.rowPtr.size() - 1;
    if (V <= 0 || source >= V || source < 0) {
        cerr << "Graph is empty or source node is invalid." << endl;
        return;
    }
    
    int deviceCount = 0;
    cudaError_t deviceCheckError = cudaGetDeviceCount(&deviceCount);

    if (deviceCheckError != cudaSuccess || deviceCount == 0) {
        cerr << "--- CUDA ERROR: No devices found or driver failed. ---" << endl;
        return;
    }
    
    CUDA_CHECK(cudaSetDevice(0));

    // GPU memory
    int *d_rowPtr, *d_colInd, *d_frontier, *d_nextFrontier, *d_visited;
    int *d_done; 

    // --- NEW VERIFICATION PRINT ---
    cout << "Allocating memory for V=" << V << " nodes (Tiny size, should succeed)..." << endl;
    // --- END NEW VERIFICATION PRINT ---

    // Allocation Block 
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colInd, graph.colInd.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nextFrontier, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_done, sizeof(int))); 

    // Host vectors for initialization
    vector<int> current_frontier_h(V, 0), visited_h(V, 0);
    current_frontier_h[source] = 1; 
    visited_h[source] = 1;          

    // Copy initial data to device and check for errors
    CUDA_CHECK(cudaMemcpy(d_rowPtr, graph.rowPtr.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colInd, graph.colInd.data(), graph.colInd.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier, current_frontier_h.data(), V * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited, visited_h.data(), V * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_nextFrontier, 0, V * sizeof(int)));

    int threads = 256;
    int blocks = (V + threads - 1) / threads;

    vector<int> traversal;
    traversal.push_back(source); 

    cout << "Running CUDA BFS from source " << source << "..." << endl;

    while (true) {
        int done_h = 1; 
        
        CUDA_CHECK(cudaMemcpy(d_done, &done_h, sizeof(int), cudaMemcpyHostToDevice));

        // Kernel launch
        bfsKernel<<<blocks, threads>>>(d_rowPtr, d_colInd, d_frontier, d_nextFrontier, d_visited, V, d_done);
        CUDA_CHECK(cudaDeviceSynchronize()); 

        // Copy flag back from device
        CUDA_CHECK(cudaMemcpy(&done_h, d_done, sizeof(int), cudaMemcpyDeviceToHost));

        if (done_h == 1) break;

        // Copy the new frontier (d_nextFrontier) to the host vector
        CUDA_CHECK(cudaMemcpy(current_frontier_h.data(), d_nextFrontier, V * sizeof(int), cudaMemcpyDeviceToHost));

        // Add nodes from the current level to the traversal order
        for (int i = 0; i < V; i++) {
            if (current_frontier_h[i] == 1) {
                traversal.push_back(i);
            }
        }
        
        // Prepare for next iteration: Swap d_nextFrontier -> d_frontier
        CUDA_CHECK(cudaMemcpy(d_frontier, d_nextFrontier, V * sizeof(int), cudaMemcpyDeviceToDevice));
        // Clear d_nextFrontier
        CUDA_CHECK(cudaMemset(d_nextFrontier, 0, V * sizeof(int)));
    }

    cout << "\nBFS Traversal starting from node " << source << ":\n";
    for (int node : traversal)
        cout << node << " ";
    cout << endl;

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colInd));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_nextFrontier));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_done));
    
    CUDA_CHECK(cudaDeviceReset());
}
