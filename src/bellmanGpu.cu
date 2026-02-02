#include "bellman.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>
#include <float.h>
#include <chrono>
#include <cstring>
#include <atomic>

using namespace std;


// GPU kernel (Bellman–Ford relax all edges)
__global__ void BFKernel(int *rowPtr, int *colInd, float *weights, float *dist, int V, int *d_done)
{

    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= V || dist[u] == FLT_MAX) return;

    
    int start = rowPtr[u];
    int end = rowPtr[u + 1];


    for (int i = start; i < end; i++) {
        int v = colInd[i];
        float w = weights[i];
        float new_dist = dist[u] + w;

        float old_dist = dist[v];

        // Atomic CAS loop
        while (new_dist < old_dist) {
            float result = __int_as_float(atomicCAS((int*)&dist[v],__float_as_int(old_dist),__float_as_int(new_dist)));

            // CAS success → distance updated
            if (result == old_dist) {
                atomicExch(d_done, 0);
                break;
            }
            old_dist = result;
        }
    }
}


// Class implementation (converted to Bellman–Ford)
bellman::bellman(const CSRGraphWeighted &g) : graph(g) {}

void bellman::runGPU(int source)
{
    // Time start
    auto start_total = chrono::high_resolution_clock::now();


    int V = graph.rowPtr.size() - 1;
    if (V <= 0 || source >= V || source < 0) {
        cerr << "Graph is empty or invalid source.\n";
        return;
    }

    cudaSetDevice(0);

    int *d_rowPtr, *d_colInd;
    float *d_weights, *d_dist;
    int *d_done;

    vector<float> dist_h(V, FLT_MAX);
    dist_h[source] = 0.0f;

    vector<float> weights_f(graph.weights.begin(), graph.weights.end());


    // Allocate GPU
    cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int));
    cudaMalloc(&d_colInd, graph.colInd.size() * sizeof(int));
    cudaMalloc(&d_weights, weights_f.size() * sizeof(float));
    cudaMalloc(&d_dist, V * sizeof(float));
    cudaMalloc(&d_done, sizeof(int));



    // Copy to GPU
    cudaMemcpy(d_rowPtr, graph.rowPtr.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, graph.colInd.data(), graph.colInd.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_f.data(), weights_f.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist_h.data(), V * sizeof(float), cudaMemcpyHostToDevice);


    int threads = 1024;
    int blocks = (V + threads - 1) / threads;


    // Bellman–Ford loop (V - 1 iterations)
    for (int iter = 0; iter < V - 1; iter++) {

        int done_h = 1;
        cudaMemcpy(d_done, &done_h, sizeof(int), cudaMemcpyHostToDevice);

        BFKernel<<<blocks, threads>>>(d_rowPtr, d_colInd, d_weights, d_dist, V, d_done);
        cudaDeviceSynchronize();

        cudaMemcpy(&done_h, d_done, sizeof(int), cudaMemcpyDeviceToHost);

        if (done_h == 1) {
            cout << "Converged early at iteration " << iter << endl;
            break;
        }
    }

    // Copy back results
    cudaMemcpy(dist_h.data(), d_dist, V * sizeof(float), cudaMemcpyDeviceToHost);



    // Timing end
    auto end_total = chrono::high_resolution_clock::now();

    // Output
    // cout << "\nBellman-Ford shortest distances:\n";
    // for (int i = 0; i < V; i++) {
    //     if (dist_h[i] == FLT_MAX)
    //         cout << "Node " << i << " -> unreachable\n";
    //     else
    //         cout << "Node " << i << " -> " << dist_h[i] << "\n";
    // }


    gpuTime = chrono::duration<double, milli>(end_total - start_total).count();
    cout << "\nGPU Execution Time: " << gpuTime << " ms\n";

    // Free GPU
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_weights);
    cudaFree(d_dist);
    cudaFree(d_done);

    cudaDeviceReset();
}