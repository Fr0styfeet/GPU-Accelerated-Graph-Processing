#include "delta.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>
#include <float.h>
#include <chrono>
#include <cstring>
#include <algorithm>

using namespace std;

// Kernel to relax light edges (weight <= delta)
__global__ void relaxLightEdges(int *rowPtr, int *colInd, float *weights, float *dist, int *bucket, int bucketIdx, int V, float delta, int *nextBucket, int *changed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;

    // Check if this vertex is in current bucket
    if (bucket[tid] != bucketIdx) return;

    int u = tid;
    float dist_u = dist[u];
    
    int start = rowPtr[u];
    int end = rowPtr[u + 1];

    for (int i = start; i < end; i++) {
        int v = colInd[i];
        float w = weights[i];
        
        // Only relax light edges
        if (w > delta) continue;
        
        float new_dist = dist_u + w;
        float old_dist = dist[v];

        // Atomic CAS loop for distance update
        while (new_dist < old_dist) {
            float result = __int_as_float(
                atomicCAS((int*)&dist[v], 
                         __float_as_int(old_dist), 
                         __float_as_int(new_dist))
            );

            if (result == old_dist) {
                // Successfully updated distance
                int newBucket = (int)(new_dist / delta);
                atomicExch(&nextBucket[v], newBucket);
                atomicExch(changed, 1);
                break;
            }
            old_dist = result;
        }
    }
}

// Kernel to relax heavy edges (weight > delta)
__global__ void relaxHeavyEdges(int *rowPtr, int *colInd, float *weights, float *dist, int *bucket, int bucketIdx, int V, float delta, int *nextBucket, int *changed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;

    // Check if this vertex is in current bucket
    if (bucket[tid] != bucketIdx) return;

    int u = tid;
    float dist_u = dist[u];
    
    int start = rowPtr[u];
    int end = rowPtr[u + 1];

    for (int i = start; i < end; i++) {
        int v = colInd[i];
        float w = weights[i];
        
        // Only relax heavy edges
        if (w <= delta) continue;
        
        float new_dist = dist_u + w;
        float old_dist = dist[v];

        // Atomic CAS loop for distance update
        while (new_dist < old_dist) {
            float result = __int_as_float(
                atomicCAS((int*)&dist[v], 
                         __float_as_int(old_dist), 
                         __float_as_int(new_dist))
            );

            if (result == old_dist) {
                // Successfully updated distance
                int newBucket = (int)(new_dist / delta);
                atomicExch(&nextBucket[v], newBucket);
                atomicExch(changed, 1);
                break;
            }
            old_dist = result;
        }
    }
}

// Kernel to update bucket assignments
__global__ void updateBuckets(int *bucket, int *nextBucket, int V)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;
    
    bucket[tid] = nextBucket[tid];
}

// Class implementation
DeltaStepping::DeltaStepping(const CSRGraphWeighted &g, float delta_param) : graph(g), delta(delta_param) {}

void DeltaStepping::runGPU(int source)
{
    auto start_total = chrono::high_resolution_clock::now();

    int V = graph.rowPtr.size() - 1;
    if (V <= 0 || source >= V || source < 0) {
        cerr << "Graph is empty or invalid source.\n";
        return;
    }

    cudaSetDevice(0);

    // Device pointers
    int *d_rowPtr, *d_colInd;
    float *d_weights, *d_dist;
    int *d_bucket, *d_nextBucket;
    int *d_changed;

    // Host data initialization
    vector<float> dist_h(V, FLT_MAX);
    dist_h[source] = 0.0f;

    vector<int> bucket_h(V, INT_MAX);
    bucket_h[source] = 0; // Source is in bucket 0

    vector<int> nextBucket_h(V, INT_MAX);
    nextBucket_h[source] = 0;

    vector<float> weights_f(graph.weights.begin(), graph.weights.end());

    // Allocate GPU memory
    cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int));
    cudaMalloc(&d_colInd, graph.colInd.size() * sizeof(int));
    cudaMalloc(&d_weights, weights_f.size() * sizeof(float));
    cudaMalloc(&d_dist, V * sizeof(float));
    cudaMalloc(&d_bucket, V * sizeof(int));
    cudaMalloc(&d_nextBucket, V * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    // Copy to GPU
    cudaMemcpy(d_rowPtr, graph.rowPtr.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, graph.colInd.data(), graph.colInd.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_f.data(), weights_f.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist_h.data(), V * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bucket, bucket_h.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nextBucket, nextBucket_h.data(), V * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (V + threads - 1) / threads;

    // Delta-Stepping main loop
    int currentBucket = 0;
    int maxIterations = V * 2; // Safety limit
    
    for (int iter = 0; iter < maxIterations; iter++) {
        int changed_h = 0;
        
        // Phase 1: Relax light edges until no changes in current bucket
        bool lightPhaseActive = true;
        while (lightPhaseActive) {
            changed_h = 0;
            cudaMemcpy(d_changed, &changed_h, sizeof(int), cudaMemcpyHostToDevice);
            
            relaxLightEdges<<<blocks, threads>>>(d_rowPtr, d_colInd, d_weights, d_dist, d_bucket, currentBucket, V, delta, d_nextBucket, d_changed);
            cudaDeviceSynchronize();
            
            cudaMemcpy(&changed_h, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
            
            if (changed_h == 0) {
                lightPhaseActive = false;
            } else {
                // Update buckets
                updateBuckets<<<blocks, threads>>>(d_bucket, d_nextBucket, V);
                cudaDeviceSynchronize();
            }
        }
        
        // Phase 2: Relax heavy edges once
        changed_h = 0;
        cudaMemcpy(d_changed, &changed_h, sizeof(int), cudaMemcpyHostToDevice);
        
        relaxHeavyEdges<<<blocks, threads>>>(d_rowPtr, d_colInd, d_weights, d_dist, d_bucket, currentBucket, V, delta, d_nextBucket, d_changed);
        cudaDeviceSynchronize();
        
        // Update buckets after heavy edge relaxation
        updateBuckets<<<blocks, threads>>>(d_bucket, d_nextBucket, V);
        cudaDeviceSynchronize();
        
        // Move to next non-empty bucket
        cudaMemcpy(bucket_h.data(), d_bucket, V * sizeof(int), cudaMemcpyDeviceToHost);
        
        int nextBucket = INT_MAX;
        for (int i = 0; i < V; i++) {
            if (bucket_h[i] > currentBucket && bucket_h[i] < nextBucket) {
                nextBucket = bucket_h[i];
            }
        }
        
        if (nextBucket == INT_MAX) {
            cout << "Converged at iteration " << iter << endl;
            break;
        }
        
        currentBucket = nextBucket;
    }

    // Copy results back
    cudaMemcpy(dist_h.data(), d_dist, V * sizeof(float), cudaMemcpyDeviceToHost);

    auto end_total = chrono::high_resolution_clock::now();

    // Output results (commented out for performance testing)
    // cout << "\nDelta-Stepping shortest distances:\n";
    // for (int i = 0; i < V; i++) {
    //     if (dist_h[i] == FLT_MAX)
    //         cout << "Node " << i << " -> unreachable\n";
    //     else
    //         cout << "Node " << i << " -> " << dist_h[i] << "\n";
    // }

    gpuTime = chrono::duration<double, milli>(end_total - start_total).count();
    cout << "\nGPU Execution Time (Delta-Stepping): " << gpuTime << " ms\n";

    // Free GPU memory
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_weights);
    cudaFree(d_dist);
    cudaFree(d_bucket);
    cudaFree(d_nextBucket);
    cudaFree(d_changed);

    cudaDeviceReset();
}
