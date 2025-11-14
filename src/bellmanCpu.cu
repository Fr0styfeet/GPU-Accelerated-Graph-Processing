#include "bellman.h"
#include <limits>
#include <chrono>
#include <vector>
using namespace std;

#define INF 1e9

void bellman::runCPU(int source) {

    int V = graph.rowPtr.size() - 1;     // number of vertices (correct for CSR)
    int E = graph.colInd.size();         // number of edges

    vector<float> dist(V, INF);
    dist[source] = 0;

    auto start = chrono::high_resolution_clock::now();

    // Main Bellman-Ford (V-1 rounds)
    for (int iter = 0; iter < V - 1; iter++) {
        bool changed = false;

        for (int u = 0; u < V; u++) {
            int startEdge = graph.rowPtr[u];
            int endEdge   = graph.rowPtr[u + 1];

            for (int e = startEdge; e < endEdge; e++) {
                int v = graph.colInd[e];
                float w = graph.weights[e];

                if (dist[u] != INF && dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    changed = true;
                }
            }
        }

        if (!changed) break;
    }

    auto end = chrono::high_resolution_clock::now();
    cpuTime = chrono::duration<double, milli>(end - start).count();
    cout << "CPU Execution Time: " << cpuTime << " ms\n";
    
}
