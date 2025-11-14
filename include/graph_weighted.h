#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <tuple>
#include <omp.h>

struct CSRGraphWeighted {
    std::vector<int> rowPtr;
    std::vector<int> colInd;
    std::vector<float> weights;
};

inline CSRGraphWeighted loadWeightedGraphToCSR(const std::string &filename, int &V) 
{
    std::vector<std::tuple<int,int,float>> edges;
    std::ifstream fin(filename);

    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    int u, v;
    float w;
    int maxNode = -1;

    // File reading must remain serial (I/O bottleneck anyway)
    while (fin >> u >> v >> w) {
        edges.push_back({u, v, w});
        maxNode = std::max({maxNode, u, v});
    }
    fin.close();

    V = maxNode + 1;

    // Build adjacency list (simple)
    std::vector<std::vector<std::pair<int,float>>> adj(V);

    // This loop CAN be parallelized because each index writes to adj[u] only
    #pragma omp parallel for
    for (int i = 0; i < edges.size(); i++) {
        auto &e = edges[i];
        int u = std::get<0>(e);

        // Each adj[u] is a separate vector → safe push_back
        #pragma omp critical
        adj[u].push_back({ std::get<1>(e), std::get<2>(e) });
    }

    // Build CSR rowPtr
    CSRGraphWeighted g;
    g.rowPtr.resize(V + 1);

    g.rowPtr[0] = 0;

    for (int i = 0; i < V; i++)
        g.rowPtr[i + 1] = g.rowPtr[i] + adj[i].size();

    int E = g.rowPtr[V];
    g.colInd.resize(E);
    g.weights.resize(E);

    // Fill CSR data (parallel)
    #pragma omp parallel for
    for (int u = 0; u < V; u++) {
        int index = g.rowPtr[u];
        for (auto &p : adj[u]) {
            g.colInd[index] = p.first;
            g.weights[index] = p.second;
            index++;
        }
    }

    return g;
}
