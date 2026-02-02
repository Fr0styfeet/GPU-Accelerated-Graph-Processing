#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <omp.h>

using namespace std;

struct Edge {
    int u;
    int v;
    float w;
};

struct CSRGraphWeighted {
    vector<int> rowPtr;
    vector<int> colInd;
    vector<float> weights;
};

inline CSRGraphWeighted CSR_conversion(const string &filename, int &V) 
{
    vector<Edge> edges;   // using struct Edge
    ifstream fin(filename);

    if (!fin.is_open()) {
        cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    int u, v;
    float w;
    int maxNode = -1;

    // File reading 
    while (fin >> u >> v >> w) {
        edges.push_back({u, v, w});
        maxNode = max({maxNode, u, v});
    }
    fin.close();

    V = maxNode + 1;

    // Build adjacency list
    vector<vector<pair<int,float>>> adj(V);

    #pragma omp parallel for
    for (int i = 0; i < edges.size(); i++) {
        int a = edges[i].u;

        // critical section
        #pragma omp critical
        adj[a].push_back({ edges[i].v, edges[i].w });
    }

    // Building CSR rowPtr
    CSRGraphWeighted g;
    g.rowPtr.resize(V + 1);
    g.rowPtr[0] = 0;

    for (int i = 0; i < V; i++){
        g.rowPtr[i + 1] = g.rowPtr[i] + adj[i].size();
    }

    int E = g.rowPtr[V];
    g.colInd.resize(E);
    g.weights.resize(E);

    // Filling CSR arrays 
    #pragma omp parallel for
    for (int u = 0; u < V; u++) {
        int index = g.rowPtr[u];
        for (auto &p:adj[u]) {
            g.colInd[index] = p.first;
            g.weights[index] = p.second;
            index++;
        }
    }

    return g;
}