#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <tuple>

struct CSRGraphWeighted {
    std::vector<int> rowPtr;        // row pointers (size = V+1)
    std::vector<int> colInd;        // column indices
    std::vector<float> weights;     // edge weights corresponding to colInd
};

// Load weighted edge list into CSR format
inline CSRGraphWeighted loadWeightedGraphToCSR(const std::string &filename, int &V) {
    std::vector<std::tuple<int,int,float>> edges;
    std::ifstream fin(filename);

    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    int u, v;
    float w;
    int maxNode = -1;

    while (fin >> u >> v >> w) {
        edges.push_back(std::make_tuple(u, v, w));
        maxNode = std::max({maxNode, u, v});
    }
    fin.close();

    // Compute number of vertices
    V = maxNode + 1;

    // Build adjacency list
    std::vector<std::vector<std::pair<int,float>>> adj(V);
    for (auto &e : edges) {
        std::tie(u, v, w) = e;
        adj[u].push_back({v, w});
    }

    // Build CSR
    CSRGraphWeighted g;
    g.rowPtr.resize(V + 1, 0);
    for (int i = 0; i < V; ++i)
        g.rowPtr[i + 1] = g.rowPtr[i] + adj[i].size();

    for (int i = 0; i < V; ++i)
        for (auto &p : adj[i]) {
            g.colInd.push_back(p.first);
            g.weights.push_back(p.second);
        }

    return g;
}
