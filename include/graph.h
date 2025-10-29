#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
using namespace std;

struct CSRGraph {
    std::vector<int> rowPtr;  // row pointers (size = V+1)
    std::vector<int> colInd;  // column indices for edges
};

// Load edge list from a file and convert to CSR format
inline CSRGraph loadGraphToCSR(const std::string &filename, int &V) {
    std::vector<std::pair<int,int>> edges;
    std::ifstream fin(filename);

    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    int u, v;
    int maxNode = -1;
    while (fin >> u >> v) {
        edges.push_back({u, v});
        maxNode = std::max({maxNode, u, v});
    }
    fin.close();

    // If V is not set or passed incorrectly, override it
    V = maxNode + 1;

    // Build adjacency list
    std::vector<std::vector<int>> adj(V);
    for (auto &e : edges){
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }

    // Build CSR
    CSRGraph g;
    g.rowPtr.resize(V + 1, 0);
    for (int i = 0; i < V; ++i)
        g.rowPtr[i + 1] = g.rowPtr[i] + adj[i].size();

    for (int i = 0; i < V; ++i)
        for (int nei : adj[i])
            g.colInd.push_back(nei);

    for(auto &i:g.rowPtr) cout<<i<<" ";
    cout<<endl;

    for(auto &i:g.colInd) cout<<i<<" ";
    cout<<endl;
    
    return g;
}
