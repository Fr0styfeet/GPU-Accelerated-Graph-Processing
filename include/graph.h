#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace std;

//Compressed Sparse Row Graph
struct CSRGraph {
    vector<int> rowPtr;
    vector<int> colInd;
};


//Load edge list file to CSR
inline CSRGraph loadGraphToCSR(const string &filename, int &V) {
    vector<pair<int,int>> edges;
    ifstream fin(filename);

    if (!fin.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(1);
    }

    int u, v;
    V = 0;
    while (fin >> u >> v) {
        edges.push_back({u, v});
        V = max(V, max(u, v));
    }
    fin.close();
    V += 1; // vertices are 0-indexed


    //adjacency list
    vector<vector<int>> adj(V);
    for (auto &e : edges) adj[e.first].push_back(e.second);

    //CSR
    CSRGraph g;
    g.rowPtr.resize(V + 1, 0);
    for (int i = 0; i < V; ++i){
        g.rowPtr[i + 1] = g.rowPtr[i] + adj[i].size();
    }

    for (int i = 0; i < V; ++i){
        for (int nei : adj[i]){
            g.colInd.push_back(nei);
        }
    }

    return g;
}
