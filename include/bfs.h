#pragma once
#include "graph.h"
#include <vector>

class BFS {
private:
    CSRGraph graph;
public:
    BFS(const CSRGraph& g);
    void run(int source);
};
