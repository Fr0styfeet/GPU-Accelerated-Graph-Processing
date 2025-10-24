// dijkstra.h
#pragma once
#include "graph_weighted.h"

class Dijkstra {
private:
    CSRGraphWeighted graph;
public:
    Dijkstra(const CSRGraphWeighted& g);
    void run(int source);
};
