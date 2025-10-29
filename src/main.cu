#include <iostream>
#include "graph.h"
#include "graph_weighted.h"
#include "bfs.h"
#include "dijkstra.h"

using namespace std;

int main() {
    int choice;
    while(true) {
        cout << "1. BFS\n2. Dijkstra\n3. Exit\nChoice: ";
        cin >> choice;
        if(choice==3) break;

        int V, src;

        cout << "Enter source vertex: ";
        cin >> src;

        if(choice==1) {
            string file ="./data/facebook_combined.txt";
            V=4039; 
            CSRGraph g = loadGraphToCSR(file, V);
            BFS bfs(g);
            bfs.run(src);
        } else if(choice==2) {
            string file ="./data/weighted.txt";
            V= 4;
            CSRGraphWeighted g = loadWeightedGraphToCSR(file, V);
            Dijkstra dj(g);
            dj.run(src);
        }
    }
}
