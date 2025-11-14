#include <iostream>
#include <chrono>
#include "graph_weighted.h"
#include "bellman.h"

using namespace std;
using namespace std::chrono;

int main() {
    while (true) {
        cout << "\nB E L L M A N - F O R D   C O M P A R I S O N\n";
        cout << "1. Run Bellman-Ford on CPU\n";
        cout << "2. Run Bellman-Ford on GPU\n";
        cout << "4. Exit\n";
        cout << "Enter your choice: ";

        int choice;
        cin >> choice;
        if (choice == 4) break;

        int src, V;
        string file = "./data/large_weight.txt"; 

        cout << "Enter source vertex: ";
        cin >> src;

        CSRGraphWeighted g = loadWeightedGraphToCSR(file, V);
        bellman bf(g);

        if (choice == 1) {

            cout << "\nRunning Bellman-Ford on CPU...\n";

            bf.runCPU(src);

        }
        else if (choice == 2) {

            cout << "\nRunning Bellman-Ford on GPU...\n";
            bf.runGPU(src);

        }
        else {
            cout << "Invalid choice!\n";
        }
    }

    return 0;
}
