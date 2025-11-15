#include <iostream>
#include <chrono>
#include "graph_weighted.h"
#include "bellman.h"

using namespace std;

int main() {
    while (true) {
        cout << "\nB E L L M A N - F O R D   C O M P A R I S O N\n";
        cout << "1. Run Bellman-Ford on CPU (large dataset) \n";
        cout << "2. Run Bellman-Ford on GPU (large dataset)\n";
        cout << "3. Run Bellman-Ford on CPU (small dataset)\n";
        cout << "4. Run Bellman-Ford on GPU (small dataset)\n";
        cout << "5. Exit\n";
        cout << "Enter your choice: ";

        int choice; cin >> choice;
        
        if (choice == 5) break;

        int src, V;
        string file = "./data/large_weight.txt"; 

        cout << "Enter source vertex: "; cin >> src;

        CSRGraphWeighted g = CSR_conversion(file, V);
        bellman bf(g);

        if (choice == 1) {

            cout << "\nRunning Bellman-Ford on CPU (large dataset)...\n";

            bf.runCPU(src);

        }
        else if (choice == 2) {

            cout << "\nRunning Bellman-Ford on GPU (large dataset)...\n";
            bf.runGPU(src);

        }
        else if (choice == 3) {

            file = "./data/delivery_dataset.txt"; 

            CSRGraphWeighted g = CSR_conversion(file, V);
            bellman bf2(g);
            
            cout << "\nRunning Bellman-Ford on CPU (small dataset)...\n";
            bf2.runCPU(src);

        }
        else if (choice == 4) {
            file = "./data/delivery_dataset.txt"; 

            CSRGraphWeighted g = CSR_conversion(file, V);
            bellman bf2(g);
            
            cout << "\nRunning Bellman-Ford on GPU (small dataset)...\n";
            bf2.runGPU(src);

        }
        else {
            cout << "Invalid choice!\n";
        }
    }

    return 0;
}
