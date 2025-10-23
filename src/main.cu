#include <iostream>
#include <string>
#include "../include/graph.h"
#include "../include/utils.h"
// #include "./bfs.cu"
#include "bfs.h"
// #include "sssp.h"
// #include "pagerank.h"
// #include "connected_components.h"
// #include "utils.h"
using namespace std;

void showMenu() {
    cout << "\n============================================\n";
    cout << "  CUDA Accelerated Graph Processing System\n";
    cout << "============================================\n";
    cout << "1. Breadth First Search (BFS)\n";
    cout << "2. Single src Shortest Path (SSSP)\n";
    cout << "3. PageRank\n";
    cout << "4. Connected Components\n";
    cout << "5. Exit\n";
    cout << "--------------------------------------------\n";
    cout << "Enter your choice: ";
}

int main() {
    string file;
    int choice, V;

    while (true) {
        showMenu();
        cin >> choice;
        if (choice == 5) {
            cout << "Exiting...\n";
            break;
        }

        switch (choice) {
        case 1: {
            file = "../data/facebook_combined.txt";
            ifstream fin(file);
            // fin >> V ;
            V = 4039 ;
            CSRGraph g = loadGraphToCSR(file, V);
            cout<<V<<endl;
            fin.close();

            int src;
            cout << "Enter src vertex: "; cin >> src;

            cout << "\nBFS execution on GPU \n";
            double start = getTime();

            bfsGPU(g, src);

            double end = getTime();
            cout << "Execution time: " << (end - start) << " ms\n";
            break;
        }

        case 2: {
            file = "delivery_dataset.txt";
            CSRGraph g = loadGraphToCSR(file, V);
            ifstream fin(file);
            fin >> V ;
            fin.close();

            int src;
            cout << "Enter src vertex: "; cin >> src;
            cout << "\nSSSP execution on GPU \n";

            double start = getTime();

            //ssspGPU(g, src);

            double end = getTime();
            cout << "Execution time: " << (end - start) << " ms\n";
            break;
        }

        case 3: {
            file = "web-Google.txt";
            CSRGraph g = loadGraphToCSR(file, V);
            ifstream fin(file);
            fin >> V ;
            fin.close();

            float damping = 0.85f, epsilon = 1e-6;
            cout << "\nPageRank execution on GPU \n";

            double start = getTime();

            //pagerankGPU(g, damping, epsilon);

            double end = getTime();
            cout << "Execution time: " << (end - start) << " ms\n";
            break;
        }
        
        case 4: {
            file = "email-Eu-core.txt";
            CSRGraph g = loadGraphToCSR(file, V);
            ifstream fin(file);
            fin >> V ;
            fin.close();

            cout << "\nFinding Connected Components execution on GPU \n";

            double start = getTime();

            //connectedComponentsGPU(g);

            double end = getTime();
            cout << "Execution time: " << (end - start) << " ms\n";
            break;
        }

        default:
            cout << "Invalid choice. Try again.\n";
            break;
        }
    }

    return 0;
}
