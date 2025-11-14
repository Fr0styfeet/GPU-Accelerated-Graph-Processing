#include <omp.h>
#include <vector>

void buildCSR(int n, int m, float **A,std::vector<int> &row_ptr,std::vector<int> &col_idx,std::vector<float> &val)
{
    std::vector<int> row_count(n, 0);

    // PASS 1: Count non-zero entries
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int cnt = 0;
        for (int j = 0; j < m; j++)
            if (A[i][j] != 0) cnt++;

        row_count[i] = cnt;
    }

    // PASS 2: Build row_ptr (serial)
    row_ptr.resize(n + 1);
    row_ptr[0] = 0;
    for (int i = 0; i < n; i++)
        row_ptr[i + 1] = row_ptr[i] + row_count[i];

    int nnz = row_ptr[n];
    col_idx.resize(nnz);
    val.resize(nnz);

    // PASS 3: Fill CSR data
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int index = row_ptr[i];  // private to each thread; safe

        for (int j = 0; j < m; j++) {
            if (A[i][j] != 0) {
                col_idx[index] = j;
                val[index] = A[i][j];
                index++;
            }
        }
    }
}
