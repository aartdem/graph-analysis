#include "parent_bfs_lagraph.hpp"
#include "GraphBLAS.h"
#include <LAGraph.h>
#include <chrono>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>

namespace algos {
    void ParentBfsLagraph::load_graph(const std::filesystem::path &file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path.string());
        }

        std::string line;
        // Skip comments
        while (std::getline(file, line)) {
            if (line[0] != '%') break;
        }

        int n_rows, n_cols, nnz;
        std::istringstream iss(line);
        iss >> n_rows >> n_cols >> nnz;
        n = n_rows;

        if (n_rows != n_cols) {
            throw std::runtime_error("Matrix must be square");
        }

        LAGraph_Init(msg);
        GrB_init(GrB_NONBLOCKING);

        GrB_Matrix_new(&matrix, GrB_BOOL, n_rows, n_cols);
        p = std::vector<int>(n, -1);

        for (int i = 0; i < nnz; i++) {
            int row, col;
            file >> row >> col;

            row--;
            col--;

            GrB_Matrix_setElement_BOOL(matrix, true, row, col);
            GrB_Matrix_setElement_BOOL(matrix, true, col, row);
        }
    }

    using clock = std::chrono::steady_clock;

    std::chrono::milliseconds ParentBfsLagraph::compute() {
        auto start = clock::now();
        compute_();
        auto end = clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }

    void ParentBfsLagraph::compute_() {
        GrB_Vector_new(&level, GrB_INT32, n);
        GrB_Vector_new(&parent, GrB_INT32, n);
        LAGraph_New(&G, &matrix, LAGraph_ADJACENCY_UNDIRECTED, msg);

//        LAGraph_Graph_Print(G, static_cast<LAGraph_PrintLevel>(2), stdout, msg);

        for (int i = 0; i < n; i++) {
            if (p[i] == -1) {
                LAGr_BreadthFirstSearch(nullptr, &parent, G, i, msg);
                GrB_Index nvals;
                GrB_Vector_nvals(&nvals, parent);
                auto *indices = static_cast<GrB_Index *>(malloc(nvals * sizeof(GrB_Index)));
                auto *values = static_cast<int32_t *>(malloc(nvals * sizeof(int32_t)));
                GrB_Vector_extractTuples_INT32(indices, values, &nvals, parent);
                for (GrB_Index j = 0; j < nvals; j++) {
                    p[indices[j]] = values[j];
                }
            }
        }

//        LAGraph_Vector_Print(parent, static_cast<LAGraph_PrintLevel>(3), stdout, msg);

        GrB_Matrix_free(&matrix);
        LAGraph_Delete(&G, msg);
        GrB_Vector_free(&level);
        GrB_Vector_free(&parent);

        GrB_finalize();
        LAGraph_Finalize(msg);
    }

    Tree ParentBfsLagraph::get_result() {
        for (int i = 0; i < n; i++) {
            if (p[i] == i) {
                p[i] = -1;
            }
        }

        return Tree{static_cast<uint>(n), p, 0};
    }
}// namespace algos