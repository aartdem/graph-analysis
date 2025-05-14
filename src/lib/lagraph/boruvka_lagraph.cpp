#include "boruvka_lagraph.hpp"

#include <LAGraph.h>
#include <LAGraphX.h>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace algos {

    using clock = std::chrono::steady_clock;

    void BoruvkaLagraph::load_graph(const std::filesystem::path &file_path) {
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

        if (n_rows != n_cols) {
            throw std::runtime_error("Matrix must be square");
        }

        GrB_init(GrB_NONBLOCKING);

        GrB_Matrix_new(&matrix, GrB_UINT64, n_rows, n_cols);

        for (int i = 0; i < nnz; i++) {
            int row, col;
            uint64_t weight;
            file >> row >> col >> weight;

            row--;
            col--;

            GrB_Matrix_setElement_UINT64(matrix, weight, row, col);
            GrB_Matrix_setElement_UINT64(matrix, weight, col, row);
        }
        num_vertices = n_rows;
    }

    std::chrono::milliseconds BoruvkaLagraph::compute() {
        const auto start = clock::now();
        compute_();
        const auto end = clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }

    void BoruvkaLagraph::compute_() {
        char msg[256];
        LAGraph_msf(&mst_matrix, matrix, false, msg);

        GrB_Matrix_reduce_UINT64(
            &weight,
            GrB_NULL,
            GrB_PLUS_MONOID_UINT64,
            mst_matrix,
            GrB_NULL
        );

        //GrB_Matrix_free(&mst_matrix);
    }

    std::vector<int> extract_parent(GrB_Matrix T, int n) {
        std::vector<int> parent(n, -1);

        GrB_Index nvals;
        GrB_Matrix_nvals(&nvals, T);

        std::vector<GrB_Index> I(nvals), J(nvals);
        std::vector<uint64_t> X(nvals);
        GrB_Matrix_extractTuples_UINT64(I.data(), J.data(), X.data(), &nvals, T);

        std::vector<std::vector<int>> adj(n);
        for (GrB_Index k = 0; k < nvals; k++) {
            int i = I[k], j = J[k];
            adj[i].push_back(j);
            adj[j].push_back(i);
        }

        std::vector<bool> visited(n, false);
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                std::function<void(int, int)> dfs = [&](int v, int p) {
                    visited[v] = true;
                    parent[v] = p;
                    for (int u : adj[v]) {
                        if (!visited[u]) {
                            dfs(u, v);
                        }
                    }
                };
                dfs(i, -1);
            }
        }

        return parent;
    }

    Tree BoruvkaLagraph::get_result() {
        auto parent = extract_parent(mst_matrix, num_vertices);
        GrB_Matrix_free(&mst_matrix);
        return Tree{num_vertices, parent, weight};
    }
}