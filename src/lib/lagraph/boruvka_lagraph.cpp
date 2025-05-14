#include "boruvka_lagraph.hpp"

#include <LAGraph.h>
#include <LAGraphX.h>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
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

            GrB_Matrix_setElement_UINT32(matrix, row, col, weight);
            GrB_Matrix_setElement_UINT32(matrix, col, row, weight);
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
        GrB_Matrix mst_matrix;

        char msg[256];
        LAGraph_msf(&mst_matrix, matrix, true, msg);

        GrB_Matrix_reduce_UINT64(
            &weight,
            GrB_NULL,
            GrB_PLUS_MONOID_UINT64,
            mst_matrix,
            GrB_NULL
        );

        GrB_Matrix_free(&mst_matrix);
    }

    Tree BoruvkaLagraph::get_result() {
        if (!tree) {
            throw std::runtime_error("Computation not done yet");
        }
        return *tree;
    }
}