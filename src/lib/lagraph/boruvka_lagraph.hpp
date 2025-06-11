#pragma once
#include "common/mst_algorithm.hpp"

#include "GraphBLAS.h"
#include <LAGraph.h>

namespace algos {
    class BoruvkaLagraph : public MstAlgorithm {

    public:
        void load_graph(const std::filesystem::path &file_path) final;

        std::chrono::milliseconds compute() final;

        Tree get_result() final;

    private:
        void compute_();

        char msg[LAGRAPH_MSG_LEN];
        uint64_t weight = 0;
        GrB_Matrix matrix = nullptr;
        GrB_Matrix mst_matrix = nullptr;
        uint num_vertices = 0;
    };
}// namespace algos
