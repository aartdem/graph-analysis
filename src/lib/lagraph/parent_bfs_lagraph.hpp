#pragma once

#include "common/parent_bfs_algorithm.hpp"

#include "GraphBLAS.h"
#include "LAGraph.h"

namespace algos {
    class ParentBfsLagraph : public ParentBfsAlgorithm {

    public:
        void load_graph(const std::filesystem::path &file_path) final;

        std::chrono::milliseconds compute() final;

        Tree get_result() final;

    private:
        void compute_();

        int n = 0;
        char msg[LAGRAPH_MSG_LEN];
        GrB_Vector level = nullptr;
        GrB_Vector parent = nullptr;
        GrB_Matrix matrix = nullptr;
        LAGraph_Graph G = nullptr;
        std::vector<int> p;
    };
}// namespace algos