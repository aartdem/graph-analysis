#pragma once
#include "common/mst_algorithm.hpp"

namespace algos {
    class BoruvkaLagraph : public MstAlgorithm {

    public:
        void load_graph(const std::filesystem::path &file_path) final;

        std::chrono::milliseconds compute() final;

        Tree get_result() final;

    private:
        void compute_();

        std::unique_ptr<Tree> tree = nullptr;
        uint64_t weight = 0;
    };
}// namespace algos
