#pragma once

#include <spla.hpp>
#include "common/mst_algorithm.hpp"

namespace algos {
    class BoruvkaSpla : public MstAlgorithm {

    public:
        void load_graph(const std::filesystem::path &file_path) final;

        std::chrono::seconds compute() final;

        Tree get_result() final;

    private:
        void compute_();

        std::unique_ptr<Tree> tree = nullptr;
        int n;
        int edges;
        uint64_t weight = 0;
        spla::ref_ptr<spla::Matrix> a;
        spla::ref_ptr<spla::Vector> mst;
        std::vector<int> buffer_int;
        std::vector<float> buffer_float;
    };
}
