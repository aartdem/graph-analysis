#pragma once

#include "common/mst_algorithm.hpp"
#include <spla.hpp>

namespace algos {
    class BoruvkaSpla : public MstAlgorithm {

    public:
        void load_graph(const std::filesystem::path &file_path) final;

        std::chrono::milliseconds compute() final;

        Tree get_result() final;

    private:
        void compute_();

        std::unique_ptr<Tree> tree = nullptr;
        uint n;
        int edges;
        uint64_t weight = 0;
        spla::ref_ptr<spla::Matrix> a;
        spla::ref_ptr<spla::Vector> mst;
        std::vector<std::vector<uint32_t>> adj_list;
        std::vector<int> buffer_int;
        std::vector<float> buffer_float;
    };
}// namespace algos
