#pragma once

#include "common/parent_bfs_algorithm.hpp"
#include <chrono>
#include <set>
#include <spla.hpp>

namespace algos {
    class ParentBfsSpla : public ParentBfsAlgorithm {

    public:
        void load_graph(const std::filesystem::path &file_path) final;

        std::chrono::milliseconds compute() final;

        Tree get_result() final;

    private:
        void compute_();

        void print_vector(const spla::ref_ptr<spla::Vector> &v, const std::string &name = "");

        int n;
        int edges_count;
        spla::ref_ptr<spla::Matrix> a;
        spla::ref_ptr<spla::Vector> parent;
        std::vector<int> buffer1;
        std::vector<int> buffer2;
        spla::ref_ptr<spla::Scalar> zero_int = spla::Scalar::make_int(0);
        spla::ref_ptr<spla::Scalar> one_int = spla::Scalar::make_int(1);
    };
}// namespace algos