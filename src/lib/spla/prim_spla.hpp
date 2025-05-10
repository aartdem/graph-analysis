#pragma once

#include <spla.hpp>
#include "common/mst_algorithm.hpp"

namespace algos {
    class PrimSpla : public MstAlgorithm {

    public:
        void load_graph(const std::filesystem::path &file_path) final;

        std::chrono::seconds compute() final;

        Tree get_result() final;

    private:
        void compute_();

        std::pair<int, int> get_min_with_arg(const spla::ref_ptr<spla::Vector> &vec);

        void copy_vector(const spla::ref_ptr<spla::Vector> &from, const spla::ref_ptr<spla::Vector> &to);

        void print_vector(const spla::ref_ptr<spla::Vector> &vec, const std::string &name);

        const int INF = 1e9;
        int n;
        int edges;
        int weight = 0;
        spla::ref_ptr<spla::Matrix> a;
        spla::ref_ptr<spla::Vector> mst;
        std::vector<int> buffer_int1;
        std::vector<int> buffer_int2;
    };
}
