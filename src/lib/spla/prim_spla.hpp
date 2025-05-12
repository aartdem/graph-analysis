#pragma once

#include <spla.hpp>
#include "common/mst_algorithm.hpp"
#include <chrono>

namespace algos {
    class PrimSpla : public MstAlgorithm {

    public:
        void load_graph(const std::filesystem::path &file_path) final;

        std::chrono::milliseconds compute() final;

        Tree get_result() final;

    private:
        void compute_();

        [[nodiscard]] std::pair<float, int> get_min_with_arg(const spla::ref_ptr<spla::Vector> &vec) const;

//        void copy_vector(const spla::ref_ptr<spla::Vector> &from, const spla::ref_ptr<spla::Vector> &to);

        void print_vector(const spla::ref_ptr<spla::Vector> &vec, const std::string &name);

        void log(const std::string& t);
        using clock = std::chrono::steady_clock;

        const float INF = 1e18;
        int n;
        int edges_count;
        uint32_t weight = 0;
        spla::ref_ptr<spla::Matrix> a;
        spla::ref_ptr<spla::Vector> mst;
        std::vector<int> buffer_int;
        std::vector<float> buffer_float;
        spla::ref_ptr<spla::Vector> visited;
        spla::ref_ptr<spla::Vector> zero_vec;
        spla::ref_ptr<spla::Scalar> neg_one_float = spla::Scalar::make_float(-1);
        spla::ref_ptr<spla::Scalar> inf_float = spla::Scalar::make_float(INF);
        spla::ref_ptr<spla::Scalar> zero_float = spla::Scalar::make_float(0);
        std::chrono::steady_clock::time_point last_time = clock::now();
    };
}