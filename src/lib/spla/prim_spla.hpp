#pragma once

#include <spla.hpp>
#include "common/mst_algorithm.hpp"
#include <chrono>
#include <set>

namespace algos {
    class PrimSpla : public MstAlgorithm {

    public:
        void load_graph(const std::filesystem::path &file_path) final;

        std::chrono::milliseconds compute() final;

        Tree get_result() final;

    private:
        void compute_();

        void print_vector(const spla::ref_ptr<spla::Vector> &v, const std::string &name = "");

        void update(std::set<std::pair<unsigned int , unsigned int>> &s, const spla::ref_ptr<spla::Vector> &v) ;

        void log(const std::string& t);

        using clock = std::chrono::steady_clock;

        uint n;
        int edges_count;
        bool enabled_log = false;
        const unsigned int INF = UINT32_MAX;
        unsigned long long weight = 0;
        spla::ref_ptr<spla::Matrix> a;
        spla::ref_ptr<spla::Vector> mst;
        std::vector<unsigned int> buffer1;
        std::vector<unsigned int> buffer2;
        spla::ref_ptr<spla::Scalar> inf_uint = spla::Scalar::make_uint(INF);
        spla::ref_ptr<spla::Scalar> zero_uint = spla::Scalar::make_uint(0);
        std::chrono::steady_clock::time_point last_time = clock::now();
    };
}