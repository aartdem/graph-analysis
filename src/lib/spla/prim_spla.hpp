#pragma once

#include <spla.hpp>
#include "common/mst_algorithm.hpp"

namespace algos
{
    class PrimSpla : public MstAlgorithm
    {

    public:
        void load_graph(const std::filesystem::path &file_path) override final;

        std::chrono::duration<double> compute() override final;

        Tree get_result() override final;

    private:
        std::unique_ptr<spla::MtxLoader> loader = std::make_unique<spla::MtxLoader>(spla::MtxLoader());
    };

};
