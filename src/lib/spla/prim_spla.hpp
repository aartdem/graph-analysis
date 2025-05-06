#pragma once

#include <spla.hpp>
#include "common/mst_algorithm.hpp"

namespace algos
{
    class PrimSpla : public MstAlgorithm
    {
    public:
        PrimSpla() {}

        void load_graph(std::string file_path) override final;

        std::chrono::duration<double> compute() override final;

        Tree get_result() override final;

        ~PrimSpla() override final
        {
            loader.~MtxLoader();
            return;
        }

    private:
        spla::MtxLoader loader = spla::MtxLoader();
    };

};
