#include <spla.hpp>
#include "common/tree.hpp"
#include "prim_spla.hpp"

namespace algos
{
    void PrimSpla::load_graph(const std::filesystem::path &file_path)
    {
        if (!loader->load(file_path, true, false, false))
        {
            throw std::runtime_error("Can not load graph from file");
        }
        return;
    }
    std::chrono::duration<double> PrimSpla::compute()
    {
        // TODO
        return std::chrono::duration<double>(0);
    }
    Tree PrimSpla::get_result()
    {
        // TODO
        return Tree();
    }
};
