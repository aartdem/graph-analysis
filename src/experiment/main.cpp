#include <filesystem>
#include "spla/prim_spla.hpp"
#include "spla/boruvka_spla.hpp"
#include "spla/library_spla.hpp"

int main() {
//    auto file_unweighted = std::filesystem::path(DATA_DIR) / "unweighted.mtx";
    auto file = std::filesystem::path(DATA_DIR) / "test1.mtx";
    auto algo = std::make_unique<algos::BoruvkaSpla>();
    algos::initialize_spla();
    algo->load_graph(file);
    auto t = algo->compute();
    std::cout << "time: " << t.count() << '\n';
    auto res = algo->get_result();
    std::cout << "weight: " << res.weight << '\n';
    algos::finalize_spla();
}
