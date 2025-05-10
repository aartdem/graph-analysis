#include <filesystem>
#include "spla/prim_spla.hpp"

int main() {
    auto file = std::filesystem::path(DATA_DIR) / "test1.mtx";
    auto algo = std::make_unique<algos::PrimSpla>();
    algo->load_graph(file);
    auto t = algo->compute();
    std:: cout << t.count() << '\n';
    auto res = algo->get_result();
    std::cout << "weight: " << res.weight << '\n';
    return 0;
}