#include <filesystem>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "spla/prim_spla.hpp"

using namespace std;
using namespace algos;

int main() {
    auto file = std::filesystem::path(DATA_DIR) / "ca_coauth_weighted.mtx";
    auto algo = std::make_unique<algos::PrimSpla>();
    algo->load_graph(file);
    auto t = algo->compute();
    std::cout << "time: " << t.count() << '\n';
    auto res = algo->get_result();
    std::cout << "weight: " << res.weight << '\n';

    return 0;
}
