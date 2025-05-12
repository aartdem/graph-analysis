#include <iostream>
#include "gunrock/boruvka.hxx"

int main() {
    auto file = "data/test1.mtx";
    auto algo = std::make_unique<algos::BoruvkaGunrock>();
    algo->load_graph(file);
    auto t = algo->compute();
    std:: cout << t.count() << '\n';
    auto res = algo->get_result();
    std::cout << "weight: " << res.weight << '\n';
    return 0;
}