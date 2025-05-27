#pragma once

namespace tests {
    struct GraphCase {
        std::string filename;
        uint64_t expected_weight = 0;
    };

    bool has_cycle(int v, int p, const std::vector<std::vector<int>> &g, std::vector<bool> &visited);

    bool is_tree_or_forest(const std::vector<int> &parent);
}// namespace tests