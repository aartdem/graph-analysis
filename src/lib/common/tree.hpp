#pragma once

#include <vector>

namespace algos {
    struct Tree {
        uint n;
        // parent[i] = -1 if i is root
        std::vector<int> parent;
        uint64_t weight;

        Tree(uint n, std::vector<int> parent, uint64_t w) : n(n), parent(std::move(parent)), weight(w) {}
    };
}// namespace algos