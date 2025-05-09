#pragma once

namespace algos {
    struct Tree {
        int n;
        std::vector<int> parent;
        int weight;

        Tree(int n, std::vector<int> parent, int w) : n(n), parent(std::move(parent)), weight(w) {}
    };
}