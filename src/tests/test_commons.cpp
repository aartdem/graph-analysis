#include <string>
#include <vector>

namespace tests {
    bool has_cycle(int v, int p, const std::vector<std::vector<int>> &g, std::vector<bool> &visited) {
        visited[v] = true;
        bool acc = false;
        for (auto u: g[v]) {
            if (u == p) {
                continue;
            }
            if (visited[u]) {
                return true;
            }
            acc |= has_cycle(u, v, g, visited);
        }
        return acc;
    };

    bool is_tree_or_forest(const std::vector<int> &parent) {
        int n = int(parent.size());
        std::vector visited(n, false);
        std::vector<std::vector<int>> g(n);
        for (int i = 0; i < n; ++i) {
            if (parent[i] > n) {
                return false;
            }
            if (parent[i] != -1) {
                g[i].push_back(parent[i]);
                g[parent[i]].push_back(i);
            }
        }
        bool has_cycle_acc = false;
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                has_cycle_acc |= has_cycle(i, -1, g, visited);
            }
        }

        return !has_cycle_acc;
    }
}// namespace tests