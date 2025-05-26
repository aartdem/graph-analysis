#include <filesystem>
#include <gtest/gtest.h>

#include "common/mst_algorithm.hpp"
#include "lagraph/boruvka_lagraph.hpp"
#include "spla/boruvka_spla.hpp"
#include "spla/prim_spla.hpp"

namespace tests {
    struct GraphCase {
        std::string filename;
        double expected_weight;
    };

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

    template<class T>
    algos::MstAlgorithm *create_mst_algo();// implement parametrized function for each MST algorithm

    template<>
    algos::MstAlgorithm *create_mst_algo<algos::PrimSpla>() {
        return new algos::PrimSpla();
    }


    template<>
    algos::MstAlgorithm *create_mst_algo<algos::BoruvkaSpla>() {
        return new algos::BoruvkaSpla();
    }

    template<>
    algos::MstAlgorithm *create_mst_algo<algos::BoruvkaLagraph>() {
        return new algos::BoruvkaLagraph();
    }

    template<typename T>
    class MstAlgorithmTest : public ::testing::Test {
    protected:
        MstAlgorithmTest() : algo(create_mst_algo<T>()) {}

        ~MstAlgorithmTest() override { delete algo; }

        algos::MstAlgorithm *const algo;
    };

    using AlgosTypes = ::testing::Types<algos::BoruvkaSpla, algos::PrimSpla, algos::BoruvkaLagraph>;// extend this with other MST algorimths
    TYPED_TEST_SUITE(MstAlgorithmTest, AlgosTypes);

    static const GraphCase mst_test_cases[] = {
            {"point.mtx", 0},
            {"one_edge.mtx", 2},
            {"test1.mtx", 22},
            {"small.mtx", 120},
            {"Trefethen_2000.mtx", 1999},
            {"two_components_int.mtx", 15}};

    TYPED_TEST(MstAlgorithmTest, IsCorrectMst) {
        for (const GraphCase &test_case: mst_test_cases) {
            auto file = std::filesystem::path(DATA_DIR) / test_case.filename;
            std::cout << test_case.filename << std::endl;
            this->algo->load_graph(file);
            this->algo->compute();
            auto res = this->algo->get_result();
            ASSERT_FLOAT_EQ(test_case.expected_weight, res.weight);
            ASSERT_TRUE(is_tree_or_forest(res.parent));
        }
    }
}// namespace tests
