#include "common/parent_bfs_algorithm.hpp"
#include "spla/parent_bfs_spla.hpp"
#include "test_commons.hpp"
#include <filesystem>
#include <gtest/gtest.h>

namespace tests {
    template<class T>
    algos::ParentBfsAlgorithm *create_bfs_algo();// implement parametrized function for each MST algorithm

    template<>
    algos::ParentBfsAlgorithm *create_bfs_algo<algos::ParentBfsSpla>() {
        return new algos::ParentBfsSpla();
    }

    template<typename T>
    class BfsAlgorithmTest : public ::testing::Test {
    protected:
        BfsAlgorithmTest() : algo(create_bfs_algo<T>()) {}

        ~BfsAlgorithmTest() override { delete algo; }

        algos::ParentBfsAlgorithm *const algo;
    };

    using AlgosTypes = ::testing::Types<algos::ParentBfsSpla>;// extend this with other MST algorimths
    TYPED_TEST_SUITE(BfsAlgorithmTest, AlgosTypes);

    static const GraphCase mst_test_cases[] = {
            {"test1_unweighted.mtx"},
            {"small_unweighted.mtx"}};

    TYPED_TEST(BfsAlgorithmTest, IsCorrectParentTree) {
        for (const GraphCase &test_case: mst_test_cases) {
            auto file = std::filesystem::path(DATA_DIR) / test_case.filename;
            std::cout << test_case.filename << std::endl;
            this->algo->load_graph(file);
            this->algo->compute();
            auto res = this->algo->get_result();
            ASSERT_TRUE(is_tree_or_forest(res.parent));
        }
    }
}// namespace tests