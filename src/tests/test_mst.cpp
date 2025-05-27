#include <filesystem>
#include <gtest/gtest.h>

#include "common/mst_algorithm.hpp"
#include "lagraph/boruvka_lagraph.hpp"
#include "spla/boruvka_spla.hpp"
#include "spla/prim_spla.hpp"
#include "test_commons.hpp"
#include <filesystem>
#include <gtest/gtest.h>

namespace tests {
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
            ASSERT_EQ(test_case.expected_weight, res.weight);
            ASSERT_TRUE(is_tree_or_forest(res.parent));
        }
    }
}// namespace tests
