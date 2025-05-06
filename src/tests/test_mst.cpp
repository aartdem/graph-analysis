#include <gtest/gtest.h>
#include <filesystem>
#include "common/mst_algorithm.hpp"
#include "spla/prim_spla.hpp"

namespace tests
{
  struct GraphCase
  {
    std::string filename;
    double expected_weight;
  };

  template <class T>
  algos::MstAlgorithm *create_mst_algo();

  template <>
  algos::MstAlgorithm *create_mst_algo<algos::PrimSpla>()
  {
    return new algos::PrimSpla();
  }

  template <typename T>
  class MstAlgorithmTest : public ::testing::Test
  {
  protected:
    MstAlgorithmTest() : algo(create_mst_algo<T>()) {}

    ~MstAlgorithmTest() override { delete algo; }

    algos::MstAlgorithm *const algo;
  };

  using AlgosTypes = ::testing::Types<algos::PrimSpla>; // extend this with other MST algorimths
  TYPED_TEST_SUITE(MstAlgorithmTest, AlgosTypes);

  static const GraphCase mst_test_cases[] = {
      {"empty.mtx", 0},
      {"small.mtx", 0}};

  TYPED_TEST(MstAlgorithmTest, IsCorrectMst)
  {
    for (const GraphCase &test_case : mst_test_cases)
    {
      auto file = std::filesystem::path(DATA_DIR) / test_case.filename;
      this->algo->load_graph(file);
    }
  }
}
