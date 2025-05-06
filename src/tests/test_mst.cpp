#include <gtest/gtest.h>
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
  algos::MstAlgorithm *create_algo();

  template <>
  algos::MstAlgorithm *create_algo<algos::PrimSpla>()
  {
    return new algos::PrimSpla();
  }

  template <typename T>
  class MstAlgorithmTest : public ::testing::Test
  {
  protected:
    MstAlgorithmTest() : algo(create_algo<T>()) {}

    ~MstAlgorithmTest() override { delete algo; }

    algos::MstAlgorithm *const algo;
  };

  using AlgosTypes = ::testing::Types<algos::PrimSpla>; // extend this with outher MST algorimths

  TYPED_TEST_SUITE(MstAlgorithmTest, AlgosTypes);

  TYPED_TEST(MstAlgorithmTest, IsCorrectMst)
  {
    // Implement test logic here
  }
}
