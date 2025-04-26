#include <gtest/gtest.h>

#include "common/sum.hpp"

// Demonstrate some basic assertions.
TEST(SumTest, SumTest)
{
    EXPECT_EQ(sum(1, 2), 3);
}