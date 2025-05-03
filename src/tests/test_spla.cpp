#include <gtest/gtest.h>

#include "spla/using_spla.hpp"

// Demonstrate some basic assertions.
TEST(SplaTest, SplaTest)
{
  EXPECT_EQ(hello_world(), 0);
}
