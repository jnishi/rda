#include <gtest/gtest.h>
#include "RDA.hpp"

TEST(RDA, readVector)
{
  std::string line = "1 0:1.0 201:2.2 744:-0.3 15:3.0";
  SparseVector expectedVector;
  expectedVector.push_back(std::make_pair<int, float>(0,1.0));
  expectedVector.push_back(std::make_pair<int, float>(15,3.0));
  expectedVector.push_back(std::make_pair<int, float>(201,2.2));
  expectedVector.push_back(std::make_pair<int, float>(744,-0.3));
  TrainData expected = make_pair(POSITIVE, expectedVector);

  TrainData td = convert(line);
  EXPECT_EQ(td, expected);
}
