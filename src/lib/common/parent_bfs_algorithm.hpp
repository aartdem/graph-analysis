#pragma once

#include "algorithm.hpp"
#include "tree.hpp"

namespace algos {
    class ParentBfsAlgorithm : public Algorithm {
    public:
        virtual Tree get_result() = 0;
    };
}// namespace algos