#pragma once

#include <chrono>
#include <filesystem>

namespace algos {
    class Algorithm {
    public:
        virtual ~Algorithm() = default;

        virtual void load_graph(const std::filesystem::path &path) = 0;

        virtual std::chrono::milliseconds compute() = 0;
    };
};