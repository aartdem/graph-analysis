#pragma once

namespace algos
{
    class Algorithm
    {
    public:
        virtual ~Algorithm() {}
        virtual void load_graph(const std::filesystem::path &path) = 0;
        virtual std::chrono::duration<double> compute() = 0;
    };
};
