#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "common/algorithm.hpp"

using namespace algos;
using namespace std;

namespace bench {

    template<typename T>
    Algorithm *create_algorithm() { return new T(); }

    struct BenchmarkResult {
        string algorithm_name;
        string graph_name;
        vector<double> execution_times;// in seconds
    };

    template<typename AlgoType>
    BenchmarkResult run_benchmark(const string &algo_name, const string &graph_path, int num_runs) {
        BenchmarkResult result;
        result.algorithm_name = algo_name;
        result.graph_name = filesystem::path(graph_path).filename().string();

        cout << "Running " << algo_name << " on " << result.graph_name << "..." << endl;

        result.execution_times.reserve(num_runs);
        for (int i = 0; i < num_runs; ++i) {
            const unique_ptr<Algorithm> algorithm(create_algorithm<AlgoType>());

            cout << "  Run " << (i + 1) << "/" << num_runs << "..." << flush;

            algorithm->load_graph(graph_path);

            auto time = algorithm->compute();
            double seconds = time.count();
            result.execution_times.push_back(seconds);

            cout << " " << fixed << setprecision(2) << seconds << " ms" << endl;
        }

        return result;
    }

    inline void save_results_to_csv(const vector<BenchmarkResult> &results, const string &output_file) {
        ofstream file(output_file);

        if (!file.is_open()) {
            throw runtime_error("Failed to open output file: " + output_file);
        }

        file << "Algorithm,Graph,Run";
        for (size_t i = 1; i <= results[0].execution_times.size(); ++i) {
            file << "," << i;
        }
        file << endl;

        for (const auto &result: results) {
            file << result.algorithm_name << "," << result.graph_name;
            for (double execution_time: result.execution_times) {
                file << "," << execution_time;
            }
            file << endl;
        }

        file.close();
        cout << "Results saved to " << output_file << endl;
    }
}

