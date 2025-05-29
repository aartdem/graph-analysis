#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "bench_commons.h"
#include "spla/parent_bfs_spla.hpp"
#include "spla/library_spla.hpp"
#include "lagraph/parent_bfs_lagraph.hpp"

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

using namespace std;
using namespace algos;
using namespace bench;

int main() {
    cout << "Prent BFS Algorithms Benchmark" << endl;

    print_spla_accelerator_info();

    // List of algorithms to benchmark
    vector<pair<string, function<BenchmarkResult(const string &, int)>>> algorithms{};

    algorithms.emplace_back("BfsLagraph", [](const string &graph_path, int num_runs) {
        return run_benchmark<ParentBfsLagraph>("BfsLagraph", graph_path,
                                               num_runs);
    });
    algorithms.emplace_back("BfsSpla", [](const string &graph_path, int num_runs) {
        return run_benchmark<ParentBfsSpla>("BfsSpla", graph_path, num_runs);
    });

    vector<string> graph_files;
    for (const auto &entry: filesystem::directory_iterator(DATA_DIR)) {
        if (entry.path().extension() == ".mtx") {
            graph_files.push_back(entry.path().string());
        }
    }

    if (graph_files.empty()) {
        cout << "No .mtx files found in the data directory." << endl;
        cout << "Please place graph files in the data directory and try again." << endl;
        return 1;
    }

    cout << "Found " << graph_files.size() << " graph files in the data directory." << endl;
    for (const auto &file: graph_files) {
        cout << "  - " << filesystem::path(file).filename().string() << endl;
    }

    const int NUM_RUNS = 20;

    vector<BenchmarkResult> all_results;
    for (const auto &graph_file: graph_files) {
        for (const auto &[algo_name, benchmark_func]: algorithms) {
            try {
                BenchmarkResult result = benchmark_func(graph_file, NUM_RUNS);
                all_results.push_back(result);
            } catch (const exception &e) {
                cerr << "Error running " << algo_name << " on " << filesystem::path(graph_file).filename().string()
                     << ": " << e.what() << endl;
            }
        }
    }

    string output_file = "benchmark_results_bfs.csv";
    save_results_to_csv(all_results, output_file);

    return 0;
}