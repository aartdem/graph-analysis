#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "bench_commons.h"
#include "lagraph/boruvka_lagraph.hpp"
#include "spla/boruvka_spla.hpp"
#include "spla/library_spla.hpp"
#include "spla/prim_spla.hpp"

#if defined(CUDA_ENABLED)
#define USE_GUNROCK 1

#include "gunrock/boruvka.hxx"
#include "gunrock/prim.hxx"

#else
#define USE_GUNROCK 0
#endif

#ifndef DATA_DIR
#define DATA_DIR "data"
#endif

using namespace std;
using namespace algos;
using namespace bench;

int main() {
    cout << "MST Algorithms Benchmark" << endl;

#if USE_GUNROCK
    cout << "CUDA support: ENABLED (Gunrock algorithms available)" << endl;
#else
    cout << "CUDA support: DISABLED (Gunrock algorithms unavailable)" << endl;
#endif

    print_spla_accelerator_info();

    // List of algorithms to benchmark
    vector<pair<string, function<BenchmarkResult(const string &, int)>>>
            algorithms = {{"PrimSpla", [](const string &graph_path, int num_runs) {
                               return run_benchmark<PrimSpla>("PrimSpla", graph_path,
                                                              num_runs);
                           }}};
    algorithms.emplace_back("BoruvkaSpla", [](const string &graph_path, int num_runs) {
        return run_benchmark<BoruvkaSpla>("BoruvkaSpla", graph_path, num_runs);
    });
    algorithms.emplace_back("BoruvkaLagraph", [](const string &graph_path, int num_runs) {
        return run_benchmark<BoruvkaLagraph>("BoruvkaLagraph", graph_path, num_runs);
    });

#if USE_GUNROCK
    algorithms.emplace_back("BoruvkaGunrock", [](const string &graph_path, int num_runs) {
        return run_benchmark<BoruvkaGunrock>("BoruvkaGunrock", graph_path,
                                             num_runs);
    });

    algorithms.emplace_back("PrimGunrock", [](const string &graph_path, int num_runs) {
        return run_benchmark<PrimGunrock>("PrimGunrock", graph_path, num_runs);
    });
#endif

    // Find all .mtx files in the data directory
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

    const int NUM_RUNS = 1;

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

    string output_file = "benchmark_results.csv";
    save_results_to_csv(all_results, output_file);

    return 0;
}
