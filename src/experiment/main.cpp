#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "spla/boruvka_spla.hpp"
#include "spla/prim_spla.hpp"

#if defined(HAVE_CUDA) || defined(CUDA_ENABLED)
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

template<typename T>
MstAlgorithm *create_algorithm() { return new T(); }

struct BenchmarkResult {
    string algorithm_name;
    string graph_name;
    vector<double> execution_times;// in seconds
};

template<typename AlgoType>
BenchmarkResult run_benchmark(const string &algo_name, const string &graph_path,
                              int num_runs = 20) {
    BenchmarkResult result;
    result.algorithm_name = algo_name;
    result.graph_name = filesystem::path(graph_path).filename().string();

    cout << "Running " << algo_name << " on " << result.graph_name << "..."
         << endl;

    result.execution_times.reserve(num_runs);
    for (int i = 0; i < num_runs; ++i) {
        const unique_ptr<MstAlgorithm> algorithm(create_algorithm<AlgoType>());

        cout << "  Run " << (i + 1) << "/" << num_runs << "..." << flush;

        algorithm->load_graph(graph_path);

        auto time = algorithm->compute();
        double seconds = time.count();
        result.execution_times.push_back(seconds);

        cout << " " << fixed << setprecision(2) << seconds << " s" << endl;
    }

    return result;
}

void save_results_to_csv(const vector<BenchmarkResult> &results,
                         const string &output_file) {
    ofstream file(output_file);

    if (!file.is_open()) {
        throw runtime_error("Failed to open output file: " + output_file);
    }

    file << "Algorithm,Graph,Run";
    for (int i = 1; i <= results[0].execution_times.size(); ++i) {
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

int main() {
    cout << "MST Algorithms Benchmark" << endl;

#if USE_GUNROCK
    cout << "CUDA поддержка: ВКЛЮЧЕНА (доступны алгоритмы Gunrock)" << endl;
#else
    cout << "CUDA поддержка: ОТКЛЮЧЕНА (алгоритмы Gunrock недоступны)" << endl;
#endif

    // List of algorithms to benchmark
    vector<pair<string, function<BenchmarkResult(const string &, int)>>>
            algorithms = {{"PrimSpla", [](const string &graph_path, int num_runs) {
                               return run_benchmark<PrimSpla>("PrimSpla", graph_path,
                                                              num_runs);
                           }}};
    algorithms.emplace_back("BoruvkaSpla", [](const string &graph_path, int num_runs) {
        return run_benchmark<BoruvkaSpla>("BoruvkaSpla", graph_path, num_runs);
    });

#if USE_GUNROCK
    algorithms.push_back(
            {"BoruvkaGunrock", [](const string &graph_path, int num_runs) {
                 return run_benchmark<BoruvkaGunrock>("BoruvkaGunrock", graph_path,
                                                      num_runs);
             }});

    algorithms.push_back(
            {"PrimGunrock", [](const string &graph_path, int num_runs) {
                 return run_benchmark<PrimGunrock>("PrimGunrock", graph_path, num_runs);
             }});
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
        cout << "Please place graph files in the data directory and try again."
             << endl;
        return 1;
    }

    cout << "Found " << graph_files.size()
         << " graph files in the data directory." << endl;
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
                cerr << "Error running " << algo_name << " on "
                     << filesystem::path(graph_file).filename().string() << ": "
                     << e.what() << endl;
            }
        }
    }

    string output_file = "benchmark_results.csv";
    save_results_to_csv(all_results, output_file);

    return 0;
}
