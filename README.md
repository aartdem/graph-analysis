# Graph Analysis

Library for graph algorithms analysis with GPU acceleration.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

This project provides a platform for analyzing various graph algorithms using different implementations. It integrates several high-performance graph processing libraries including SPLA, Gunrock and LAGraph to enable comparative analysis of graph algorithm performance.

## Requirements

- CMake (version 3.20 or higher)
- C++20 compatible compiler
- CUDA Toolkit 12.0+ (for GPU acceleration)
- Python 3.7+ (for visualizing experiment results)

## Dependencies

The project uses the following libraries:
- **GraphBLAS** - Sparse linear algebra library for graph algorithms
- **LAGraph** - Library of graph algorithms based on GraphBLAS
- **Gunrock** - GPU-accelerated graph processing library
- **SPLA** - Sparse linear algebra framework with GPU acceleration

All dependencies are included as git submodules.

## Getting Started

### Clone the repository

```bash
git clone --recurse-submodules https://github.com/aartdem/graph-analysis.git
cd graph-analysis
```

### Build the project

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

If you want to disable CUDA support, use:

```bash
cmake .. -DUSE_CUDA=OFF
```

### Run tests

```bash
cd build
ctest
```

Or run tests directly:

```bash
cd build
./src/tests/graph_analysis_tests
```

## Project Structure

```
graph-analysis/
├── data/            # Graph datasets
├── deps/            # Dependencies (submodules)
│   ├── GraphBLAS/   # GraphBLAS library
│   ├── LAGraph/     # LAGraph library
│   ├── gunrock/     # Gunrock library
│   └── spla/        # SPLA library
├── src/             # Source code
│   ├── lib/         # Main library code
│   ├── tests/       # Unit tests
│   └── experiment/  # Benchmark code
└── test_data/       # Data for testing
```

## Experiment

The experiment compares the performance of various Minimum Spanning Tree (MST) algorithms across different graph datasets.

### Experiment Setup

- **Hardware Platform**:
  - **OS**: Ubuntu 24.04 LTS
  - **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM)
  - **GPU Driver**: Version 32.0.15.5597
  - **CUDA Toolkit**: 12.3
  - **CUDA Cores**: 2560
  - **CPU**: 11th Gen Intel Core i7-11800H @ 2.30GHz
  - **Cache**: L1 – 640KB, L2 – 10MB, L3 – 24MB
  - **RAM**: 16 GB

### Algorithms Compared

1. PrimSpla - Prim's algorithm implementation using SPLA
2. BoruvkaSpla - Borůvka's algorithm implementation using SPLA
3. BoruvkaGunrock - Borůvka's algorithm implementation using Gunrock
4. PrimGunrock - Prim's algorithm implementation using Gunrock

### Running the Experiment

```bash
cd build
./src/experiment/mst_benchmark
```

### Analyzing Results

The experiment generates CSV files with detailed performance measurements. Use the provided Python script to visualize the results:

```bash
python src/experiment/benchmark_visualize.py benchmark_results/detailed_results_*.csv benchmark_results/summary_results_*.csv
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.