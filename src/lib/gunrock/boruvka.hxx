#pragma once

#include <cuda_runtime.h> // defines threadIdx, blockIdx, blockDim, gridDim
#include <device_launch_parameters.h>
#include <gunrock/algorithms/algorithms.hxx> // Gunrock core
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "common/mst_algorithm.hpp"
#include "common/tree.hpp"

namespace algos {

class BoruvkaGunrock : public MstAlgorithm {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // Graph data
  vertex_t num_vertices;
  edge_t num_edges;
  thrust::device_vector<vertex_t> d_src;    // source vertex array (device)
  thrust::device_vector<vertex_t> d_dst;    // destination vertex array (device)
  thrust::device_vector<weight_t> d_weight; // edge weights (device)

  // MST result: list of edges (host)
  std::vector<std::tuple<vertex_t, vertex_t, weight_t>> mst_edges;

public:
  // Helper struct for reduction
  struct EdgePair {
    weight_t w;
    edge_t idx;
  };

  // Compare by weight (min reduction)
  struct MinEdgeOp {
    __host__ __device__ EdgePair operator()(const EdgePair &a,
                                            const EdgePair &b) const {
      return (a.w <= b.w) ? a : b;
    }
  };

  // Load graph from Matrix Market file
  void load_graph(const std::filesystem::path &file_path) override;

  // Compute MST and return duration
  std::chrono::seconds compute() override;

  // Build result tree
  Tree get_result() override;
};

} // namespace algos
