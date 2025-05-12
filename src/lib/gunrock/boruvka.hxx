#pragma once

#include <tuple>
#include <memory>
#include <vector>

#include "common/mst_algorithm.hpp"
#include "common/tree.hpp"

namespace algos {

class BoruvkaGunrock : public MstAlgorithm {
  long long num_original_edges;
public:
  BoruvkaGunrock();
  ~BoruvkaGunrock() override;

  void load_graph(const std::filesystem::path &file_path) override;
  std::chrono::seconds compute() override;
  Tree get_result() override;

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  vertex_t num_vertices;
  edge_t num_edges;

  struct EdgePair;
  struct MinEdgeOp;

  class DeviceData;
  std::unique_ptr<DeviceData> dev_;
  std::vector<std::tuple<vertex_t, vertex_t, weight_t>> mst_edges;
};

} // namespace algos
