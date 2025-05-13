#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include "loader.hxx"
#include "prim.hxx"

namespace algos {

// Structure holding the graph in CSR format and auxiliary GPU buffers
struct PrimGunrock::DeviceData {
  using vertex_t = PrimGunrock::vertex_t;
  using edge_t = PrimGunrock::edge_t;
  using weight_t = PrimGunrock::weight_t;

  // CSR representation
  thrust::device_vector<edge_t> d_row_offsets;   // length = num_vertices + 1
  thrust::device_vector<vertex_t> d_col_indices; // length = num_edges
  thrust::device_vector<weight_t> d_weight;      // length = num_edges

  // Host copy of row_offsets for host-side access
  thrust::host_vector<edge_t> h_row_offsets; // length = num_vertices + 1

  // Buffers for Prim's algorithm
  thrust::device_vector<weight_t> d_key;    // minimum edge weights
  thrust::device_vector<vertex_t> d_parent; // MST parent pointers
  thrust::device_vector<char> d_inMST;      // flags marking inclusion in MST

  DeviceData() = default;
};

PrimGunrock::PrimGunrock()
    : num_vertices(0), num_edges(0), dev_(new DeviceData()) {}

PrimGunrock::~PrimGunrock() = default;

void PrimGunrock::load_graph(const std::filesystem::path &file_path) {
  using vertex_t = PrimGunrock::vertex_t;
  using edge_t = PrimGunrock::edge_t;
  using weight_t = PrimGunrock::weight_t;

  // Load the graph in COO format
  std::vector<vertex_t> coo_row, coo_col;
  std::vector<weight_t> coo_val;
  detail::load_mtx_coo<vertex_t, edge_t, weight_t>(file_path, coo_row, coo_col,
                                                   coo_val);

  // Determine number of vertices and original edges
  edge_t original_edges = static_cast<edge_t>(coo_row.size());
  vertex_t max_v = 0;
  for (vertex_t u : coo_row)
    max_v = std::max(max_v, u);
  for (vertex_t v : coo_col)
    max_v = std::max(max_v, v);
  num_vertices = max_v + 1;

  // Build a symmetric COO (undirected graph)
  std::vector<vertex_t> srcs, dsts;
  std::vector<weight_t> weights;
  srcs.reserve(2 * original_edges);
  dsts.reserve(2 * original_edges);
  weights.reserve(2 * original_edges);

  for (edge_t i = 0; i < original_edges; ++i) {
    vertex_t u = coo_row[i], v = coo_col[i];
    weight_t w = coo_val[i];
    srcs.push_back(u);
    dsts.push_back(v);
    weights.push_back(w);
    if (u != v) {
      srcs.push_back(v);
      dsts.push_back(u);
      weights.push_back(w);
    }
  }
  num_edges = static_cast<edge_t>(srcs.size());

  // Build CSR structure locally
  // Compute vertex degrees
  std::vector<vertex_t> degrees(num_vertices, 0);
  for (edge_t i = 0; i < num_edges; ++i) {
    degrees[srcs[i]]++;
  }

  // Compute exclusive prefix sum for row offsets
  std::vector<vertex_t> row_offsets_local(num_vertices + 1);
  row_offsets_local[0] = 0;
  for (vertex_t i = 0; i < num_vertices; ++i)
    row_offsets_local[i + 1] = row_offsets_local[i] + degrees[i];

  // Initialize cursors and fill column and weight arrays
  std::vector<edge_t> cursor(row_offsets_local.begin(),
                             row_offsets_local.end());
  std::vector<vertex_t> col_idx_local(num_edges);
  std::vector<weight_t> w_local(num_edges);

  for (edge_t i = 0; i < num_edges; ++i) {
    vertex_t u = srcs[i];
    edge_t pos = cursor[u]++;
    col_idx_local[pos] = dsts[i];
    w_local[pos] = weights[i];
  }
  dev_->h_row_offsets = thrust::host_vector<vertex_t>(row_offsets_local.begin(),
                                                      row_offsets_local.end());

  // Copy CSR data to device vectors
  dev_->d_row_offsets = thrust::device_vector<vertex_t>(
      row_offsets_local.begin(), row_offsets_local.end());
  dev_->d_col_indices = thrust::device_vector<vertex_t>(col_idx_local.begin(),
                                                        col_idx_local.end());
  dev_->d_weight =
      thrust::device_vector<weight_t>(w_local.begin(), w_local.end());
}

std::chrono::seconds PrimGunrock::compute() {
  mst_edges.clear();
  using vertex_t = PrimGunrock::vertex_t;
  using edge_t = PrimGunrock::edge_t;
  using weight_t = PrimGunrock::weight_t;
  const weight_t INF = std::numeric_limits<weight_t>::max();

  auto start = std::chrono::steady_clock::now();
  if (num_vertices == 0)
    return {};

  auto &D = *dev_;
  // Resize buffers for Prim's algorithm
  D.d_key.resize(num_vertices);
  D.d_parent.resize(num_vertices);
  D.d_inMST.resize(num_vertices);

  thrust::fill(D.d_key.begin(), D.d_key.end(), INF);
  thrust::fill(D.d_parent.begin(), D.d_parent.end(), -1);
  thrust::fill(D.d_inMST.begin(), D.d_inMST.end(), 0);

  // Start from vertex 0
  D.d_key[0] = 0;
  D.d_parent[0] = 0;

  // Raw pointers for device data
  auto row_ptr = thrust::raw_pointer_cast(D.d_row_offsets.data());
  auto col_ptr = thrust::raw_pointer_cast(D.d_col_indices.data());
  auto w_ptr = thrust::raw_pointer_cast(D.d_weight.data());
  auto key_ptr = thrust::raw_pointer_cast(D.d_key.data());
  auto parent_ptr = thrust::raw_pointer_cast(D.d_parent.data());
  auto inMST_ptr = thrust::raw_pointer_cast(D.d_inMST.data());

  // Main Prim's loop
  for (size_t iter = 0; iter < num_vertices; ++iter) {
    // Select the vertex with the minimum key
    auto it =
        thrust::min_element(thrust::device, D.d_key.begin(), D.d_key.end());
    vertex_t u = it - D.d_key.begin();
    weight_t minKey = *it;
    if (minKey == INF) {
      // Handle disconnected components by resetting a new start vertex
      bool found = false;
      for (vertex_t x = 0; x < num_vertices; ++x) {
        if (D.d_inMST[x] == 0) {
          D.d_key[x] = 0;
          D.d_parent[x] = x;
          it = D.d_key.begin() + x;
          u = x;
          minKey = 0;
          found = true;
          break;
        }
      }
      if (!found)
        break;
    }

    // Include vertex u in the MST
    D.d_inMST[u] = 1;
    *it = INF;
    if (u != D.d_parent[u])
      mst_edges.emplace_back(D.d_parent[u], u, minKey);

    // Update keys of adjacent vertices in parallel
    edge_t e_start = D.h_row_offsets[u];
    edge_t e_end = D.h_row_offsets[u + 1];
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<edge_t>(e_start),
                     thrust::make_counting_iterator<edge_t>(e_end),
                     [=] __device__(edge_t e) {
                       vertex_t v = col_ptr[e];
                       weight_t w = w_ptr[e];
                       if (!inMST_ptr[v] && w < key_ptr[v]) {
                         key_ptr[v] = w;
                         parent_ptr[v] = u;
                       }
                     });
  }

  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::seconds>(end - start);
}

Tree PrimGunrock::get_result() {
  // Copy parent array back to host
  std::vector<int> parent(num_vertices);
  thrust::copy(dev_->d_parent.begin(), dev_->d_parent.end(), parent.begin());

  // Compute total weight of the MST
  float totalWeight = 0;
  for (auto &edge : mst_edges)
    totalWeight += std::get<2>(edge);
  return Tree(num_vertices, parent, totalWeight);
}

} // namespace algos
