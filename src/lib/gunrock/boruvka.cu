
#include "boruvka.hxx"
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

namespace algos {
struct BoruvkaGunrock::DeviceData {
  using vertex_t = BoruvkaGunrock::vertex_t;
  using edge_t = BoruvkaGunrock::edge_t;
  using weight_t = BoruvkaGunrock::weight_t;

  thrust::device_vector<vertex_t> d_src, d_dst;
  thrust::device_vector<weight_t> d_weight;

  DeviceData() = default;
};

struct BoruvkaGunrock::EdgePair {
  weight_t w;
  edge_t idx;
};

struct BoruvkaGunrock::MinEdgeOp {
  __host__ __device__ EdgePair operator()(EdgePair const &a,
                                          EdgePair const &b) const {
    return (a.w <= b.w) ? a : b;
  }
};

BoruvkaGunrock::BoruvkaGunrock()
    : num_vertices(0), num_edges(0), dev_(new DeviceData()) {}

BoruvkaGunrock::~BoruvkaGunrock() = default;

void BoruvkaGunrock::load_graph(const std::filesystem::path &file_path) {
  using namespace gunrock;
  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  auto loaded = mm.load(file_path);
  auto &coo = std::get<1>(loaded);

  // Define amount edges and verticies
  auto host_rows = coo.row_indices;
  auto host_cols = coo.column_indices;
  auto host_vals = coo.nonzero_values;
  edge_t original_edges = static_cast<edge_t>(host_rows.size());

  // Define max id of the vertex
  vertex_t max_v = 0;
  for (edge_t i = 0; i < original_edges; ++i) {
    max_v = std::max({max_v, host_rows[i], host_cols[i]});
  }
  num_vertices = max_v + 1;

  // Arrays of edges
  thrust::host_vector<vertex_t> h_src;
  thrust::host_vector<vertex_t> h_dst;
  thrust::host_vector<weight_t> h_weight;
  h_src.reserve(2 * original_edges);
  h_dst.reserve(2 * original_edges);
  h_weight.reserve(2 * original_edges);
  for (edge_t i = 0; i < original_edges; ++i) {
    auto u = host_rows[i];
    auto v = host_cols[i];
    auto w = host_vals[i];
    h_src.push_back(u);
    h_dst.push_back(v);
    h_weight.push_back(w);
    if (u != v) {
      h_src.push_back(v);
      h_dst.push_back(u);
      h_weight.push_back(w);
    }
  }
  num_edges = static_cast<edge_t>(h_src.size());

  // Copy to GPU
  dev_->d_src = h_src;
  dev_->d_dst = h_dst;
  dev_->d_weight = h_weight;
}

std::chrono::seconds BoruvkaGunrock::compute() {
  mst_edges.clear();
  auto start = std::chrono::steady_clock::now();
  if (num_vertices == 0)
    return {};

  // Initialize component array: each vertex in its own component (comp[v] = v)
  thrust::device_vector<vertex_t> comp(num_vertices);
  thrust::sequence(comp.begin(), comp.end(), 0); // comp[i] = i

  // Buffers for keys and values when finding minimum edges
  thrust::device_vector<vertex_t> keys(2 * num_edges);
  thrust::device_vector<EdgePair> vals(2 * num_edges);
  thrust::device_vector<vertex_t> comp_keys_out(
      num_vertices); // keys: component IDs
  thrust::device_vector<EdgePair> comp_vals_out(
      num_vertices); // vals: candidate min-edge per component
  bool merged = true;

  // Repeat until no more merges occur
  while (merged) {
    merged = false;
    // Step 1: Record edge candidates into keys/vals
    // Each edge (u,v) adds two entries—one for u’s comp, one for v’s.
    // If both ends share a comp, mark entry invalid (key = -1).
    vertex_t *comp_ptr = thrust::raw_pointer_cast(comp.data());
    vertex_t *src_ptr = thrust::raw_pointer_cast(dev_->d_src.data());
    vertex_t *dst_ptr = thrust::raw_pointer_cast(dev_->d_dst.data());
    weight_t *w_ptr = thrust::raw_pointer_cast(dev_->d_weight.data());
    vertex_t *keys_ptr = thrust::raw_pointer_cast(keys.data());
    EdgePair *vals_ptr = thrust::raw_pointer_cast(vals.data());
    edge_t m = num_edges;

// Launch CUDA kernel: one thread per edge
#pragma omp target teams distribute parallel for is_device_ptr(                \
        comp_ptr, src_ptr, dst_ptr, w_ptr, keys_ptr, vals_ptr)
    for (edge_t e = 0; e < m; ++e) {
      vertex_t u = src_ptr[e];
      vertex_t v = dst_ptr[e];
      weight_t w = w_ptr[e];
      vertex_t comp_u = comp_ptr[u];
      vertex_t comp_v = comp_ptr[v];
      if (comp_u == comp_v) {
        // If edge is internal to a component: mark invalid
        keys_ptr[2 * e] = -1;
        vals_ptr[2 * e] = {w, e};
        keys_ptr[2 * e + 1] = -1;
        vals_ptr[2 * e + 1] = {w, e};
      } else {
        // Otherwise: emit both directions as valid
        keys_ptr[2 * e] = comp_u;
        vals_ptr[2 * e] = {w, e};
        keys_ptr[2 * e + 1] = comp_v;
        vals_ptr[2 * e + 1] = {w, e};
      }
    }

    // Step 2: Filter out invalid entries (key = -1)
    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(keys.begin(), vals.begin()));
    auto zip_end =
        thrust::make_zip_iterator(thrust::make_tuple(keys.end(), vals.end()));
    auto new_end = thrust::remove_if(
        zip_begin, zip_end,
        [] __host__ __device__(const thrust::tuple<vertex_t, EdgePair> &kv) {
          vertex_t comp_id = thrust::get<0>(kv);
          return comp_id == -1;
        });
    size_t new_size = new_end - zip_begin;
    if (new_size == 0) {
      // If no entries remain, MST is complete.
      break;
    }

    // Resize vectors to new_size after filtering.
    keys.resize(new_size);
    vals.resize(new_size);
    // Step 3: Sort by component ID and pick the minimum edge for each group
    thrust::sort_by_key(keys.begin(), keys.end(), vals.begin());

    // Perform reduce_by_key on (keys, vals) to pick the lightest edge for each
    auto reduce_end = thrust::reduce_by_key(
        keys.begin(), keys.end(), vals.begin(), comp_keys_out.begin(),
        comp_vals_out.begin(), thrust::equal_to<vertex_t>(), MinEdgeOp());
    size_t num_comps_found =
        reduce_end.first -
        comp_keys_out
            .begin(); // comp numFound = number of comps that got an edge

    if (num_comps_found == 0) {
      break; // if numFound == 0: no more cross‐component edges → MST complete
    }

    // Step 4: merge components using those edges
    thrust::device_vector<vertex_t> newComp_map(num_vertices);
    // Initialize newComp_map[i] = i
    thrust::sequence(newComp_map.begin(), newComp_map.end(), 0);
    vertex_t *newcomp_ptr = thrust::raw_pointer_cast(newComp_map.data());
    EdgePair *out_vals_ptr = thrust::raw_pointer_cast(comp_vals_out.data());
    size_t k = num_comps_found;

// CUDA kernel: for each (compID → EdgePair{w,e}):
#pragma omp target teams distribute parallel for is_device_ptr(                \
        comp_ptr, src_ptr, dst_ptr, out_keys_ptr, out_vals_ptr, newcomp_ptr)
    for (size_t i = 0; i < k; ++i) {
      EdgePair ep = out_vals_ptr[i];
      edge_t e = ep.idx;

      // Recover endpoints u,v and their comp labels
      vertex_t u = src_ptr[e];
      vertex_t v = dst_ptr[e];
      vertex_t comp_u = comp_ptr[u];
      vertex_t comp_v = comp_ptr[v];
      if (comp_u == comp_v) {
        continue; // if same label: skip
      }
      // Select the new component representative (smallest ID for determinism)
      vertex_t root = (comp_u < comp_v ? comp_u : comp_v);
      vertex_t other = (comp_u < comp_v ? comp_v : comp_u);

      // Merge: redirect 'other' to 'root'
      newcomp_ptr[other] = root;
    }

    // Flatten union chains with pointer jumping on newComp_map
    bool updated = true;
    int iter = 0;
    while (updated && iter < 10) {
      updated = false;

// Atomic update: newComp_map[x] = newComp_map[newComp_map[x]]
#pragma omp target teams distribute parallel for is_device_ptr(newcomp_ptr)
      for (vertex_t i = 0; i < num_vertices; ++i) {
        vertex_t parent = newcomp_ptr[i];
        vertex_t grandparent = newcomp_ptr[parent];
        if (grandparent != parent) {
          newcomp_ptr[i] = grandparent;
          updated = true;
        }
      }
      iter++;
    }

// Update component labels for all vertices: comp[v] = newComp_map[ comp[v] ]
#pragma omp target teams distribute parallel for is_device_ptr(                \
        comp_ptr, newcomp_ptr)
    for (vertex_t v = 0; v < num_vertices; ++v) {
      comp_ptr[v] = newcomp_ptr[comp_ptr[v]];
    }

    // Step 5: add the chosen edges to the MST result (on host)
    // Copy (weight, edgeIndex) pairs from comp_vals_out and corresponding
    // component IDs from comp_keys_out to host To avoid duplicate edges, use a
    // flag array marking edges already added
    std::vector<vertex_t> out_keys_host(num_comps_found);
    std::vector<EdgePair> out_vals_host(num_comps_found);
    thrust::copy(comp_keys_out.begin(), comp_keys_out.begin() + num_comps_found,
                 out_keys_host.begin());
    thrust::copy(comp_vals_out.begin(), comp_vals_out.begin() + num_comps_found,
                 out_vals_host.begin());

    // Recover the actual vertices of each edge
    static std::vector<char> edge_used;
    edge_used.assign(num_edges, 0);
    for (size_t i = 0; i < num_comps_found; ++i) {
      edge_t e = out_vals_host[i].idx;
      weight_t w = out_vals_host[i].w;

      // After the final comp update, u and v are in the same component
      vertex_t u = (vertex_t)dev_->d_src[e];
      vertex_t v = (vertex_t)dev_->d_dst[e];
      if (!edge_used[e]) {
        edge_used[e] = 1;
        mst_edges.emplace_back(u, v, w);
      }
    }
    merged = true; // if merges occurred, continue the loop
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::seconds>(end - start);
}

Tree BoruvkaGunrock::get_result() {
  // Initialize tree structure: parent = -1 for all vertices
  std::vector<int> parent(num_vertices, -1);
  float total_weight = 0.0f;

  // Fill parent array from the MST edge list
  for (auto &e : mst_edges) {
    vertex_t u, v;
    weight_t w;
    std::tie(u, v, w) = e;

    // If v has no parent yet, set u as its parent
    if (parent[v] == -1 && parent[u] != v) {
      parent[v] = u;
    } else if (parent[u] == -1 && parent[v] != u) {
      parent[u] = v;
    }
    total_weight += w;
  }
  return Tree(num_vertices, parent, total_weight);
}
} // namespace algos