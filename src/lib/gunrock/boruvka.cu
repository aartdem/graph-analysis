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

#include "boruvka.hxx"
#include "loader.hxx"

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

// CUDA kernel to record edge candidates into keys/vals
__global__ void
step1_kernel(int m,         // number of edges
             int *comp_ptr, // comp array, length = num_vertices
             int *src_ptr,  // src array,  length = m
             int *dst_ptr,  // dst array,  length = m
             float *w_ptr,  // weight array,length = m
             int *keys_ptr, // keys array, length = 2*m
             BoruvkaGunrock::EdgePair *vals_ptr) // vals array, length = 2*m
{
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= m)
    return;

  int u = src_ptr[e];
  int v = dst_ptr[e];
  float w = w_ptr[e];
  int cu = comp_ptr[u];
  int cv = comp_ptr[v];

  if (cu == cv) {
    keys_ptr[2 * e] = -1;
    vals_ptr[2 * e] = {w, e};
    keys_ptr[2 * e + 1] = -1;
    vals_ptr[2 * e + 1] = {w, e};
  } else {
    keys_ptr[2 * e] = cu;
    vals_ptr[2 * e] = {w, e};
    keys_ptr[2 * e + 1] = cv;
    vals_ptr[2 * e + 1] = {w, e};
  }
}

__global__ void step2_kernel(int k, BoruvkaGunrock::vertex_t *src_ptr,
                             BoruvkaGunrock::vertex_t *dst_ptr,
                             BoruvkaGunrock::vertex_t *comp_ptr,
                             BoruvkaGunrock::vertex_t *newcomp_ptr,
                             BoruvkaGunrock::EdgePair *out_vals_ptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= k)
    return;

  auto ep = out_vals_ptr[i];
  int e = ep.idx;

  auto u = src_ptr[e];
  auto v = dst_ptr[e];
  auto cu = comp_ptr[u];
  auto cv = comp_ptr[v];
  if (cu == cv)
    return;

  // choose root/other
  auto root = (cu < cv ? cu : cv);
  auto other = (cu < cv ? cv : cu);
  // merge
  newcomp_ptr[other] = root;
}

__global__ void pointer_jump_kernel(int num_vertices,
                                    BoruvkaGunrock::vertex_t *newcomp_ptr) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= num_vertices)
    return;
  auto parent = newcomp_ptr[v];
  auto grandparent = newcomp_ptr[parent];
  if (grandparent != parent) {
    newcomp_ptr[v] = grandparent;
  }
}

__global__ void update_comp_kernel(int num_vertices,
                                   BoruvkaGunrock::vertex_t *comp_ptr,
                                   BoruvkaGunrock::vertex_t *newcomp_ptr) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= num_vertices)
    return;
  comp_ptr[v] = newcomp_ptr[comp_ptr[v]];
}

BoruvkaGunrock::BoruvkaGunrock()
    : num_vertices(0), num_edges(0), dev_(new DeviceData()) {}

BoruvkaGunrock::~BoruvkaGunrock() = default;

void BoruvkaGunrock::load_graph(const std::filesystem::path &file_path) {
  std::vector<vertex_t> host_rows, host_cols;
  std::vector<weight_t> host_vals;
  detail::load_mtx_coo<vertex_t, edge_t, weight_t>(file_path, host_rows,
                                                   host_cols, host_vals);
  edge_t original_edges = static_cast<edge_t>(host_rows.size());

  // 2) Compute number of vertices = max index + 1
  vertex_t max_v = 0;
  for (edge_t i = 0; i < original_edges; ++i) {
    max_v = std::max({max_v, host_rows[i], host_cols[i]});
  }
  num_vertices = max_v + 1;

  // 3) Build undirected edge lists (duplicate if u!=v)
  thrust::host_vector<vertex_t> h_src, h_dst;
  thrust::host_vector<weight_t> h_w;
  h_src.reserve(2 * original_edges);
  h_dst.reserve(2 * original_edges);
  h_w.reserve(2 * original_edges);

  for (edge_t i = 0; i < original_edges; ++i) {
    auto u = host_rows[i];
    auto v = host_cols[i];
    auto w = host_vals[i];
    h_src.push_back(u);
    h_dst.push_back(v);
    h_w.push_back(w);
    if (u != v) {
      h_src.push_back(v);
      h_dst.push_back(u);
      h_w.push_back(w);
    }
  }
  num_edges = static_cast<edge_t>(h_src.size());

  // 4) Copy into your device‐side vectors
  dev_->d_src = h_src;
  dev_->d_dst = h_dst;
  dev_->d_weight = h_w;
}

std::chrono::milliseconds BoruvkaGunrock::compute() {
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
  edge_t m0 = num_edges / 2;
  std::vector<char> edge_used(m0, 0);
  edge_used.assign(m0, 0);
  bool merged = true;

  // Repeat until no more merges occur
  while (merged) {
    merged = false;
    keys.resize(2 * num_edges);
    vals.resize(2 * num_edges);
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
    {
      int threads = 256;
      int blocks = (m + threads - 1) / threads;
      step1_kernel<<<blocks, threads>>>(m, comp_ptr, src_ptr, dst_ptr, w_ptr,
                                        keys_ptr, vals_ptr);
      cudaDeviceSynchronize();
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
    if (new_size == 0)
      // If no entries remain, MST is complete.
      break;

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

    if (num_comps_found == 0)
      break; // if numFound == 0: no more cross‐component edges → MST complete

    // Step 4: merge components using those edges
    thrust::device_vector<vertex_t> newComp_map(num_vertices);
    // Initialize newComp_map[i] = i
    thrust::sequence(newComp_map.begin(), newComp_map.end(), 0);
    vertex_t *newcomp_ptr = thrust::raw_pointer_cast(newComp_map.data());
    EdgePair *out_vals_ptr = thrust::raw_pointer_cast(comp_vals_out.data());
    size_t k = num_comps_found;

    // CUDA kernel: for each (compID → EdgePair{w,e}):
    {
      int threads = 256;
      int blocks = (k + threads - 1) / threads;
      step2_kernel<<<blocks, threads>>>(k, src_ptr, dst_ptr, comp_ptr,
                                        newcomp_ptr, out_vals_ptr);
      cudaDeviceSynchronize();
    }

    // Flatten union chains with pointer jumping on newComp_map
    bool updated = true;
    int iter = 0;
    while (updated && iter < 10) {
      updated = false;

      // Atomic update: newComp_map[x] = newComp_map[newComp_map[x]]
      {
        int threads = 256;
        int blocks = (num_vertices + threads - 1) / threads;
        for (int iter = 0; iter < 10; ++iter) {
          pointer_jump_kernel<<<blocks, threads>>>(num_vertices, newcomp_ptr);
          cudaDeviceSynchronize();
        }
      }
    }

    // Update component labels for all vertices: comp[v] = newComp_map[
    // comp[v] ]
    {
      int threads = 256;
      int blocks = (num_vertices + threads - 1) / threads;
      update_comp_kernel<<<blocks, threads>>>(num_vertices, comp_ptr,
                                              newcomp_ptr);
      cudaDeviceSynchronize();
    }

    // Step 5: add the chosen edges to the MST result (on host)
    // Copy (weight, edgeIndex) pairs from comp_vals_out and corresponding
    // component IDs from comp_keys_out to host To avoid duplicate edges, use
    // a flag array marking edges already added
    std::vector<vertex_t> out_keys_host(num_comps_found);
    std::vector<EdgePair> out_vals_host(num_comps_found);
    thrust::copy(comp_keys_out.begin(), comp_keys_out.begin() + num_comps_found,
                 out_keys_host.begin());
    thrust::copy(comp_vals_out.begin(), comp_vals_out.begin() + num_comps_found,
                 out_vals_host.begin());

    // Recover the actual vertices of each edge
    for (size_t i = 0; i < num_comps_found; ++i) {
      edge_t e = out_vals_host[i].idx;
      edge_t orig = e / 2; // индекс неориентированного ребра
      weight_t w = out_vals_host[i].w;
      if (!edge_used[orig]) {
        edge_used[orig] = 1;
        vertex_t u = dev_->d_src[e];
        vertex_t v = dev_->d_dst[e];
        mst_edges.emplace_back(u, v, w);
      }
    }
    merged = true; // if merges occurred, continue the loop
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
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