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

  thrust::device_vector<edge_t> d_row_offsets;
  thrust::device_vector<vertex_t> d_col_indices;
  thrust::device_vector<weight_t> d_weight;

  thrust::device_vector<weight_t> d_key;
  thrust::device_vector<vertex_t> d_parent;
  thrust::device_vector<char> d_inMST;

  // buffers for block reduction
  thrust::device_vector<weight_t> d_min_key;
  thrust::device_vector<vertex_t> d_min_idx;
};

// Block-level reduction kernel: find min(key) and its index in each block
__global__ void minKeyReduce(const PrimGunrock::weight_t *keys,
                             PrimGunrock::weight_t *block_min_key,
                             PrimGunrock::vertex_t *block_min_idx, int n,
                             PrimGunrock::weight_t INF) {
  extern __shared__ char smem[];
  auto *s_keys = reinterpret_cast<PrimGunrock::weight_t *>(smem);
  auto *s_idx = reinterpret_cast<PrimGunrock::vertex_t *>(&s_keys[blockDim.x]);

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  // initialize
  PrimGunrock::weight_t v = (gid < n) ? keys[gid] : INF;
  PrimGunrock::vertex_t idx = (gid < n) ? gid : -1;
  s_keys[tid] = v;
  s_idx[tid] = idx;
  __syncthreads();

  // tree-based reduction
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_keys[tid + offset] < s_keys[tid]) {
        s_keys[tid] = s_keys[tid + offset];
        s_idx[tid] = s_idx[tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_min_key[blockIdx.x] = s_keys[0];
    block_min_idx[blockIdx.x] = s_idx[0];
  }
}

// Kernel to relax edges of a single vertex u
__global__ void relaxEdges(const PrimGunrock::edge_t *row_offsets,
                           const PrimGunrock::vertex_t *col_indices,
                           const PrimGunrock::weight_t *weights,
                           PrimGunrock::weight_t *keys,
                           PrimGunrock::vertex_t *parents, const char *inMST,
                           PrimGunrock::vertex_t u) {
  int lane = threadIdx.x;
  auto start = row_offsets[u];
  auto end = row_offsets[u + 1];
  for (auto e = start + lane; e < end; e += blockDim.x) {
    auto v = col_indices[e];
    auto w = weights[e];
    if (!inMST[v] && w < keys[v]) {
      keys[v] = w;
      parents[v] = u;
    }
  }
}

PrimGunrock::PrimGunrock()
    : num_vertices(0), num_edges(0), dev_(new DeviceData()) {}

PrimGunrock::~PrimGunrock() = default;

void PrimGunrock::load_graph(const std::filesystem::path &file_path) {
  using vertex_t = PrimGunrock::vertex_t;
  using edge_t = PrimGunrock::edge_t;
  using weight_t = PrimGunrock::weight_t;

  std::vector<vertex_t> coo_row, coo_col;
  std::vector<weight_t> coo_val;
  detail::load_mtx_coo<vertex_t, edge_t, weight_t>(file_path, coo_row, coo_col,
                                                   coo_val);

  edge_t orig_e = (edge_t)coo_row.size();
  vertex_t max_v = 0;
  for (auto u : coo_row)
    max_v = max(max_v, u);
  for (auto v : coo_col)
    max_v = max(max_v, v);
  num_vertices = max_v + 1;

  std::vector<vertex_t> src, dst;
  std::vector<weight_t> wts;
  src.reserve(2 * orig_e);
  dst.reserve(2 * orig_e);
  wts.reserve(2 * orig_e);
  for (edge_t i = 0; i < orig_e; i++) {
    auto u = coo_row[i], v = coo_col[i];
    auto w = coo_val[i];
    src.push_back(u);
    dst.push_back(v);
    wts.push_back(w);
    if (u != v) {
      src.push_back(v);
      dst.push_back(u);
      wts.push_back(w);
    }
  }
  num_edges = (edge_t)src.size();

  std::vector<edge_t> degrees(num_vertices, 0);
  for (edge_t i = 0; i < num_edges; i++)
    degrees[src[i]]++;

  std::vector<edge_t> row_off(num_vertices + 1);
  row_off[0] = 0;
  for (vertex_t i = 0; i < num_vertices; i++)
    row_off[i + 1] = row_off[i] + degrees[i];

  std::vector<edge_t> cursor = row_off;
  std::vector<vertex_t> col_idx(num_edges);
  std::vector<weight_t> w_local(num_edges);
  for (edge_t i = 0; i < num_edges; i++) {
    auto u = src[i];
    auto pos = cursor[u]++;
    col_idx[pos] = dst[i];
    w_local[pos] = wts[i];
  }

  // Copy CSR to device
  auto &D = *dev_;
  D.d_row_offsets = row_off;
  D.d_col_indices = col_idx;
  D.d_weight = w_local;

  // Pre-allocate buffers
  D.d_key.resize(num_vertices);
  D.d_parent.resize(num_vertices);
  D.d_inMST.resize(num_vertices);

  int threads = 256;
  int blocks = (num_vertices + threads - 1) / threads;
  D.d_min_key.resize(blocks);
  D.d_min_idx.resize(blocks);
}

std::chrono::milliseconds PrimGunrock::compute() {
  using weight_t = PrimGunrock::weight_t;
  const weight_t INF = std::numeric_limits<weight_t>::max();
  auto start = std::chrono::steady_clock::now();
  if (num_vertices == 0)
    return {};
  auto &D = *dev_;

  // init
  thrust::fill(D.d_key.begin(), D.d_key.end(), INF);
  thrust::fill(D.d_parent.begin(), D.d_parent.end(), -1);
  thrust::fill(D.d_inMST.begin(), D.d_inMST.end(), 0);
  D.d_key[0] = 0;
  D.d_parent[0] = 0;

  // raw pointers
  auto row_ptr = thrust::raw_pointer_cast(D.d_row_offsets.data());
  auto col_ptr = thrust::raw_pointer_cast(D.d_col_indices.data());
  auto w_ptr = thrust::raw_pointer_cast(D.d_weight.data());
  auto key_ptr = thrust::raw_pointer_cast(D.d_key.data());
  auto p_ptr = thrust::raw_pointer_cast(D.d_parent.data());
  auto in_ptr = thrust::raw_pointer_cast(D.d_inMST.data());
  auto bk_ptr = thrust::raw_pointer_cast(D.d_min_key.data());
  auto bi_ptr = thrust::raw_pointer_cast(D.d_min_idx.data());

  int threads = 256;
  int blocks = (num_vertices + threads - 1) / threads;
  size_t shared_mem =
      threads * (sizeof(weight_t) + sizeof(PrimGunrock::vertex_t));

  for (size_t i = 0; i < num_vertices; ++i) {
    // phase 1: block-level minima
    minKeyReduce<<<blocks, threads, shared_mem>>>(key_ptr, bk_ptr, bi_ptr,
                                                  num_vertices, INF);
    // phase 2: global min over block results
    minKeyReduce<<<1, threads, shared_mem>>>(bk_ptr, bk_ptr, bi_ptr, blocks,
                                             INF);

    // copy result
    weight_t minKey;
    int minIdx;
    cudaMemcpy(&minKey, bk_ptr, sizeof(weight_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&minIdx, bi_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    if (minKey == INF)
      break;

    int u = minIdx;
    D.d_inMST[u] = 1;
    cudaMemcpy(key_ptr + u, &INF, sizeof(weight_t), cudaMemcpyHostToDevice);
    auto parent = D.d_parent[u];
    if (u != parent)
      mst_edges.emplace_back(parent, u, minKey);

    relaxEdges<<<1, 128>>>(row_ptr, col_ptr, w_ptr, key_ptr, p_ptr, in_ptr, u);
    cudaDeviceSynchronize();
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
