// boruvka_gunrock.hpp
#pragma once

#include "common/mst_algorithm.hpp"
#include "common/tree.hpp"
#include <chrono>
#include <cuda_runtime.h> // даёт определения threadIdx, blockIdx, blockDim, gridDim
#include <device_launch_parameters.h>
#include <gunrock/algorithms/algorithms.hxx> // Gunrock core (включает необходимые заголовки)
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace algos {

class BoruvkaGunrock : public MstAlgorithm {
  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  // Поля для хранения графа
  vertex_t num_vertices;
  edge_t num_edges;
  thrust::device_vector<vertex_t>
      d_src; // массив номеров исходных вершин для каждого ребра (device)
  thrust::device_vector<vertex_t>
      d_dst; // массив номеров конечных вершин для каждого ребра
  thrust::device_vector<weight_t> d_weight; // массив весов ребер

  // Результат: список рёбер MST (на хосте)
  std::vector<std::tuple<vertex_t, vertex_t, weight_t>> mst_edges;

public:
  // Вспомогательные структуры для редукции
  struct EdgePair {
    weight_t w;
    edge_t idx;
  };
  // Функтор для сравнения EdgePair по весу (для редукции минимума)
  struct MinEdgeOp {
    __host__ __device__ EdgePair operator()(const EdgePair &a,
                                            const EdgePair &b) const {
      return (a.w <= b.w) ? a : b;
    }
  };
  // Загрузка графа из файла в формате Matrix Market (MTX)
  void load_graph(const std::filesystem::path &file_path) override;
  // Основной метод вычисления MST
  std::chrono::seconds compute() override;
  // Возвращает результат в формате Tree
  Tree get_result() override;
};

} // namespace algos