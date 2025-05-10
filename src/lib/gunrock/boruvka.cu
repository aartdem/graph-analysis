
// #include "boruvka.hxx"

// namespace algos {
// // Загрузка графа из файла в формате Matrix Market (MTX)
// void BoruvkaGunrock::load_graph(const std::filesystem::path &file_path) {
//   using namespace gunrock;
//   io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
//   auto loaded = mm.load(file_path);
//   auto &coo = std::get<1>(loaded);

//   // Определяем количество рёбер и вершин
//   auto host_rows = coo.row_indices;
//   auto host_cols = coo.column_indices;
//   auto host_vals = coo.nonzero_values;
//   edge_t original_edges = static_cast<edge_t>(host_rows.size());
//   // Определяем максимальный идентификатор вершины
//   vertex_t max_v = 0;
//   for (edge_t i = 0; i < original_edges; ++i) {
//     max_v = std::max({max_v, host_rows[i], host_cols[i]});
//   }
//   num_vertices = max_v + 1;

//   // Подготавливаем массивы ребер (неориентированный, дублируем)
//   thrust::host_vector<vertex_t> h_src;
//   thrust::host_vector<vertex_t> h_dst;
//   thrust::host_vector<weight_t> h_weight;
//   h_src.reserve(2 * original_edges);
//   h_dst.reserve(2 * original_edges);
//   h_weight.reserve(2 * original_edges);

//   for (edge_t i = 0; i < original_edges; ++i) {
//     auto u = host_rows[i];
//     auto v = host_cols[i];
//     auto w = host_vals[i];
//     h_src.push_back(u);
//     h_dst.push_back(v);
//     h_weight.push_back(w);
//     if (u != v) {
//       h_src.push_back(v);
//       h_dst.push_back(u);
//       h_weight.push_back(w);
//     }
//   }
//   num_edges = static_cast<edge_t>(h_src.size());

//   // Копируем на GPU
//   d_src = h_src;
//   d_dst = h_dst;
//   d_weight = h_weight;
// }

// // Основной метод вычисления MST
// std::chrono::seconds BoruvkaGunrock::compute() {
//   mst_edges.clear();
//   auto start = std::chrono::steady_clock::now();
//   if (num_vertices == 0)
//     return {};
//   // Инициализация массива компонент: в начале каждая вершина в своей
//   // компоненте (comp[v] = v)
//   thrust::device_vector<vertex_t> comp(num_vertices);
//   thrust::sequence(comp.begin(), comp.end(), 0); // comp[i] = i

//   // Массивы-буферы для ключей и значений при поиске минимальных ребер
//   thrust::device_vector<vertex_t> keys(2 * num_edges);
//   thrust::device_vector<EdgePair> vals(2 * num_edges);
//   thrust::device_vector<vertex_t> comp_keys_out(
//       num_vertices); // выходные ключи (компоненты)
//   thrust::device_vector<EdgePair> comp_vals_out(
//       num_vertices); // выходные значения (мин.ребро для компоненты)

//   bool merged = true;
//   // Выполняем итерации, пока происходит объединение компонент
//   while (merged) {
//     merged = false;
//     // Шаг 1: Для каждого ребра записываем кандидаты в массивы keys/vals.
//     // Каждый ребро (u,v) дает два кандидата: для компоненты u и для
//     // компоненты v. Если вершины в одной компоненте, помечаем кандидатов как
//     // невалидных (ключ = -1).
//     vertex_t *comp_ptr = thrust::raw_pointer_cast(comp.data());
//     vertex_t *src_ptr = thrust::raw_pointer_cast(d_src.data());
//     vertex_t *dst_ptr = thrust::raw_pointer_cast(d_dst.data());
//     weight_t *w_ptr = thrust::raw_pointer_cast(d_weight.data());
//     vertex_t *keys_ptr = thrust::raw_pointer_cast(keys.data());
//     EdgePair *vals_ptr = thrust::raw_pointer_cast(vals.data());
//     edge_t m = num_edges;
// // Запускаем CUDA-ядро одной нитью на ребро:
// // (В реальном коде следует проверить CUDA ошибки, для краткости опущено)
// #pragma omp target teams distribute parallel for is_device_ptr(                \
//         comp_ptr, src_ptr, dst_ptr, w_ptr, keys_ptr, vals_ptr)
//     for (edge_t e = 0; e < m; ++e) {
//       vertex_t u = src_ptr[e];
//       vertex_t v = dst_ptr[e];
//       weight_t w = w_ptr[e];
//       vertex_t comp_u = comp_ptr[u];
//       vertex_t comp_v = comp_ptr[v];
//       if (comp_u == comp_v) {
//         // Ребро внутри одной компоненты — помечаем как невалидное
//         keys_ptr[2 * e] = -1;
//         vals_ptr[2 * e] = {w, e};
//         keys_ptr[2 * e + 1] = -1;
//         vals_ptr[2 * e + 1] = {w, e};
//       } else {
//         // Ребро между разными компонентами: добавляем два направления
//         keys_ptr[2 * e] = comp_u;
//         vals_ptr[2 * e] = {w, e};
//         keys_ptr[2 * e + 1] = comp_v;
//         vals_ptr[2 * e + 1] = {w, e};
//       }
//     }
//     // Шаг 2: Фильтрация невалидных записей (где ключ = -1)
//     auto zip_begin = thrust::make_zip_iterator(
//         thrust::make_tuple(keys.begin(), vals.begin()));
//     auto zip_end =
//         thrust::make_zip_iterator(thrust::make_tuple(keys.end(), vals.end()));
//     auto new_end = thrust::remove_if(
//         zip_begin, zip_end,
//         [] __host__ __device__(const thrust::tuple<vertex_t, EdgePair> &kv) {
//           vertex_t comp_id = thrust::get<0>(kv);
//           return comp_id == -1;
//         });
//     size_t new_size = new_end - zip_begin;
//     if (new_size == 0) {
//       // Нет ребер между компонентами – MST построено (либо несколько
//       // изолированных компонент)
//       break;
//     }
//     // Обрезаем векторы до new_size после удаления невалидных элементов
//     keys.resize(new_size);
//     vals.resize(new_size);
//     // Шаг 3: Группировка по компонентам и выбор минимального ребра для каждой
//     // компоненты Сортируем по ключам-компонентам, чтобы одинаковые компоненты
//     // шли подряд
//     thrust::sort_by_key(keys.begin(), keys.end(), vals.begin());
//     // Выполняем reduce_by_key для нахождения минимального ребра по весу для
//     // каждой группы ключей
//     auto reduce_end = thrust::reduce_by_key(
//         keys.begin(), keys.end(), vals.begin(), comp_keys_out.begin(),
//         comp_vals_out.begin(), thrust::equal_to<vertex_t>(), MinEdgeOp());
//     size_t num_comps_found =
//         reduce_end.first -
//         comp_keys_out.begin(); // число компонент, для которых найдено ребро
//     if (num_comps_found == 0) {
//       break; // не найдено ни одного внешнего ребра (на случай неполной связи)
//     }
//     // Шаг 4: Объединение компонент по выбранным ребрам.
//     // Для каждой записи (comp_id -> EdgePair{w, e}) из результата:
//     // определяем соседнюю компоненту и выполняем слияние (переназначение
//     // меток). newComp_map[x] будет новой меткой для компоненты с
//     // идентификатором x после объединения.
//     thrust::device_vector<vertex_t> newComp_map(num_vertices);
//     // Изначально newComp_map[i] = i (каждая компонента остается сама собой,
//     // если не будет переопределена)
//     thrust::sequence(newComp_map.begin(), newComp_map.end(), 0);
//     vertex_t *newcomp_ptr = thrust::raw_pointer_cast(newComp_map.data());
//     vertex_t *out_keys_ptr = thrust::raw_pointer_cast(comp_keys_out.data());
//     EdgePair *out_vals_ptr = thrust::raw_pointer_cast(comp_vals_out.data());
//     size_t k = num_comps_found;
// // Ядро: объединяем компоненты (связываем "старшую" компоненту с "младшей")
// #pragma omp target teams distribute parallel for is_device_ptr(                \
//         comp_ptr, src_ptr, dst_ptr, out_keys_ptr, out_vals_ptr, newcomp_ptr)
//     for (size_t i = 0; i < k; ++i) {
//       vertex_t comp_id = out_keys_ptr[i];
//       EdgePair ep = out_vals_ptr[i];
//       edge_t e = ep.idx;
//       // Восстанавливаем концы ребра
//       vertex_t u = src_ptr[e];
//       vertex_t v = dst_ptr[e];
//       // Определяем метки компонентов концов ребра (могут совпадать с comp_id
//       // или быть другой стороной)
//       vertex_t comp_u = comp_ptr[u];
//       vertex_t comp_v = comp_ptr[v];
//       if (comp_u == comp_v) {
//         continue; // обе вершины уже в одной компоненте
//       }
//       // Выбираем нового представителя компоненты (минимальный id для
//       // детерминизма)
//       vertex_t root = (comp_u < comp_v ? comp_u : comp_v);
//       vertex_t other = (comp_u < comp_v ? comp_v : comp_u);
//       // Объединяем: перенаправляем 'other' к 'root'
//       newcomp_ptr[other] = root;
//     }
//     // Сжимаем цепочки объединения (pointer jumping для newComp_map)
//     bool updated = true;
//     int iter = 0;
//     while (updated && iter < 10) { // максимум 10 итераций для безопасности
//       updated = false;
// // Одновременное обновление: newComp_map[x] = newComp_map[newComp_map[x]]
// #pragma omp target teams distribute parallel for is_device_ptr(newcomp_ptr)
//       for (vertex_t i = 0; i < num_vertices; ++i) {
//         vertex_t parent = newcomp_ptr[i];
//         vertex_t grandparent = newcomp_ptr[parent];
//         if (grandparent != parent) {
//           newcomp_ptr[i] = grandparent;
//           updated = true;
//         }
//       }
//       iter++;
//     }
// // Обновляем метки компонент для всех вершин: comp[v] = newComp_map[ comp[v] ]
// #pragma omp target teams distribute parallel for is_device_ptr(                \
//         comp_ptr, newcomp_ptr)
//     for (vertex_t v = 0; v < num_vertices; ++v) {
//       comp_ptr[v] = newcomp_ptr[comp_ptr[v]];
//     }
//     // Шаг 5: Добавление выбранных ребер в результат MST (на хосте).
//     // Копируем пары (вес, индекс ребра) из comp_vals_out и соответствующие
//     // компоненты из comp_keys_out на хост.
//     std::vector<vertex_t> out_keys_host(num_comps_found);
//     std::vector<EdgePair> out_vals_host(num_comps_found);
//     thrust::copy(comp_keys_out.begin(), comp_keys_out.begin() + num_comps_found,
//                  out_keys_host.begin());
//     thrust::copy(comp_vals_out.begin(), comp_vals_out.begin() + num_comps_found,
//                  out_vals_host.begin());
//     // Чтобы избежать дублирования одного и того же ребра дважды, используем
//     // массив пометок уже добавленных ребер.
//     static std::vector<char>
//         edge_used; // статический, чтобы не перераспределять каждый раз (можно
//                    // и как поле класса)
//     edge_used.assign(num_edges, 0);
//     for (size_t i = 0; i < num_comps_found; ++i) {
//       edge_t e = out_vals_host[i].idx;
//       weight_t w = out_vals_host[i].w;
//       // Найдем фактические вершины этого ребра
//       // (После последнего обновления comp, вершины u и v уже в одной
//       // компоненте)
//       vertex_t u = (vertex_t)d_src[e];
//       vertex_t v = (vertex_t)d_dst[e];
//       if (!edge_used[e]) {
//         edge_used[e] = 1;
//         mst_edges.emplace_back(u, v, w);
//       }
//     }
//     merged = true; // произошли объединения, продолжаем цикл
//   } // конец while

//   auto end = std::chrono::steady_clock::now();
//   return std::chrono::duration_cast<std::chrono::seconds>(end - start);
// }

// Tree BoruvkaGunrock::get_result() {
//   // Инициализируем дерево: parent = -1 для всех вершин
//   std::vector<int> parent(num_vertices, -1);
//   float total_weight = 0.0f;
//   // Заполняем parent по списку рёбер MST
//   for (auto &e : mst_edges) {
//     vertex_t u, v;
//     weight_t w;
//     std::tie(u, v, w) = e;
//     // Если v еще без родителя, делаем u его родителем
//     if (parent[v] == -1 && parent[u] != v) {
//       parent[v] = u;
//     } else if (parent[u] == -1 && parent[v] != u) {
//       parent[u] = v;
//     }
//     total_weight += w;
//   }
//   return Tree(num_vertices, parent, total_weight);
// }
// } // namespace algos