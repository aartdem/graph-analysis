#include "boruvka_spla.hpp"

#include <spla.hpp>
#include <fstream>
#include <sstream>
#include <cmath>
#include <set>
#include <unordered_set>

using namespace std;
using namespace spla;

namespace algos {

    using clock = chrono::steady_clock;
    constexpr uint32_t INF = 1e9;

    void BoruvkaSpla::load_graph(const filesystem::path &file_path) {
        ifstream input(file_path);
        string line;

        // Skip comments
        while (getline(input, line))
            if (line[0] != '%') break;

        int n_input, m_input, nnz_input;
        istringstream iss(line);
        iss >> n_input >> m_input >> nnz_input;
        if (n_input < 0) {
            throw runtime_error("Invalid mtx format, n < 0");
        }
        if (nnz_input < 0) {
            throw runtime_error("Invalid mtx format, nnz < 0");
        }
        n = n_input;
        edges = nnz_input;
        buffer_int = vector<int>(n);
        adj_list.resize(n);

        // Создаем матрицу и заполняем бесконечностями сразу
        constexpr uint32_t WEIGHT_SHIFT = 22;
        constexpr uint32_t INF_ENCODED = UINT32_MAX;

        a = Matrix::make(n, n, UINT);
        a->set_format(FormatMatrix::AccCsr);
        a->set_fill_value(Scalar::make_uint(INF_ENCODED));

        int u, v;
        uint32_t w;
        for (int i = 0; i < nnz_input; ++i) {
            input >> u >> v >> w;
            u--;
            v--;
            if (u < 0 || v < 0 || u > n || v > n) {
                throw runtime_error("Invalid graph, incorrect vertex numbers");
            }
            if (w <= 0) {
                throw runtime_error("Invalid graph, negative edges");
            }
            if (w >= (1 << 10)) {
                throw runtime_error("Edge weight too large, max is 1023");
            }

            if (u != v) {
                uint32_t encoded_u_v = (w << WEIGHT_SHIFT) | v;
                uint32_t encoded_v_u = (w << WEIGHT_SHIFT) | u;

                a->set_uint(u, v, encoded_u_v);
                a->set_uint(v, u, encoded_v_u);

                adj_list[u].emplace_back(v);
                adj_list[v].emplace_back(u);
            }
        }
    }

    chrono::seconds BoruvkaSpla::compute() {
        const auto start = clock::now();
        compute_();
        const auto end = clock::now();
        return chrono::duration_cast<chrono::seconds>(end - start);
    }

    Tree BoruvkaSpla::get_result() {
        const auto sparse_sz = Scalar::make_int(0);
        vector p(n, -1);
        ref_ptr<Vector> zero_vec = Vector::make(n, INT);
        zero_vec->fill_with(spla::Scalar::make_int(0));
        auto mst1 = Vector::make(n, INT);
        mst1->set_fill_value(spla::Scalar::make_int(-1));
        mst1->fill_with(spla::Scalar::make_int(-1));
        exec_v_eadd(mst1, mst, zero_vec, spla::PLUS_INT);
        exec_v_count_mf(sparse_sz, mst);
        auto keys_view = MemView::make(buffer_int.data(), sparse_sz->as_int());
        auto values_view = MemView::make(buffer_int.data(), sparse_sz->as_int());
        mst1->read(keys_view, values_view);
        const auto keys = static_cast<int *>(keys_view->get_buffer());
        const auto values = static_cast<int*>(values_view->get_buffer());
        for (int i = 0; i < sparse_sz->as_int(); i++) {
            p[keys[i]] = static_cast<int>(std::round(values[i]));
        }
        return Tree{n, p, weight};
    }

    // for debug
    void print_vector(const ref_ptr<Vector> &v, const std::string &name = "") {
        std::cout << "-- " << name << " --\n";
        auto sz = Scalar::make_uint(0);
        exec_v_count_mf(sz, v);
        auto buffer_int = std::vector<uint>(sz->as_int());
        auto keys_view = MemView::make(buffer_int.data(), sz->as_uint());
        auto values_view = MemView::make(buffer_int.data(), sz->as_uint());
        v->read(keys_view, values_view);
        auto keys = static_cast<uint32_t *>(keys_view->get_buffer());
        auto values = static_cast<uint32_t*>(values_view->get_buffer());
        for (int i = 0; i < sz->as_int(); i++) {
            std::cout << keys[i] << ' ' << values[i] << '\n';
        }
    }

    size_t count_nonzero_elements(const ref_ptr<Matrix>& mat) {
        size_t count = 0;
        const uint n_rows = mat->get_n_rows();
        const uint n_cols = mat->get_n_cols();

        for (uint i = 0; i < n_rows; ++i) {
            for (uint j = 0; j < n_cols; ++j) {
                int val;
                Status status = mat->get_int(i, j, val);
                if (status == Status::Ok && val != INF) {
                    count++;
                }
            }
        }
        return count;
    }

    void print_matrix(const ref_ptr<Matrix>& m, const std::string& name = "") {
        std::cout << "-- " << name << " --\n";
        int nnz = count_nonzero_elements(m);

        auto buffer_int = std::vector<uint32_t>(nnz);
        auto rows_view = MemView::make(buffer_int.data(), nnz);
        auto cols_view = MemView::make(buffer_int.data(), nnz);
        auto values_view = MemView::make(buffer_int.data(), nnz);
        m->read(rows_view, cols_view, values_view);

        auto rows = static_cast<uint32_t *>(rows_view->get_buffer());
        auto cols = static_cast<uint32_t *>(cols_view->get_buffer());
        auto values = static_cast<uint32_t*>(values_view->get_buffer());

        for (uint i = 0; i < nnz; ++i) {
            std::cout << rows[i] << " " << cols[i] << " " << values[i] << "\n";
        }
    }


    void BoruvkaSpla::compute_() {
        mst = Vector::make(n, INT);
        mst->set_fill_value(Scalar::make_int(-1));
        mst->fill_with(Scalar::make_int(-1));
        weight = 0;

        constexpr uint32_t WEIGHT_SHIFT = 22;
        constexpr uint32_t INDEX_MASK = (1 << 22) - 1;
        constexpr uint32_t INF_ENCODED = UINT32_MAX;

        std::vector<uint32_t> f_array(n);
        for (uint v = 0; v < n; v++) {
            f_array[v] = v;
        }

        const auto edge = Vector::make(n, UINT);
        std::vector edge_array(n, INF_ENCODED);
        std::vector cedge_array(n, INF_ENCODED);

        std::vector<std::vector<uint>> comp_to_vertices(n);
        for (uint i = 0; i < n; i++) {
            comp_to_vertices[i].push_back(i);
        }

        // Компоненты, которые изменились на текущей итерации
        std::set<uint> modified_comps;

        for (int iter = 0; iter < n; iter++) {
            cout << "Weight: " << weight << ", Iteration: " << iter << '\n';
            edge->set_fill_value(Scalar::make_uint(INF_ENCODED));
            edge->fill_with(Scalar::make_uint(INF_ENCODED));

            // Находим минимальное ребро для каждой вершины
            exec_m_reduce_by_row(edge, a, MIN_UINT, Scalar::make_uint(INF_ENCODED));

            // Находим минимальное ребро для каждой компоненты
            ranges::fill(cedge_array, INF_ENCODED);
            for (uint v = 0; v < n; v++) {
                const uint root = f_array[v];

                uint edge_v;
                edge->get_uint(v, edge_v);
                edge_array[v] = edge_v;

                if (edge_v < cedge_array[root]) {
                    cedge_array[root] = edge_v;
                }
            }

            // Очищаем список изменённых компонент
            modified_comps.clear();

            // Добавляем рёбра в MST и объединяем компоненты
            bool added_edges = false;
            for (uint i = 0; i < n; i++) {
                uint comp_i = f_array[i];

                if (i == comp_i) {  // i - корень компоненты
                    uint cedge_i = cedge_array[i];

                    if (cedge_i != INF_ENCODED) {
                        const uint dest = cedge_i & INDEX_MASK;
                        const uint w = cedge_i >> WEIGHT_SHIFT;

                        // Находим источник минимального ребра
                        uint src = UINT_MAX;
                        for (uint v : comp_to_vertices[comp_i]) {
                            if (edge_array[v] == cedge_i) {
                                src = v;
                                break;
                            }
                        }

                        if (src != UINT_MAX) {
                            mst->set_int(src, dest);
                            weight += w;
                            added_edges = true;
                        }

                        // Объединяем компоненты
                        uint comp_dest = f_array[dest];
                        const uint new_comp = i < comp_dest ? i : comp_dest;

                        // Запоминаем изменённые компоненты
                        modified_comps.insert(new_comp);

                        // Объединяем списки вершин
                        if (new_comp != comp_i) {
                            comp_to_vertices[new_comp].insert(
                                comp_to_vertices[new_comp].end(),
                                comp_to_vertices[comp_i].begin(),
                                comp_to_vertices[comp_i].end());
                            comp_to_vertices[comp_i].clear();
                        }

                        if (new_comp != comp_dest) {
                            comp_to_vertices[new_comp].insert(
                                comp_to_vertices[new_comp].end(),
                                comp_to_vertices[comp_dest].begin(),
                                comp_to_vertices[comp_dest].end());
                            comp_to_vertices[comp_dest].clear();
                        }

                        // Обновляем компоненты в f_array
                        for (uint v : comp_to_vertices[new_comp]) {
                            f_array[v] = new_comp;
                        }
                    }
                }
            }

            if (!added_edges) break;

            // Удаляем рёбра внутри компонент
            for (const uint comp : modified_comps) {
                const auto& vertices = comp_to_vertices[comp];
                unordered_set set_vertices(vertices.begin(), vertices.end());

                for (uint i : set_vertices) {
                    for (uint j : adj_list[i]) {
                        if (set_vertices.contains(j)) {
                            a->set_uint(i, j, INF_ENCODED);
                        }
                    }
                }
            }
        }
    }
}
