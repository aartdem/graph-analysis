#include "boruvka_spla.hpp"

#include <spla.hpp>
#include <fstream>
#include <sstream>

using namespace std;
using namespace spla;

namespace algos {
    std::pair<ref_ptr<Vector>, ref_ptr<Vector>> comb_min_product(
        const ref_ptr<Vector> &v,
        const ref_ptr<Matrix> &A
    );
    int find_root(int* parent, int x);
    void update_v_parent(const ref_ptr<Vector> &f, int* parent, int n);

    using clock = chrono::steady_clock;
    constexpr int INF = 1e9;

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
        a = Matrix::make(n, n, INT);

        int u, v, w;
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
            if (u != v) {
                a->set_int(u, v, w);
                a->set_int(v, u, w);
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
        const auto sparse_sz = Scalar::make_uint(0);
        auto buffer_int = vector<int>(n);
        vector p(n, -1);
        exec_v_count_mf(sparse_sz, mst);
        auto keys_view = MemView::make(buffer_int.data(), sparse_sz->as_int());
        auto values_view = MemView::make(buffer_int.data(), sparse_sz->as_int());
        mst->read(keys_view, values_view);
        const auto keys = static_cast<int *>(keys_view->get_buffer());
        const auto values = static_cast<int *>(values_view->get_buffer());
        for (int i = 0; i < sparse_sz->as_int(); i++) {
            p[keys[i]] = values[i];
        }
        return Tree{n, p, weight};
    }

    void print_vector(const ref_ptr<Vector> &v, const std::string &name = "") {
        std::cout << "-- " << name << " --\n";
        auto sz = Scalar::make_int(0);
        exec_v_count_mf(sz, v);
        auto buffer_int = std::vector<int>(sz->as_int());
        auto keys_view = MemView::make(buffer_int.data(), sz->as_int());
        auto values_view = MemView::make(buffer_int.data(), sz->as_int());
        v->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (int *) values_view->get_buffer();
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
                if (status == Status::Ok && val != 0) {
                    count++;
                }
            }
        }
        return count;
    }

    void print_matrix(const ref_ptr<Matrix>& m, const std::string& name = "") {
        std::cout << "-- " << name << " --\n";
        int foo = count_nonzero_elements(m);

        // Получаем все ненулевые элементы
        auto buffer_int = std::vector<int>(foo);
        auto rows_view = MemView::make(buffer_int.data(), foo);
        auto cols_view = MemView::make(buffer_int.data(), foo);
        auto values_view = MemView::make(buffer_int.data(), foo);
        m->read(rows_view, cols_view, values_view);

        auto rows = (int *) rows_view->get_buffer();
        auto cols = (int *) cols_view->get_buffer();
        auto values = (int *) values_view->get_buffer();

        // Печатаем в формате "i j value"
        for (uint i = 0; i < foo; ++i) {
            std::cout << rows[i] << " " << cols[i] << " " << values[i] << "\n";
        }
    }

    void BoruvkaSpla::compute_() {
        // Инициализация mst и веса
        mst = Vector::make(n, INT);
        const auto neg_one = Scalar::make_int(-1);
        mst->set_fill_value(neg_one);
        mst->fill_with(neg_one);
        weight = 0;

        const auto parent = static_cast<int *>(malloc(sizeof(int) * n));
        for (int i = 0; i < n; i++)
            parent[i] = i;

        const ref_ptr<Vector> f = Vector::make(n, INT);

        while (true) {
            update_v_parent(f, parent, n);
            auto [min_values, min_indices] = comb_min_product(f, a);

            // Выделяем память для хранения рёбер
            auto cedge_weight = static_cast<int *>(malloc(sizeof(int) * n));
            auto cedge_j = static_cast<int *>(malloc(sizeof(int) * n));
            auto cedge_u = static_cast<int *>(malloc(sizeof(int) * n)); // Для хранения вершины u
            for (int p = 0; p < n; p++) {
                cedge_weight[p] = INF;
                cedge_j[p] = -1;
                cedge_u[p] = -1;
            }

            // Находим минимальные рёбра для каждой компоненты
            for (int i = 0; i < n; i++) {
                int p = find_root(parent, i);
                int edge_weight_i, edge_j_i;
                min_values->get_int(i, edge_weight_i);
                min_indices->get_int(i, edge_j_i);

                if (edge_weight_i < cedge_weight[p]) {
                    cedge_weight[p] = edge_weight_i;
                    cedge_j[p] = edge_j_i;
                    cedge_u[p] = i; // Сохраняем вершину i
                }
            }

            // Объединяем компоненты и добавляем рёбра в mst
            bool changed = false;
            for (int p = 0; p < n; p++) {
                if (parent[p] == p && cedge_j[p] != -1) {
                    const int v = cedge_j[p];
                    if (const int root_v = find_root(parent, v); p != root_v) {
                        const int u = cedge_u[p];
                        mst->set_int(u, v);
                        weight += cedge_weight[p];

                        parent[p] = root_v;
                        changed = true;
                    }
                }
            }

            free(cedge_weight);
            free(cedge_j);
            free(cedge_u);

            if (!changed) break;

            // Фильтруем рёбра внутри компонент
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int val;
                    a->get_int(i, j, val);
                    if (val != 0 && find_root(parent, i) == find_root(parent, j)) {
                        a->set_int(i, j, 0);
                    }
                }
            }
        }
        free(parent);
    }

    pair<ref_ptr<Vector>, ref_ptr<Vector>> row_min_and_argmin(const ref_ptr<Matrix> &A) {
        const uint n_rows = A->get_n_rows();
        const uint n_cols = A->get_n_cols();

        ref_ptr<Vector> min_values = Vector::make(n_rows, INT);
        ref_ptr<Vector> min_indices = Vector::make(n_rows, INT);

#pragma omp parallel for
        for (uint i = 0; i < n_rows; i++) {
            int min_val = INF;
            int min_idx = -1;

            for (uint j = 0; j < n_cols; j++) {
                int val;
                Status status = A->get_int(i, j, val);

                if (status == Status::Ok && val != 0.0f && val < min_val) { // фильтрация что в разных компонентах?
                    min_val = val;
                    min_idx = j;
                }
            }

            if (min_idx >= 0) {
                min_values->set_int(i, min_val);
                min_indices->set_int(i, min_idx);
            } else {
                min_values->set_int(i, INF);
                min_indices->set_int(i, -1);
            }
        }

        return {min_values, min_indices};
    }

    pair<ref_ptr<Vector>, ref_ptr<Vector>> comb_min_product(const ref_ptr<Vector> &v, const ref_ptr<Matrix> &A) {
        const uint32_t n = v->get_n_rows();

        vector<int> component(n);
        for (uint i = 0; i < n; i++) {
            v->get_int(i, component[i]);
        }

        ref_ptr<Vector> min_values = Vector::make(n, INT);
        ref_ptr<Vector> min_indices = Vector::make(n, INT);

        for (uint i = 0; i < n; i++) {
            int min_val = INF;
            int min_idx = -1;

            for (uint j = 0; j < n; j++) {
                int edge_weight;
                Status status = A->get_int(i, j, edge_weight);

                if (status == Status::Ok && edge_weight != 0 && component[i] != component[j]) {
                    if (edge_weight < min_val) {
                        min_val = edge_weight;
                        min_idx = j;
                    }
                }
            }

            min_values->set_int(i, min_val);
            min_indices->set_int(i, min_idx);
        }

        return {min_values, min_indices};
    }


    int find_root(int *parent, int x) {
        if (parent[x] != x) {
            parent[x] = find_root(parent, parent[x]);
        }
        return parent[x];
    }

    void update_v_parent(const ref_ptr<Vector> &f, int* parent, int n) {
        for (int i = 0; i < n; ++i) {
            f->set_int(i, find_root(parent, i));
        }
    }
}