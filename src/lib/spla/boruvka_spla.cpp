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
    void update_v_parent(ref_ptr<Vector> v, int* parent, int n);

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
        a = Matrix::make(n, n, spla::INT);

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

    using clock = chrono::steady_clock;
    constexpr int INF = 1e9;

    chrono::seconds BoruvkaSpla::compute() {
        auto start = clock::now();
        compute_();
        auto end = clock::now();
        return chrono::duration_cast<chrono::seconds>(end - start);
    }

    Tree BoruvkaSpla::get_result() {
        auto sparse_sz = Scalar::make_uint(0);
        auto buffer_int = vector<int>(n);
        vector<int> p(n, -1);
        exec_v_count_mf(sparse_sz, mst);
        auto keys_view = MemView::make(buffer_int.data(), sparse_sz->as_int());
        auto values_view = MemView::make(buffer_int.data(), sparse_sz->as_int());
        mst->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (int *) values_view->get_buffer();
        for (int i = 0; i < sparse_sz->as_int(); i++) {
            p[keys[i]] = values[i];
        }
        return Tree{n, p, weight};
    }

    void BoruvkaSpla::compute_() {
        // надо еще научиться обрабатывать лес, пока что будет только дерево для связного графа
        auto parent = static_cast<int *>(malloc(sizeof(int) * n));
        for (int i = 0; i < n; i++)
            parent[i] = i;
    
        ref_ptr<Vector> f = Vector::make(n, INT);
        for (int i = 0; i < n; ++i) {
            f->set_int(i, i);
        }
    
        while (true)
        {
            update_v_parent(f, parent, n);
            auto [min_values, min_indices] = comb_min_product(f, a);
    
            // calculate of cedge - норм ли каждый раз их тут инициализировать?
            auto cedge_weight = static_cast<int *>(malloc(sizeof(int) * n));
            auto cedge_j = static_cast<int *>(malloc(sizeof(int) * n));
            for (int p = 0; p < n; p++) {
                cedge_weight[p] = INF;
                cedge_j[p] = -1;
            }
    
            for (int i = 0; i < n; i++) {
                int p = parent[i];
                int edge_weight_i, edge_j_i;
                min_values->get_int(i, edge_weight_i);
                min_indices->get_int(i, edge_j_i);
            
                if (edge_weight_i < cedge_weight[p]) {
                    cedge_weight[p] = edge_weight_i;
                    cedge_j[p] = edge_j_i;
                }
            }
    
            // calculate of parent
            bool changed = false;
            for (int p = 0; p < n; p++) {
                if (parent[p] == p && cedge_j[p] != -1) {
                    int new_parent = find_root(parent, cedge_j[p]);
                    if (p != new_parent) {
                        parent[p] = new_parent;
                        changed = true;
                    }
                }
            }
            if (!changed) break;
    
            // filter edges in A
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int val;
                    a->get_int(i, j, val);
                    if (val != 0 && find_root(parent, i) == find_root(parent, j)) {
                        a->set_int(i, j, 0);
                    }
                }
            }
    
            free(cedge_weight);
            free(cedge_j);
        }
    }

    pair<ref_ptr<Vector>, ref_ptr<Vector>> row_min_and_argmin(ref_ptr<Matrix> A) {
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
        ref_ptr<Matrix> F = Matrix::make(n, n, INT);
        for (size_t row = 0; row < n; row++)
        {
            int col;
            v->get_int(row, col);
            if (col != -1) {
                F->set_int(row, col, 1);
            }
        }
        ref_ptr<Scalar> zero = Scalar::make_int(0);
    
        ref_ptr<Matrix> W = Matrix::make(n, n, INT);
        exec_mxm(W, A, F, MULT_INT, MIN_INT, zero);
    
        return row_min_and_argmin(W);
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