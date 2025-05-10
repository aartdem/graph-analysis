#include "boruvka_spla.hpp"

#include <spla.hpp>
#include <fstream>
#include <sstream>

using namespace std;
using namespace spla;

namespace algos {

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
        const auto sparse_sz = Scalar::make_int(0);
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

    // for debug
    void print_vector(const ref_ptr<Vector> &v, const std::string &name = "") {
        std::cout << "-- " << name << " --\n";
        auto sz = Scalar::make_int(0);
        exec_v_count_mf(sz, v);
        auto buffer_int = std::vector<int>(sz->as_int());
        auto keys_view = MemView::make(buffer_int.data(), sz->as_int());
        auto values_view = MemView::make(buffer_int.data(), sz->as_int());
        v->read(keys_view, values_view);
        auto keys = static_cast<int *>(keys_view->get_buffer());
        auto values = static_cast<int *>(values_view->get_buffer());
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

        auto buffer_int = std::vector<int>(nnz);
        auto rows_view = MemView::make(buffer_int.data(), nnz);
        auto cols_view = MemView::make(buffer_int.data(), nnz);
        auto values_view = MemView::make(buffer_int.data(), nnz);
        m->read(rows_view, cols_view, values_view);

        auto rows = static_cast<int *>(rows_view->get_buffer());
        auto cols = static_cast<int *>(cols_view->get_buffer());
        auto values = static_cast<int *>(values_view->get_buffer());

        for (uint i = 0; i < nnz; ++i) {
            std::cout << rows[i] << " " << cols[i] << " " << values[i] << "\n";
        }
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

    pair<ref_ptr<Vector>, ref_ptr<Vector>> comb_min_product(
        const ref_ptr<Vector> &v,
        const ref_ptr<Matrix> &A) {

        const uint32_t n = v->get_n_rows();
        const auto inf_scalar = Scalar::make_int(INF);

        ref_ptr<Vector> min_values = Vector::make(n, INT);
        min_values->set_fill_value(inf_scalar);
        ref_ptr<Vector> min_indices = Vector::make(n, INT);
        min_indices->set_fill_value(Scalar::make_int(-1));

        auto filtered_A = Matrix::make(n, n, INT);
        filtered_A->set_fill_value(Scalar::make_int(0));

        vector<int> component(n);
        for (uint i = 0; i < n; i++) {
            v->get_int(i, component[i]);
        }

        for (uint i = 0; i < n; i++) {
            for (uint j = 0; j < n; j++) {
                int edge_weight;
                Status status = A->get_int(i, j, edge_weight);

                if (status == Status::Ok && edge_weight != 0 && component[i] != component[j]) {
                    filtered_A->set_int(i, j, edge_weight);
                }
            }
        }

        // Search for Minimum Weight for Each Vertex
        exec_m_reduce_by_row(min_values, filtered_A, MIN_INT, inf_scalar);

        for (uint i = 0; i < n; i++) {
            int min_val;
            min_values->get_int(i, min_val);

            if (min_val == INF) {
                continue;
            }

            for (uint j = 0; j < n; j++) {
                int val;
                Status status = filtered_A->get_int(i, j, val);
                if (status == Status::Ok && val == min_val) {
                    min_indices->set_int(i, j);
                    break;
                }
            }
        }

        return {min_values, min_indices};
    }

    void BoruvkaSpla::compute_() {
        mst = Vector::make(n, INT);
        const auto neg_one = Scalar::make_int(-1);
        mst->set_fill_value(neg_one);
        mst->fill_with(neg_one);
        weight = 0;

        const auto parent = static_cast<int *>(malloc(sizeof(int) * n));
        for (int i = 0; i < n; i++)
            parent[i] = i;

        const ref_ptr<Vector> v_parent = Vector::make(n, INT);

        while (true) {
            update_v_parent(v_parent, parent, n);

            // Search for Minimal Outgoing Edges for Each Vertex
            auto [min_values, min_indices] = comb_min_product(v_parent, a);

            bool changed = false;

            std::vector cedge_weight(n, INF);
            std::vector cedge_j(n, -1);
            std::vector cedge_u(n, -1);

            // Identifying the Optimal Edge for Each Component
            for (int i = 0; i < n; i++) {
                int p = find_root(parent, i);
                int edge_weight_i, edge_j_i;
                const Status s1 = min_values->get_int(i, edge_weight_i);
                const Status s2 = min_indices->get_int(i, edge_j_i);

                if (s1 == Status::Ok && s2 == Status::Ok &&
                    edge_j_i != -1 && edge_weight_i < cedge_weight[p]) {
                    cedge_weight[p] = edge_weight_i;
                    cedge_j[p] = edge_j_i;
                    cedge_u[p] = i;
                }
            }

            for (int p = 0; p < n; p++) {
                if (parent[p] == p && cedge_j[p] != -1) {
                    const int v = cedge_j[p];
                    const int root_v = find_root(parent, v);
                    if (p != root_v) {
                        const int u = cedge_u[p];
                        parent[p] = root_v;
                        changed = true;

                        mst->set_int(u, v);
                        weight += cedge_weight[p];
                    }
                }
            }

            if (!changed) break;
        }

        free(parent);
    }
}
