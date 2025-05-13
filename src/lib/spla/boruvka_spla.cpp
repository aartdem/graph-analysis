#include "boruvka_spla.hpp"

#include <spla.hpp>
#include <fstream>
#include <sstream>
#include <cmath>

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
        a = Matrix::make(n, n, UINT);

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
            if (u != v) {
                a->set_uint(u, v, w);
                a->set_uint(v, u, w);
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

        const auto f = Vector::make(n, UINT);
        const auto edge = Vector::make(n, UINT);
        const auto cedge = Vector::make(n, UINT);
        const auto t = Vector::make(n, UINT);      // Аналог t в GraphBLAS
        const auto mask = Vector::make(n, UINT);   // Аналог mask в GraphBLAS

        for (uint v = 0; v < n; v++) {
            f->set_uint(v, v);
        }

        cout << "инициализирую матрицу S" << endl;

        const auto S = Matrix::make(n, n, UINT);
        S->set_fill_value(Scalar::make_uint(INF_ENCODED));

        for (uint src = 0; src < n; src++) {
            for (uint dst = 0; dst < n; dst++) {
                uint w;
                a->get_uint(src, dst, w);
                if ( w > 0 && w < 1 << 10) {
                    const uint32_t encoded = w << WEIGHT_SHIFT | dst;
                    S->set_uint(src, dst, encoded);
                } else {
                    // кинуть ошибку
                    //S->set_uint(src, dst, INF_ENCODED);
                }
            }
        }

        std::vector<uint> parent(n);

        uint nvals = n * n;
        for (int iter = 0; nvals > 0 && iter < n; iter++) {
            cout << weight << '\n';
            edge->set_fill_value(Scalar::make_uint(INF_ENCODED));
            edge->fill_with(Scalar::make_uint(INF_ENCODED));

            // для каждой вершины находим минимальное ребро
            exec_m_reduce_by_row(edge, S, MIN_UINT, Scalar::make_uint(INF_ENCODED));

            // 2. Минимальное ребро для каждой компоненты (вес + индекс вершины, куда идет)
            cedge->set_fill_value(Scalar::make_uint(INF_ENCODED));
            cedge->fill_with(Scalar::make_uint(INF_ENCODED));

            for (uint v = 0; v < n; v++) {
                // нашли компоненту вершины
                uint root;
                f->get_uint(v, root);

                uint edge_v;
                edge->get_uint(v, edge_v);

                uint cedge_root;
                cedge->get_uint(root, cedge_root);

                if (edge_v < cedge_root) {
                    cedge->set_uint(root, edge_v);
                }
            }

            // 3. Добавляем рёбра в MST и объединяем компоненты
            bool added_edges = false;
            for (uint i = 0; i < n; i++) {
                uint comp_i;
                f->get_uint(i, comp_i);

                if (i == comp_i) {  // i - корень компоненты
                    uint cedge_i;
                    cedge->get_uint(i, cedge_i);

                    if (cedge_i != INF_ENCODED) {
                        const uint dest = cedge_i & INDEX_MASK;
                        const uint w = cedge_i >> WEIGHT_SHIFT;

                        for (uint src = 0; src < n; src++) { // находим вершину с минимальным ребром в компоненте
                            uint comp_src;
                            f->get_uint(src, comp_src);

                            if (comp_src == comp_i) {
                                uint edge_src;
                                edge->get_uint(src, edge_src);

                                if (edge_src == cedge_i) {
                                    mst->set_int(src, dest);
                                    weight += w;
                                    added_edges = true;
                                    break;
                                }
                            }
                        }

                        uint comp_dest;
                        f->get_uint(dest, comp_dest);

                        const uint new_comp = i < comp_dest ? i : comp_dest;

                        for (uint v = 0; v < n; v++) {
                            uint comp_v;
                            f->get_uint(v, comp_v);

                            if (comp_v == i || comp_v == comp_dest) {
                                f->set_uint(v, new_comp);
                            }
                        }
                    }
                }
            }

            // 4. Удаляем рёбра внутри компонент
            for (uint i = 0; i < n; i++) {
                uint comp_i;
                f->get_uint(i, comp_i);

                for (uint j = 0; j < n; j++) {
                    uint comp_j;
                    f->get_uint(j, comp_j);

                    // Если вершины в одной компоненте, удаляем ребро
                    if (comp_i == comp_j) {
                        S->set_uint(i, j, INF_ENCODED);
                    }
                }
            }

            // Подсчитываем количество оставшихся компонент
            const auto unique_comps = Vector::make(n, UINT);
            unique_comps->set_fill_value(Scalar::make_uint(0));
            unique_comps->fill_with(Scalar::make_uint(0));

            for (uint v = 0; v < n; v++) {
                uint comp;
                f->get_uint(v, comp);
                unique_comps->set_uint(comp, 1);
            }

            const auto scalar_count = Scalar::make_uint(0);
            exec_v_reduce(scalar_count, Scalar::make_uint(0), unique_comps, PLUS_UINT);
            nvals = scalar_count->as_uint();

            if (nvals <= 1 || !added_edges) break;
        }
    }
}
