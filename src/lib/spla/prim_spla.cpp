#include <spla.hpp>
#include "common/tree.hpp"
#include "prim_spla.hpp"
#include <vector>
#include <fstream>

namespace algos {
    void PrimSpla::load_graph(const std::filesystem::path &file_path) {
        std::ifstream input(file_path);
        std::string line;

        // Skip comments
        while (std::getline(input, line))
            if (line[0] != '%') break;

        int n_input, m_input, nnz_input;
        std::istringstream iss(line);
        iss >> n_input >> m_input >> nnz_input;
        if (n_input < 0) {
            throw std::runtime_error("Invalid mtx format, n < 0");
        }
        if (nnz_input < 0) {
            throw std::runtime_error("Invalid mtx format, nnz < 0");
        }
        n = n_input;
        edges = nnz_input;
        a = spla::Matrix::make(n, n, spla::INT);

        int u, v, w;
        for (int i = 0; i < nnz_input; ++i) {
            input >> u >> v >> w;
            u--;
            v--;
            if (u < 0 || v < 0 || u > n || v > n) {
                throw std::runtime_error("Invalid graph, incorrect vertex numbers");
            }
            if (w <= 0) {
                throw std::runtime_error("Invalid graph, negative edges");
            }
            if (u != v) {
                a->set_int(u, v, w);
                a->set_int(v, u, w);
            }
        }
    }

    // for debug
    void print_vector(const spla::ref_ptr<spla::Vector> &v, const std::string &name = "") {
        std::cout << "-- " << name << " --\n";
        auto sz = spla::Scalar::make_int(0);
        spla::exec_v_count_mf(sz, v);
        auto buffer_int = std::vector<int>(sz->as_int());
        auto keys_view = spla::MemView::make(buffer_int.data(), sz->as_int());
        auto values_view = spla::MemView::make(buffer_int.data(), sz->as_int());
        v->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (int *) values_view->get_buffer();
        for (int i = 0; i < sz->as_int(); i++) {
            std::cout << keys[i] << ' ' << values[i] << '\n';
        }
    }

    using clock = std::chrono::steady_clock;
    const int INF = 1e9;
//    const float EPS = 1e-6;

//    bool are_equals(float a, float b) {
//        return std::fabs(a - b) < EPS;
//    }
//
//    bool is_less(float a, float b) {
//        return std::fabs(b - a) > EPS;
//    }

    std::chrono::seconds PrimSpla::compute() {
        auto start = clock::now();
        compute_();
        auto end = clock::now();
        return std::chrono::duration_cast<std::chrono::seconds>(end - start);
    }

    Tree PrimSpla::get_result() {
        auto sparse_sz = spla::Scalar::make_uint(0);
        auto buffer_int = std::vector<int>(n);
        std::vector<int> p(n, -1);
        spla::exec_v_count_mf(sparse_sz, mst);
        auto keys_view = spla::MemView::make(buffer_int.data(), sparse_sz->as_int());
        auto values_view = spla::MemView::make(buffer_int.data(), sparse_sz->as_int());
        mst->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (int *) values_view->get_buffer();
        for (int i = 0; i < sparse_sz->as_int(); i++) {
            p[keys[i]] = values[i];
        }
        return Tree{n, p, weight};
    }

    void PrimSpla::compute_() {
        auto neg_one_int = spla::Scalar::make_int(-1);
        auto inf_int = spla::Scalar::make_int(INF);

        mst = spla::Vector::make(n, spla::INT);
        auto d = spla::Vector::make(n, spla::INT);
        auto d_new = spla::Vector::make(n, spla::INT);
        auto d_mask = spla::Vector::make(n, spla::INT);
        auto v_row = spla::Vector::make(n, spla::INT);
        std::vector<bool> used(n, false);

        mst->set_fill_value(neg_one_int);
        mst->fill_with(neg_one_int);
        d->set_fill_value(inf_int);
        d->fill_with(inf_int);
        d_new->set_fill_value(inf_int);
        d_new->fill_with(inf_int);
        d_mask->set_fill_value(inf_int);
        d_mask->fill_with(inf_int);
        v_row->set_fill_value(inf_int);
        v_row->fill_with(inf_int);

        auto buffer_int = std::vector<int>(n);
        auto sparse_sz = spla::Scalar::make_uint(0);

        weight = 0;
        int last_unused_inf = 0;
        if (n <= 1 || edges == 0) {
            return;
        }

        d->set_int(0, 0);

        while (true) {
            spla::exec_v_count_mf(sparse_sz, d);
            int sz = sparse_sz->as_int();
            auto keys_view = spla::MemView::make(buffer_int.data(), sz);
            auto values_view = spla::MemView::make(buffer_int.data(), sz);
            d->read(keys_view, values_view);
            auto keys = (int *) keys_view->get_buffer();
            auto values = (int *) values_view->get_buffer();
            int v = -1;
            int min_dist = INF;
            for (int i = 0; i < sz; i++) {
                int dist = values[i];
                if (!used[keys[i]] && dist < min_dist) {
                    min_dist = dist;
                    v = keys[i];
                }
            }
            if (v == -1) {
                while (used[last_unused_inf]) {
                    last_unused_inf++;
                }
                if (last_unused_inf == n) {
                    break;
                }
                v = last_unused_inf;
            }
            used[v] = true;
            if (min_dist != INF) {
                weight += min_dist;
            }
            d->set_int(v, 0);
            spla::exec_m_extract_row(v_row, a, v, spla::IDENTITY_INT);
            spla::exec_v_eadd(d_new, d, v_row, spla::MIN_INT);
            spla::exec_v_eadd(d_mask, d_new, d, spla::MINUS_INT);

            // update parents
            spla::exec_v_assign_masked(mst, d_mask, spla::Scalar::make_int(v), spla::SECOND_INT, spla::NQZERO_INT);
            std::swap(d, d_new);

            std::cout << weight << '\n';
        }
    }
}
