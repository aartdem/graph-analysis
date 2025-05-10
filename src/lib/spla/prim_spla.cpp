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
        buffer_int1 = std::vector<int>(n);
        buffer_int2 = std::vector<int>(n);
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
    void PrimSpla::print_vector(const spla::ref_ptr<spla::Vector> &v, const std::string &name = "") {
        std::cout << "-- " << name << " --\n";
        auto sz = spla::Scalar::make_int(0);
        spla::exec_v_count_mf(sz, v);
        auto keys_view = spla::MemView::make(buffer_int1.data(), sz->as_int());
        auto values_view = spla::MemView::make(buffer_int2.data(), sz->as_int());
        v->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (int *) values_view->get_buffer();
        for (int i = 0; i < sz->as_int(); i++) {
            std::cout << keys[i] << ' ' << values[i] << '\n';
        }
    }

    using clock = std::chrono::steady_clock;
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

    void PrimSpla::copy_vector(const spla::ref_ptr<spla::Vector> &from, const spla::ref_ptr<spla::Vector> &to) {
        auto sparse_sz = spla::Scalar::make_uint(0);
        spla::exec_v_count_mf(sparse_sz, from);
        int sz = sparse_sz->as_int();
        auto keys_view = spla::MemView::make(buffer_int1.data(), sz);
        auto values_view = spla::MemView::make(buffer_int2.data(), sz);
        from->read(keys_view, values_view);
        to->build(keys_view, values_view);
    }

    std::pair<int, int> PrimSpla::get_min_with_arg(const spla::ref_ptr<spla::Vector> &vec) {
        auto sparse_sz = spla::Scalar::make_uint(0);
        spla::exec_v_count_mf(sparse_sz, vec);
        int sz = sparse_sz->as_int();
        auto keys_view = spla::MemView::make(buffer_int1.data(), sz);
        auto values_view = spla::MemView::make(buffer_int2.data(), sz);
        vec->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (int *) values_view->get_buffer();
        int v = 0;
        int min_dist = INF;
        vec->get_int(v, min_dist);
        for (int i = 0; i < sz; i++) {
            int dist = values[i];
            if (dist < min_dist) {
                min_dist = dist;
                v = keys[i];
            }
        }
        return {v, min_dist};
    }

    void PrimSpla::compute_() {
        auto neg_one_int = spla::Scalar::make_int(-1);
        auto inf_int = spla::Scalar::make_int(INF);
        auto zero_int = spla::Scalar::make_int(0);

        mst = spla::Vector::make(n, spla::INT);
        auto d = spla::Vector::make(n, spla::INT);
        auto d_new = spla::Vector::make(n, spla::INT);
        auto d_mask = spla::Vector::make(n, spla::INT);
        auto d_modified = spla::Vector::make(n, spla::INT);
        auto v_row = spla::Vector::make(n, spla::INT);

        mst->set_fill_value(neg_one_int);
        mst->fill_with(neg_one_int);
        d->set_fill_value(inf_int);
        d->fill_with(inf_int);
        d_new->set_fill_value(inf_int);
        d_new->fill_with(inf_int);
        d_mask->set_fill_value(inf_int);
        d_mask->fill_with(inf_int);
        d_modified->set_fill_value(inf_int);
        d_modified->fill_with(inf_int);
        v_row->set_fill_value(inf_int);
        v_row->fill_with(inf_int);

        weight = 0;
        if (n <= 1 || edges == 0) {
            return;
        }
        auto current_d_max = spla::Scalar::make_int(INF);
        while (current_d_max->as_int() > 0) {
            copy_vector(d, d_modified);
            spla::exec_v_assign_masked(d_modified, d, inf_int, spla::SECOND_INT, spla::EQZERO_INT);
            auto [v, min_dist] = get_min_with_arg(d_modified);
            if (min_dist < INF) {
                weight += min_dist;
            }
            d->set_int(v, 0);
            spla::exec_m_extract_row(v_row, a, v, spla::IDENTITY_INT);
            spla::exec_v_eadd(d_new, d, v_row, spla::MIN_INT);
            spla::exec_v_eadd(d_mask, d_new, d, spla::MINUS_INT);

            spla::exec_v_assign_masked(mst, d_mask, spla::Scalar::make_int(v), spla::SECOND_INT, spla::NQZERO_INT);
            std::swap(d, d_new);
            spla::exec_v_reduce(current_d_max, zero_int, d, spla::MAX_INT);
        }
    }

    Tree PrimSpla::get_result() {
        auto sparse_sz = spla::Scalar::make_uint(0);
        std::vector<int> p(n, -1);
        spla::exec_v_count_mf(sparse_sz, mst);
        auto keys_view = spla::MemView::make(buffer_int1.data(), sparse_sz->as_int());
        auto values_view = spla::MemView::make(buffer_int2.data(), sparse_sz->as_int());
        mst->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (int *) values_view->get_buffer();
        for (int i = 0; i < sparse_sz->as_int(); i++) {
            p[keys[i]] = values[i];
        }
        return Tree{n, p, weight};
    }
}
