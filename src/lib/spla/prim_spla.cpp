#include <spla.hpp>
#include "common/tree.hpp"
#include "prim_spla.hpp"
#include <vector>
#include <fstream>
#include <cmath>

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
        buffer_int = std::vector<int>(n);
        buffer_float = std::vector<float>(n);
        edges_count = nnz_input;
        a = spla::Matrix::make(n, n, spla::FLOAT);
        a->set_fill_value(spla::Scalar::make_float(PROCESSED));

        int u, v;
        float w;
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
                a->set_float(u, v, w);
                a->set_float(v, u, w);
            }
        }
    }

    // for debug
    void PrimSpla::print_vector(const spla::ref_ptr<spla::Vector> &v, const std::string &name = "") {
        std::cout << "-- " << name << " --\n";
        auto sz = spla::Scalar::make_int(0);
        spla::exec_v_count_mf(sz, v);
        auto keys_view = spla::MemView::make(buffer_int.data(), sz->as_int());
        auto values_view = spla::MemView::make(buffer_float.data(), sz->as_int());
        v->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (float *) values_view->get_buffer();
        for (int i = 0; i < sz->as_int(); i++) {
            std::cout << keys[i] << ' ' << values[i] << '\n';
        }
    }

    using clock = std::chrono::steady_clock;

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
        auto keys_view = spla::MemView::make(buffer_int.data(), sz);
        auto values_view = spla::MemView::make(buffer_float.data(), sz);
        from->read(keys_view, values_view);
        to->build(keys_view, values_view);
    }

    std::pair<float, int> PrimSpla::get_min_with_arg(const spla::ref_ptr<spla::Vector> &vec) {
        auto sparse_sz = spla::Scalar::make_uint(0);
        spla::exec_v_count_mf(sparse_sz, vec);
        int sz = sparse_sz->as_int();
        auto keys_view = spla::MemView::make(buffer_int.data(), sz);
        auto values_view = spla::MemView::make(buffer_float.data(), sz);
        vec->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (float *) values_view->get_buffer();
        int v = 0;
        float min_dist;
        vec->get_float(v, min_dist);
        for (int i = 0; i < sz; i++) {
            float dist = values[i];
            if (dist < min_dist) {
                min_dist = dist;
                v = keys[i];
            }
        }
        return {min_dist, v};
    }

    void PrimSpla::compute_() {
        auto neg_one_float = spla::Scalar::make_float(-1);
        auto not_processed = spla::Scalar::make_float(NOT_PROCESSED);
        auto processed = spla::Scalar::make_float(PROCESSED);
        auto zero_float = spla::Scalar::make_float(0);

        mst = spla::Vector::make(n, spla::FLOAT);
        auto d = spla::Vector::make(n, spla::FLOAT);
        auto d_new = spla::Vector::make(n, spla::FLOAT);
        auto d_mask = spla::Vector::make(n, spla::FLOAT);
        auto d_modified = spla::Vector::make(n, spla::FLOAT);
        auto v_row = spla::Vector::make(n, spla::FLOAT);

        mst->set_fill_value(neg_one_float);
        mst->fill_with(neg_one_float);
        d->set_fill_value(processed);
        d->fill_with(not_processed);
        d_new->set_fill_value(processed);
        d_mask->set_fill_value(processed);
        d_modified->set_fill_value(processed);
        v_row->set_fill_value(processed);

        weight = 0;
        if (n <= 1 || edges_count == 0) {
            return;
        }
        for (int i = 0; i < n; i++) {
            copy_vector(d, d_modified);
            spla::exec_v_assign_masked(d_modified, d, processed, spla::SECOND_FLOAT, spla::EQZERO_FLOAT);
            auto [min_dist, v] = get_min_with_arg(d_modified);
            if (min_dist < NOT_PROCESSED) {
                weight += min_dist;
            }
            d->set_float(v, 0);

            spla::exec_m_extract_row(v_row, a, v, spla::IDENTITY_FLOAT);
            spla::exec_v_eadd(d_new, d, v_row, spla::MIN_FLOAT);
            spla::exec_v_eadd(d_mask, d_new, d, spla::MINUS_FLOAT);

            spla::exec_v_assign_masked(mst, d_mask, spla::Scalar::make_float(static_cast<float>(v)), spla::SECOND_FLOAT,
                                       spla::NQZERO_FLOAT);
            std::swap(d, d_new);
        }
    }

    Tree PrimSpla::get_result() {
        auto sparse_sz = spla::Scalar::make_uint(0);
        std::vector<int> p(n, -1);
        spla::exec_v_count_mf(sparse_sz, mst);
        auto keys_view = spla::MemView::make(buffer_int.data(), sparse_sz->as_int());
        auto values_view = spla::MemView::make(buffer_float.data(), sparse_sz->as_int());
        mst->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (float *) values_view->get_buffer();
        for (int i = 0; i < sparse_sz->as_int(); i++) {
            p[keys[i]] = static_cast<int>(std::round(values[i]));
        }
        return Tree{n, p, weight};
    }
}
