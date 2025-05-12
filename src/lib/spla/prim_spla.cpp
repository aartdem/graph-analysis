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
        a->set_fill_value(spla::Scalar::make_float(INF));

        int u, v;
        float w;
        for (int i = 0; i < nnz_input; ++i) {
            if (i % 1000000 == 0) {
                std::cout << "readed: " << i << '\n';
            }
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

    std::chrono::milliseconds PrimSpla::compute() {
        auto start = clock::now();
        compute_();
        auto end = clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }

//    void PrimSpla::copy_vector(const spla::ref_ptr<spla::Vector> &from, const spla::ref_ptr<spla::Vector> &to) {
//        auto sparse_sz = spla::Scalar::make_uint(0);
//        spla::exec_v_count_mf(sparse_sz, from);
//        int sz = sparse_sz->as_int();
//        auto keys_view = spla::MemView::make(buffer_int.data(), sz);
//        auto values_view = spla::MemView::make(buffer_float.data(), sz);
//        from->read(keys_view, values_view);
//        to->build(keys_view, values_view);
//    }

    const float EPS = 1e-8;

    std::pair<float, int> PrimSpla::get_min_with_arg(const spla::ref_ptr<spla::Vector> &vec) const {
        int v = 0;
        float min_dist;
        vec->get_float(v, min_dist);

        for (int i = 1; i < n; i++) {
            float dist;
            vec->get_float(i, dist);
            if (dist < min_dist) {
                min_dist = dist;
                v = i;
            }
        }
        return {min_dist, v};
    }

    void PrimSpla::log(const std::string &t) {
        std::cout << "! " << t << ": "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - last_time).count()
                  << '\n';
        last_time = clock::now();
    }

    void PrimSpla::compute_() {
        log("");

        visited = spla::Vector::make(n, spla::INT);
        visited->set_fill_value(spla::Scalar::make_int(0));
        zero_vec = spla::Vector::make(n, spla::FLOAT);
        mst = spla::Vector::make(n, spla::FLOAT);

        auto d = spla::Vector::make(n, spla::FLOAT);
        auto changed = spla::Vector::make(n, spla::FLOAT);
        auto d_modified = spla::Vector::make(n, spla::FLOAT);
        auto v_row = spla::Vector::make(n, spla::FLOAT);

        zero_vec->set_fill_value(zero_float);
        mst->set_fill_value(inf_float);
        d->set_fill_value(inf_float);
        d_modified->set_fill_value(inf_float);
        v_row->set_fill_value(inf_float);

        changed->set_fill_value(zero_float);

        weight = 0;
        if (n <= 1 || edges_count == 0) {
            return;
        }

        int counter = 0;

        for (int i = 0; i < n; i++) {
            int v;
            float min_dist;
            d->get_float(i, min_dist);
            if (std::fabs(min_dist - INF) < EPS) {
                v = i;
                d->set_float(v, 0);

                spla::exec_m_extract_row(v_row, a, v, spla::IDENTITY_FLOAT);
                spla::exec_v_eadd_fdb(d, v_row, changed, spla::MIN_FLOAT);
                auto v_float = spla::Scalar::make_float(static_cast<float>(v));
                spla::exec_v_assign_masked(mst, changed, v_float, spla::SECOND_FLOAT, spla::NQZERO_FLOAT);

                while (true) {
//                    if (counter++ % 10000 == 0) {
//                        log("10000 iterations");
//                    }

                    exec_v_eadd(d_modified, d, zero_vec, spla::PLUS_FLOAT);
                    log("exec_v_eadd");
                    spla::exec_v_assign_masked(d_modified, d, inf_float, spla::SECOND_FLOAT, spla::EQZERO_FLOAT);
                    log("exec_v_assign_masked");
                    std::tie(min_dist, v) = get_min_with_arg(d_modified);
                    log("get_min_with_arg");

                    if (std::fabs(min_dist - INF) < EPS) {
                        break;
                    }

                    weight += min_dist;
                    d->set_float(v, 0);
                    log("set_visited");

                    spla::exec_m_extract_row(v_row, a, v, spla::IDENTITY_FLOAT);
                    log("exec_m_extract_row");
                    spla::exec_v_eadd_fdb(d, v_row, changed, spla::MIN_FLOAT);
                    log("exec_v_eadd");

                    v_float = spla::Scalar::make_float(static_cast<float>(v));
                    spla::exec_v_assign_masked(mst, changed, v_float, spla::SECOND_FLOAT, spla::NQZERO_FLOAT);
                    log("exec_v_assign_masked");
                }
            }
        }
    }

    Tree PrimSpla::get_result() {
        std::vector<int> p(n, -1);
        if (n <= 1 || edges_count == 0) {
            return Tree{n, p, weight};
        }
        auto sparse_sz = spla::Scalar::make_uint(0);
        spla::exec_v_count_mf(sparse_sz, mst);
        auto keys_view = spla::MemView::make(buffer_int.data(), sparse_sz->as_int());
        auto values_view = spla::MemView::make(buffer_float.data(), sparse_sz->as_int());
        mst->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (float *) values_view->get_buffer();
        for (int i = 0; i < sparse_sz->as_int(); i++) {
            int cur_v = keys[i];
            if (std::fabs(values[i] - INF) > EPS) {
                int cur_p = static_cast<int>(std::round(values[i]));
                p[cur_v] = cur_p;
            }
        }
        return Tree{n, p, weight};
    }
}