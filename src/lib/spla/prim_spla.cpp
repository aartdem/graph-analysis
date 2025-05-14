#include <spla.hpp>
#include "common/tree.hpp"
#include "prim_spla.hpp"
#include <vector>
#include <fstream>
#include <cmath>
#include <set>

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
        buffer1 = std::vector<unsigned int>(n);
        buffer2 = std::vector<unsigned int>(n);
        edges_count = nnz_input;
        a = spla::Matrix::make(n, n, spla::UINT);
//        a->set_format(spla::FormatMatrix::AccCsr);
        a->set_fill_value(spla::Scalar::make_uint(INF));

        int u, v;
        int w;
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
                a->set_uint(u, v, w);
                a->set_uint(v, u, w);
            }
        }
    }

    // for debug
    void PrimSpla::print_vector(const spla::ref_ptr<spla::Vector> &v, const std::string &name) {
        std::cout << "-- " << name << " --\n";
        auto sz = spla::Scalar::make_uint(0);
        spla::exec_v_count_mf(sz, v);
        auto keys_view = spla::MemView::make(buffer1.data(), sz->as_uint(), false);
        auto values_view = spla::MemView::make(buffer2.data(), sz->as_uint(), false);
        v->read(keys_view, values_view);
        auto keys = (unsigned int *) keys_view->get_buffer();
        auto values = (unsigned int *) values_view->get_buffer();
        for (unsigned int i = 0; i < sz->as_uint(); i++) {
            std::cout << keys[i] << " | " << values[i] << ' ' << values[i] / C << ' ' << values[i] % C << '\n';
        }
    }

    void PrimSpla::update(std::set<std::pair<unsigned int, unsigned int>> &s, const spla::ref_ptr<spla::Vector> &v) {
        auto sz = spla::Scalar::make_uint(0);
        spla::exec_v_count_mf(sz, v);
        auto keys_view = spla::MemView::make(buffer1.data(), sz->as_uint(), false);
        auto values_view = spla::MemView::make(buffer2.data(), sz->as_uint(), false);
        v->read(keys_view, values_view);
        auto keys = (unsigned int *) keys_view->get_buffer();
        auto values = (unsigned int *) values_view->get_buffer();
        for (unsigned int i = 0; i < sz->as_uint(); i++) {
            s.insert({values[i], keys[i]});
        }
    }

    using clock = std::chrono::steady_clock;

    std::chrono::milliseconds PrimSpla::compute() {
        auto start = clock::now();
        compute_();
        auto end = clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }

//    const float EPS = 1e-8;

    void PrimSpla::log(const std::string &t) {
        bool enabled = true;
        if (enabled) {
            auto k = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - last_time).count();
            if (k > 10) {
                std::cout << "! " << t << ": " << k << '\n';
            }
            last_time = clock::now();
        }
    }

    void PrimSpla::compute_() {
        log("");

//        visited = spla::Vector::make(n, spla::UINT);
//        zero_vec = spla::Vector::make(n, spla::UINT);
        mst = spla::Vector::make(n, spla::UINT);
//        one_vec = spla::Vector::make(n, spla::UINT);

        auto d = spla::Vector::make(n, spla::UINT);
        auto changed = spla::Vector::make(n, spla::UINT);
//        auto d_modified = spla::Vector::make(n, spla::UINT);
        auto v_row = spla::Vector::make(n, spla::UINT);
        auto min_v = spla::Scalar::make_uint(INF);

//        visited->set_fill_value(inf_uint);
//        visited->fill_with(zero_uint);
//        zero_vec->set_fill_value(zero_uint);
//        one_vec->set_fill_value(spla::Scalar::make_uint(1));
        changed->set_fill_value(zero_uint);
        mst->set_fill_value(inf_uint);
        d->set_fill_value(inf_uint);
//        d_modified->set_fill_value(inf_uint);
        v_row->set_fill_value(inf_uint);

        weight = 0;
        if (n <= 1 || edges_count == 0) {
            return;
        }

        unsigned int counter = 0;
        std::set<std::pair<unsigned int, unsigned int>> s;
        std::vector<bool> visited(n, false);

        for (int i = 0; i < n; i++) {
            log("start component prepare");
            if (!visited[i]) {
                unsigned int v = i;
                d->set_uint(v, 0);
                visited[v] = true;
                spla::exec_m_extract_row(v_row, a, v, spla::IDENTITY_UINT);
                spla::exec_v_eadd_fdb(d, v_row, changed, spla::MIN_UINT);
                spla::exec_v_assign_masked(mst, changed, spla::Scalar::make_uint(v), spla::SECOND_UINT,
                                           spla::NQZERO_UINT);

                update(s, changed);
                log("prepare");
                while (!s.empty()) {
                    if (++counter % 10000 == 0) {
                        log("10000 iterations");
                    }
                    unsigned int w = s.begin()->first;
                    v = s.begin()->second;
                    s.erase(s.begin());
                    if (visited[v]) continue;

                    weight += w;
                    log("extract");
                    d->set_uint(v, 0);
                    visited[v] = true;
                    log("set_visited");
                    spla::exec_m_extract_row(v_row, a, v, spla::IDENTITY_UINT);
                    log("exec_m_extract_row");
                    spla::exec_v_eadd_fdb(d, v_row, changed, spla::MIN_UINT);
                    log("exec_v_eadd");
                    spla::exec_v_assign_masked(mst, changed, spla::Scalar::make_uint(v), spla::SECOND_UINT,
                                               spla::NQZERO_UINT);

                    update(s, changed);
                }
            }
        }
    }

    Tree PrimSpla::get_result() {
        std::vector<int> p(n, -1);
        if (n <= 1 || edges_count == 0) {
            return Tree{n, p, 0};
        }
        auto sparse_sz = spla::Scalar::make_uint(0);
        spla::exec_v_count_mf(sparse_sz, mst);
        auto keys_view = spla::MemView::make(buffer1.data(), sparse_sz->as_int());
        auto values_view = spla::MemView::make(buffer2.data(), sparse_sz->as_int());
        mst->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (unsigned int *) values_view->get_buffer();
        for (int i = 0; i < sparse_sz->as_int(); i++) {
            int cur_v = keys[i];
            if (values[i] != INF) {
                int cur_p = static_cast<int>(values[i]);
                p[cur_v] = cur_p;
            }
        }
        return Tree{n, p, weight};
    }
}