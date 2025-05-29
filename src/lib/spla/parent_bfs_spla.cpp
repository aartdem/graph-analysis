#include "parent_bfs_spla.hpp"
#include "common/tree.hpp"
#include <fstream>
#include <spla.hpp>
#include <vector>

namespace algos {
    void ParentBfsSpla::load_graph(const std::filesystem::path &file_path) {
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
        edges_count = nnz_input;
        a = spla::Matrix::make(n, n, spla::INT);
        a->set_fill_value(zero_int);

        int u, v;
        for (int i = 0; i < nnz_input; ++i) {
            input >> u >> v;
            u--;
            v--;
            if (u < 0 || v < 0 || u > n || v > n) {
                throw std::runtime_error("Invalid graph, incorrect vertex numbers");
            }
            if (u != v) {
                a->set_int(u, v, v + 1);
                a->set_int(v, u, u + 1);
            }
        }
    }

    // for debug
    void ParentBfsSpla::print_vector(const spla::ref_ptr<spla::Vector> &v, const std::string &name) {
        std::cout << "-- " << name << " --\n";
        auto sz = spla::Scalar::make_uint(0);
        spla::exec_v_count_mf(sz, v);
        auto keys_view = spla::MemView::make(buffer1.data(), sz->as_uint(), false);
        auto values_view = spla::MemView::make(buffer2.data(), sz->as_uint(), false);
        v->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (int *) values_view->get_buffer();
        for (unsigned int i = 0; i < sz->as_uint(); i++) {
            std::cout << keys[i] + 1 << " | " << values[i] << '\n';
        }
    }

    using clock = std::chrono::steady_clock;

    std::chrono::milliseconds ParentBfsSpla::compute() {
        auto start = clock::now();
        compute_();
        auto end = clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }

    void ParentBfsSpla::compute_() {
        using namespace spla;

        parent = Vector::make(n, INT);
        parent->set_fill_value(zero_int);

        auto update_parent = Vector::make(n, INT);
        update_parent->set_fill_value(zero_int);

        auto new_parent = Vector::make(n, INT);
        new_parent->set_fill_value(zero_int);

        //        auto visited = Vector::make(n, INT);
        //        visited->set_fill_value(zero_int);

        auto new_front = Vector::make(n, INT);
        new_front->set_fill_value(zero_int);

        auto prev_front = Vector::make(n, INT);
        prev_front->set_fill_value(zero_int);

        auto desc = Descriptor::make();
        desc->set_early_exit(true);
        desc->set_struct_only(true);

        for (int v = 0; v < n; v++) {
            int parent_v;
            parent->get_int(v, parent_v);
            if (parent_v == 0) {
                parent->set_int(v, v + 1);
                prev_front->set_int(v, v + 1);

                auto front_size = Scalar::make_int(1);

                while (front_size->as_int() > 0) {
                    exec_vxm_masked(new_front, parent, prev_front, a, SECOND_INT, FIRST_INT, EQZERO_INT, zero_int, desc);
                    exec_vxm_masked(update_parent, parent, prev_front, a, FIRST_INT, FIRST_INT, EQZERO_INT, zero_int, desc);
                    exec_v_eadd(new_parent, update_parent, parent, PLUS_INT);

                    exec_v_count_mf(front_size, new_front);

                    std::swap(parent, new_parent);
                    std::swap(prev_front, new_front);
                }
            }
        }
    }

    Tree ParentBfsSpla::get_result() {
        std::vector<int> p(n, -1);
        if (n <= 1 || edges_count == 0) {
            return Tree{static_cast<uint>(n), p, 0};
        }
        auto sparse_sz = spla::Scalar::make_uint(0);
        spla::exec_v_count_mf(sparse_sz, parent);
        auto keys_view = spla::MemView::make(buffer1.data(), sparse_sz->as_uint());
        auto values_view = spla::MemView::make(buffer2.data(), sparse_sz->as_uint());
        parent->read(keys_view, values_view);
        auto keys = (int *) keys_view->get_buffer();
        auto values = (int *) values_view->get_buffer();
        for (unsigned int i = 0; i < sparse_sz->as_uint(); i++) {
            int cur_v = keys[i];
            if (values[i] != keys[i] + 1) {
                int cur_p = static_cast<int>(values[i]);
                p[cur_v] = cur_p - 1;
            }
        }
        return Tree{static_cast<uint>(n), p, 0};
    }
}// namespace algos