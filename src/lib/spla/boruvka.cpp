#include <spla.hpp>

// #define INF 1e9

using namespace std;
using namespace spla;

const int INF = numeric_limits<int>::max();

void boruvka(const ref_ptr<Matrix>& A) {
    int n = A->get_n_rows();

    int* parent = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++)
        parent[i] = i;

    ref_ptr<Vector> f = Vector::make(n, INT);
    for (int i = 0; i < n; ++i) {
        f->set_int(i, i);
    }

    while (m_has_nempty_values(A))
    {
        std::pair<ref_ptr<Vector>, ref_ptr<Vector>> result = comb_min_product(f, A);
        auto [min_values, min_indices] = result;

        // calculate of cedge
        int* cedge_weight = (int*)malloc(sizeof(int) * n);
        int* cedge_j = (int*)malloc(sizeof(int) * n);
        for (int p = 0; p < n; p++) {
            cedge_weight[p] = INF;
            cedge_j[p] = -1;
        }

        for (int i = 0; i < n; i++) {
            int p = parent[i];
            int cur_min = cedge_weight[p];
            int edge_weight_i;
            min_values->get_int(i, edge_weight_i);
            int edge_j_i;
            min_indices->get_int(i, edge_j_i);
        
            if (edge_weight_i < cur_min) {
                cedge_weight[p] = edge_weight_i;
                cedge_j[p] = edge_j_i;
            }
        }

        // calculate of parent
        for (int p = 0; p < n; p++) {
            if (parent[p] == p && cedge_j[p] != -1) {  // p — корень компоненты и  есть хотя бы одна компонента которая связывается с нашей
                int new_parent = find_root(parent, cedge_j[p]);
                parent[p] = new_parent;
            }
        }

        // filter edges in A
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int val;
                A->get_int(i, j, val);
                if (val != 0 && parent[i] == parent[j]) {
                    A->set_int(i, j, 0);
                }
            }
        }

        free(cedge_weight);
        free(cedge_j);
    }
}

bool m_has_nempty_values(ref_ptr<Matrix> A) {
    ref_ptr<Scalar> result = Scalar::make_int(0);
    ref_ptr<Scalar> init = Scalar::make_int(0);
    exec_m_reduce(result, init, A, LOR_INT);

    return result->as_int() > 0;
}

std::pair<ref_ptr<Vector>, ref_ptr<Vector>> comb_min_product(ref_ptr<Vector> v, ref_ptr<Matrix> A) {
    int n = v->get_n_rows();
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

    ref_ptr<Matrix> W;
    exec_mxm(W, A, F, MULT_INT, MIN_INT, zero);

    return row_min_and_argmin(W);
}

std::pair<ref_ptr<Vector>, ref_ptr<Vector>> row_min_and_argmin(ref_ptr<Matrix> A) {
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

int find_root(int* parent, int x) {
    if (parent[x] != x) {
        parent[x] = find_root(parent, parent[x]);
    }
    return parent[x];
}