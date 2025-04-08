#include <iostream>

#include "spla/include/spla.hpp"

int main()
{
    int N = 100;
    spla::ref_ptr<spla::Vector> v_cpu = spla::Vector::make(N, spla::INT);
    std::cout << "Hello world!\n";
    // spla::ref_ptr<spla::Vector> v_acc = spla::Vector::make(N, spla::INT);
    // spla::ref_ptr<spla::Matrix> A = spla::Matrix::make(N, N, spla::INT);
    // spla::ref_ptr<spla::Descriptor> desc = spla::Descriptor::make();
    return 0;
}