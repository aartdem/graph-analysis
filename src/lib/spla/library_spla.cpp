#include "library_spla.hpp"
#include "spla/library.hpp"
#include <iostream>

namespace algos {
    spla::Library *library;

    void print_spla_accelerator_info() {
        library = spla::Library::get();
        std::string acc_info;
        library->get_accelerator_info(acc_info);
        std::cout << "# Accelerator: " << acc_info << std::endl;
    }
}// namespace algos
