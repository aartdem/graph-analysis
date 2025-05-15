#include "library_spla.hpp"
#include "spla/library.hpp"
#include <iostream>

namespace algos {
    spla::Library *library;

    void initialize_spla() {
        library = spla::Library::get();
        std::string acc_info;
        library->get_accelerator_info(acc_info);
        std::cout << "# Accelerator: " << acc_info << std::endl;
    }

    void finalize_spla() {
        library->finalize();
        library = nullptr;
    }
}// namespace algos
