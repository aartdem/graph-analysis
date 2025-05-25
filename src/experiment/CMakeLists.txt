add_executable(mst_benchmark main.cpp)

target_include_directories(mst_benchmark PRIVATE
        ${CMAKE_SOURCE_DIR}/src/lib
)

target_link_libraries(mst_benchmark
        algos_lib
)

set_target_properties(mst_benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_compile_definitions(mst_benchmark PRIVATE
        DATA_DIR="${CMAKE_SOURCE_DIR}/data"
)

if(USE_CUDA)
    target_compile_definitions(mst_benchmark PRIVATE
            HAVE_CUDA
            CUDA_ENABLED
    )
endif()