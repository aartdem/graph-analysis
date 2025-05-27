function(add_benchmark target src)
    add_executable(${target} ${src} bench_commons.h)

    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src/lib)

    target_link_libraries(${target} PRIVATE algos_lib)

    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

    target_compile_definitions(${target} PRIVATE DATA_DIR="${CMAKE_SOURCE_DIR}/data")
endfunction()

add_benchmark(mst_benchmark mst_benchmark.cpp)
add_benchmark(bfs_benchmark bfs_benchmark.cpp)

# only for mst
if (USE_CUDA)
    target_compile_definitions(mst_benchmark PRIVATE CUDA_ENABLED)
endif ()
