add_executable(run_experiment main.cpp)

target_link_libraries(run_experiment algos_lib)

set_target_properties(run_experiment PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# make relative path of data directory (DATA_DIR) accessed in experiment
target_compile_definitions(run_experiment PRIVATE DATA_DIR="${CMAKE_SOURCE_DIR}/data")
 