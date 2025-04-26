add_executable(run_experiment main.cpp)

target_link_libraries(run_experiment algos_lib)

set_target_properties(run_experiment PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
