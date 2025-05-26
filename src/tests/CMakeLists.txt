if(USE_CUDA)
    project(algos_lib_tests LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD           20)
    set(CMAKE_CUDA_STANDARD_REQUIRED  ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
else()
    project(algos_lib_tests LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD            20)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.12.0
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

enable_testing()

if(USE_CUDA)
    file(GLOB TEST_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
    )
else()
    file(GLOB TEST_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    )
endif()

set(BINARY ${CMAKE_PROJECT_NAME}_test)
add_executable(${BINARY} ${TEST_SOURCES})
set_target_properties(${BINARY} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

if(USE_CUDA)
    get_target_property(ALGOS_ARCHS algos_lib CUDA_ARCHITECTURES)
    if(ALGOS_ARCHS)
        set_target_properties(${BINARY} PROPERTIES
                CUDA_ARCHITECTURES        "${ALGOS_ARCHS}"
        )
    endif()
endif()

target_link_libraries(${BINARY}
        PRIVATE algos_lib
        PRIVATE GTest::gtest_main
)

if(USE_CUDA)
    target_link_libraries(${BINARY}
            PRIVATE CUDA::cudart
    )
endif()

target_compile_definitions(${BINARY}
        PRIVATE DATA_DIR="${CMAKE_SOURCE_DIR}/test_data"
)

add_test(NAME MstAlgorithmTest
        COMMAND ${BINARY})

if(USE_CUDA)
    set_tests_properties(MstAlgorithmTest PROPERTIES
            ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
endif()