project(algos_lib_tests LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD            20)
set(CMAKE_CUDA_STANDARD           20)
set(CMAKE_CUDA_STANDARD_REQUIRED  ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.0
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

enable_testing()
file(GLOB TEST_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

set(BINARY ${CMAKE_PROJECT_NAME}_test)
add_executable(${BINARY} ${TEST_SOURCES})

get_target_property(ALGOS_ARCHS algos_lib CUDA_ARCHITECTURES)
if(ALGOS_ARCHS)
  set_target_properties(${BINARY} PROPERTIES
    CUDA_ARCHITECTURES        "${ALGOS_ARCHS}"
  )
endif()

target_link_libraries(${BINARY}
  PRIVATE algos_lib
  PRIVATE GTest::gtest_main
  PRIVATE CUDA::cudart
)

target_compile_definitions(${BINARY}
  PRIVATE DATA_DIR="${CMAKE_SOURCE_DIR}/data"
)

gtest_discover_tests(${BINARY}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/data
)
