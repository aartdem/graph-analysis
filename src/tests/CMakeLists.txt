set(BINARY ${CMAKE_PROJECT_NAME}_test)

include(GoogleTest)

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.0
)

FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp*"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

add_executable(${BINARY} ${TEST_SOURCES})

target_link_libraries(${BINARY} algos_lib GTest::gtest_main)

# make relative path of data directory (DATA_DIR) accessed in tests
target_compile_definitions(${BINARY} PRIVATE DATA_DIR="${CMAKE_SOURCE_DIR}/data")

gtest_discover_tests(${BINARY})
