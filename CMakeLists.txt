cmake_minimum_required(VERSION 3.20)
project(Graph_analysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_subdirectory(src/lib)
add_subdirectory(src/tests)
add_subdirectory(src/experiment)
