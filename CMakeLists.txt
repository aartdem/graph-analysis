cmake_minimum_required(VERSION 3.20)
project(Graph-Analysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(spla)

add_executable(run_experiment main.cpp)
target_link_libraries(run_experiment PRIVATE spla)
