cmake_minimum_required(VERSION 3.24)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "RTX 50-series")

project(algos_lib LANGUAGES CXX CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -DSM_TARGET=70")

set(CMAKE_CXX_STANDARD            20)
set(CMAKE_CXX_STANDARD_REQUIRED   ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

include(FetchContent)
FetchContent_Declare(
  cmake_modules
  GIT_REPOSITORY https://github.com/rpavlik/cmake-modules.git 
  GIT_TAG        main
)
FetchContent_GetProperties(cmake_modules)
if(NOT cmake_modules_POPULATED)
  FetchContent_MakeAvailable(cmake_modules)
endif()

list(APPEND CMAKE_MODULE_PATH "${cmake_modules_SOURCE_DIR}")
FetchContent_Declare(
  gunrock
  GIT_REPOSITORY https://github.com/gunrock/gunrock.git
  GIT_TAG        main          
)
#set(ESSENTIALS_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(ESSENTIALS_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(ESSENTIALS_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(GUNROCK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "")

add_subdirectory(deps/gunrock EXCLUDE_FROM_ALL)
add_subdirectory(src/lib)
add_subdirectory(src/tests)
add_subdirectory(src/experiment)
