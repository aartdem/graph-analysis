cmake_minimum_required(VERSION 3.20)

option(USE_CUDA "Enable CUDA support" ON)

if(USE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "RTX 50-series")
    add_compile_definitions(HAVE_CUDA)
    add_compile_definitions(CUDA_ENABLED)
    message(STATUS "CUDA найден. Поддержка алгоритмов Gunrock включена.")
    set(LANGUAGES_LIST CXX CUDA C)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -DSM_TARGET=70")
  else()
    message(STATUS "CUDA не найден. Алгоритмы Gunrock будут отключены.")
    set(USE_CUDA OFF)
    set(LANGUAGES_LIST CXX C)
  endif()
else()
  message(STATUS "Поддержка CUDA отключена. Алгоритмы Gunrock будут недоступны.")
  set(LANGUAGES_LIST CXX C)
endif()

project(algos_lib LANGUAGES ${LANGUAGES_LIST})

set(CMAKE_CXX_STANDARD            20)
set(CMAKE_CXX_STANDARD_REQUIRED   ON)

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

if(USE_CUDA)
  FetchContent_Declare(
    gunrock
    GIT_REPOSITORY https://github.com/gunrock/gunrock.git
    GIT_TAG        main          
  )
  set(ESSENTIALS_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
  set(ESSENTIALS_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
  set(ESSENTIALS_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(GUNROCK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "")
  
  add_subdirectory(deps/gunrock EXCLUDE_FROM_ALL)
endif()

add_subdirectory(src/lib)
add_subdirectory(src/tests)
add_subdirectory(src/experiment)
