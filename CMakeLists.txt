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

# GraphBLAS configuration
set(GRAPHBLAS_SRC_DIR "${CMAKE_SOURCE_DIR}/deps/GraphBLAS")
set(GRAPHBLAS_INCLUDE_DIR "${GRAPHBLAS_SRC_DIR}/Include")
set(GRAPHBLAS_LIB_DIR "${CMAKE_SOURCE_DIR}/deps/GraphBLAS/build")

add_library(graphblas INTERFACE)
target_include_directories(graphblas INTERFACE ${GRAPHBLAS_INCLUDE_DIR})
target_link_directories(graphblas INTERFACE ${GRAPHBLAS_LIB_DIR})

# LAGraph configuration
set(LAGRAPH_SRC_DIR "${CMAKE_SOURCE_DIR}/deps/LAGraph")
set(LAGRAPH_INCLUDE_DIR "${LAGRAPH_SRC_DIR}/include")
set(LAGRAPH_EXPERIMENTAL_DIR "${LAGRAPH_SRC_DIR}/experimental")
set(LAGRAPH_LIB_DIR "${CMAKE_SOURCE_DIR}/deps/LAGraph/build/src")

# Create a custom LAGraph library target that links to the pre-built libraries
add_library(LAGraph SHARED IMPORTED)
set_target_properties(LAGraph PROPERTIES 
  IMPORTED_LOCATION "${LAGRAPH_LIB_DIR}/liblagraph.dylib"
  INTERFACE_INCLUDE_DIRECTORIES "${LAGRAPH_INCLUDE_DIR}")

# Create an interface target for LAGraph headers and functionality
add_library(lagraph_headers INTERFACE)
target_include_directories(lagraph_headers INTERFACE 
  ${LAGRAPH_INCLUDE_DIR}
  ${LAGRAPH_EXPERIMENTAL_DIR})
target_link_libraries(lagraph_headers INTERFACE graphblas LAGraph)

# Copy the MSF implementation to make it available for linking
file(COPY ${LAGRAPH_EXPERIMENTAL_DIR}/algorithm/LAGraph_msf.c
     DESTINATION ${CMAKE_BINARY_DIR}/LAGraph_msf)

# Create our own implementation of the MSF function to use in our code
add_library(lagraph_msf STATIC ${CMAKE_BINARY_DIR}/LAGraph_msf/LAGraph_msf.c)
target_include_directories(lagraph_msf PRIVATE 
  ${LAGRAPH_INCLUDE_DIR}
  ${LAGRAPH_EXPERIMENTAL_DIR}
  ${GRAPHBLAS_INCLUDE_DIR})
target_link_libraries(lagraph_msf PUBLIC LAGraph graphblas)

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
