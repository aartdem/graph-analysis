cmake_minimum_required(VERSION 3.20)

option(USE_CUDA "Enable CUDA support" ON)

if(APPLE)
  set(LIB_EXT "dylib")
elseif(UNIX AND NOT APPLE)
  set(LIB_EXT "so")
elseif(WIN32)
  set(LIB_EXT "dll")
endif()

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

# Создаем цель для GraphBLAS
add_library(graphblas INTERFACE)
target_include_directories(graphblas INTERFACE ${GRAPHBLAS_INCLUDE_DIR})
target_link_directories(graphblas INTERFACE ${GRAPHBLAS_LIB_DIR})

# Для прямой линковки с GraphBLAS - создаем импортированную библиотеку
add_library(GraphBLAS SHARED IMPORTED)
set_target_properties(GraphBLAS PROPERTIES
  IMPORTED_LOCATION "${GRAPHBLAS_LIB_DIR}/libgraphblas.${LIB_EXT}"
  INTERFACE_INCLUDE_DIRECTORIES "${GRAPHBLAS_INCLUDE_DIR}")

# LAGraph configuration
set(LAGRAPH_SRC_DIR "${CMAKE_SOURCE_DIR}/deps/LAGraph")
set(LAGRAPH_INCLUDE_DIR "${LAGRAPH_SRC_DIR}/include")
set(LAGRAPH_EXPERIMENTAL_DIR "${LAGRAPH_SRC_DIR}/experimental")
set(LAGRAPH_LIB_DIR "${CMAKE_SOURCE_DIR}/deps/LAGraph/build/src")

# Create a custom LAGraph library target that links to the pre-built libraries
add_library(LAGraph SHARED IMPORTED)
set_target_properties(LAGraph PROPERTIES 
  IMPORTED_LOCATION "${LAGRAPH_LIB_DIR}/liblagraph.${LIB_EXT}"
  INTERFACE_INCLUDE_DIRECTORIES "${LAGRAPH_INCLUDE_DIR}")

# Create an interface target for LAGraph headers and functionality
add_library(lagraph_headers INTERFACE)
target_include_directories(lagraph_headers INTERFACE 
  ${LAGRAPH_INCLUDE_DIR}
  ${LAGRAPH_EXPERIMENTAL_DIR})
target_link_libraries(lagraph_headers INTERFACE graphblas LAGraph)

# Create a directory for MSF implementation and required headers
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/LAGraph_msf)

# Copy the required files
file(COPY ${LAGRAPH_EXPERIMENTAL_DIR}/algorithm/LAGraph_msf.c
     DESTINATION ${CMAKE_BINARY_DIR}/LAGraph_msf)
file(COPY ${LAGRAPH_SRC_DIR}/src/utility/LG_internal.h
     DESTINATION ${CMAKE_BINARY_DIR}/LAGraph_msf)

# Create our own implementation of the MSF function to use in our code
add_library(lagraph_msf STATIC ${CMAKE_BINARY_DIR}/LAGraph_msf/LAGraph_msf.c)
target_include_directories(lagraph_msf PRIVATE 
  ${CMAKE_BINARY_DIR}/LAGraph_msf
  ${LAGRAPH_INCLUDE_DIR}
  ${LAGRAPH_EXPERIMENTAL_DIR}
  ${GRAPHBLAS_INCLUDE_DIR})
target_compile_definitions(lagraph_msf PRIVATE LAGRAPH_SUITESPARSE=1)
# Прямая ссылка на GraphBLAS для lagraph_msf
target_link_libraries(lagraph_msf PUBLIC LAGraph GraphBLAS)

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
