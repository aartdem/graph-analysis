file(GLOB_RECURSE LIB_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp*"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hxx*"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
    "${CMAKE_SOURCE_DIR}/deps/gunrock/third_party/mmio/mmio.c")

set(SPLA_SRC_DIR "${CMAKE_SOURCE_DIR}/deps/spla")

add_library(algos_lib STATIC ${LIB_SOURCES})

if(USE_CUDA)
    set_target_properties(algos_lib PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

set(SPLA_BUILD_TESTS OFF CACHE BOOL "Disable building of Spla unit tests" FORCE)
set(SPLA_BUILD_EXAMPLES OFF CACHE BOOL "Disable building of Spla examples" FORCE)
set(SPLA_BUILD_OPENCL OFF CACHE BOOL "Build library with opencl backend" FORCE)

add_subdirectory(${SPLA_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/spla EXCLUDE_FROM_ALL)

# Базовые пути включения, доступные всегда
target_include_directories(algos_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})

# Пути включения для CUDA/Gunrock, доступные только если CUDA включен
if(USE_CUDA)
  target_include_directories(algos_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/deps/gunrock/include
    ${CUDAToolkit_INCLUDE_DIRS} 
    ${CMAKE_SOURCE_DIR}/deps/gunrock/third_party/Moderngpu/src 
    ${CMAKE_SOURCE_DIR}/deps/gunrock/third_party/mmio)
endif()

target_compile_definitions(algos_lib PUBLIC SPLA_SUITESPARSE_ENABLED=1)

target_link_libraries(algos_lib PUBLIC spla)