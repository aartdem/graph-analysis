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
set(SPLA_BUILD_OPENCL ON CACHE BOOL "Build library with opencl backend" FORCE)

add_subdirectory(${SPLA_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/spla EXCLUDE_FROM_ALL)
target_link_libraries(algos_lib PUBLIC spla)

# Link against LAGraph and GraphBLAS
target_link_libraries(algos_lib PUBLIC lagraph_headers graphblas)
# Link against the LAGraph shared library and our MSF implementation
target_link_libraries(algos_lib PUBLIC LAGraph lagraph_msf)

target_include_directories(algos_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(algos_lib PUBLIC 
  ${CMAKE_SOURCE_DIR}/deps/LAGraph/include
  ${CMAKE_SOURCE_DIR}/deps/LAGraph/experimental
  ${CMAKE_SOURCE_DIR}/deps/GraphBLAS/Include)

if(USE_CUDA)
  target_include_directories(algos_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/deps/gunrock/include
    ${CUDAToolkit_INCLUDE_DIRS} 
    ${CMAKE_SOURCE_DIR}/deps/gunrock/third_party/Moderngpu/src 
    ${CMAKE_SOURCE_DIR}/deps/gunrock/third_party/mmio)
endif()