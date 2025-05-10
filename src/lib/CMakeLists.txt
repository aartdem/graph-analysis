file(GLOB_RECURSE LIB_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp*"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

set(SPLA_SRC_DIR "${CMAKE_SOURCE_DIR}/deps/spla")

add_library(algos_lib STATIC ${LIB_SOURCES})

set(SPLA_BUILD_TESTS OFF CACHE BOOL "Disable building of Spla unit tests" FORCE)
set(SPLA_BUILD_EXAMPLES OFF CACHE BOOL "Disable building of Spla examples" FORCE)

add_subdirectory(${SPLA_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/spla EXCLUDE_FROM_ALL)

target_link_libraries(algos_lib PUBLIC spla)

target_include_directories(algos_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
