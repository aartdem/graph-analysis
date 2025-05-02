file(GLOB_RECURSE LIB_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp*"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

add_library(algos_lib STATIC ${LIB_SOURCES})

target_include_directories(algos_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
