cmake_minimum_required(VERSION 3.20)
project(acn CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(ACN_WITH_GMP "Build the BigInt backend against GMP" ON)
option(ACN_BUILD_TESTS "Build tests" ON)
option(ACN_BUILD_BENCH "Build benchmarks" OFF)
option(ACN_STATS "Enable instrumentation counters" OFF)

add_library(acn INTERFACE)
add_library(acn::acn ALIAS acn)
target_include_directories(acn INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(acn INTERFACE cxx_std_20)

if(ACN_STATS)
    target_compile_definitions(acn INTERFACE ACN_STATS=1)
endif()

if(ACN_WITH_GMP)
    find_library(GMP_LIBRARY gmp REQUIRED)
    find_path(GMP_INCLUDE_DIR gmp.h REQUIRED)
    target_link_libraries(acn INTERFACE ${GMP_LIBRARY})
    target_include_directories(acn INTERFACE ${GMP_INCLUDE_DIR})
    target_compile_definitions(acn INTERFACE ACN_WITH_GMP=1)
endif()

if(ACN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ACN_BUILD_BENCH)
    add_subdirectory(bench)
endif()
