# PLAN.md §8.1: "Google Benchmark if available, otherwise a small
# built-in harness (must not become a hard dependency)". This repo's dev
# environment doesn't have Google Benchmark installed, so the built-in
# harness (bench/support/micro_bench.hpp) is what's exercised in practice;
# find_package here is best-effort so a machine that *does* have it isn't
# forced onto the fallback.
find_package(benchmark QUIET)
if(benchmark_FOUND)
    message(STATUS "bench: found Google Benchmark (not yet wired up as an alternate harness)")
endif()

function(acn_add_bench name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE acn::acn)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -O2)
endfunction()

acn_add_bench(bench_arith)
acn_add_bench(bench_hash)
acn_add_bench(bench_matmul)

# bench_circuit uses Acn<4, BigInt> throughout.
if(ACN_WITH_GMP)
    acn_add_bench(bench_circuit)
endif()
