#pragma once

// A small built-in benchmark harness (PLAN.md §8.1: "Google Benchmark if
// available, otherwise a small built-in harness (must not become a hard
// dependency)"). bench/CMakeLists.txt tries find_package(benchmark) first
// and only falls back to this when it isn't found -- this repo's dev
// environment doesn't have it installed, which is exactly the case this
// harness exists for.
//
// Auto-calibrates: doubles the iteration count until a run's wall time
// clears `target_ms`, then reports that run's mean ns/op. Deliberately not
// statistically rigorous (no warmup-then-measure separation, no repeated
// samples/variance reporting the way Google Benchmark does) -- it exists
// to give PLAN.md §8's benchmarks a number to report, not to replace a
// real benchmarking library if one becomes available later.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

namespace acn::bench {

// Prevents the optimizer from eliding a computation whose result is never
// otherwise observed. The "m" (memory) constraint works for arbitrary-size
// aggregates (unlike "g"/"r", which require the value to fit a register),
// which matters here since Acn<N,Int> for larger N won't fit in one.
template <class T>
inline void do_not_optimize(const T& value) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "m"(value) : "memory");
#else
    // PLAN.md's toolchain matrix is GCC 15 / Clang 18+ only, so this path
    // is never taken in practice; (void) just keeps a non-GNU build
    // compiling rather than optimizing the benchmarked work away silently.
    (void)value;
#endif
}

struct Result {
    std::string name;
    double ns_per_op;
    std::uint64_t iterations;
};

// Runs fn(iterations) with a doubling iteration count until wall time
// clears target_ms (or an iteration cap, so a degenerate near-zero-cost
// fn can't loop forever), then reports ns/op for that final run. fn must
// call do_not_optimize on whatever it computes internally.
template <class Fn>
Result run(std::string name, Fn&& fn, double target_ms = 200.0) {
    using clock = std::chrono::steady_clock;
    std::uint64_t iterations = 1;
    for (;;) {
        auto start = clock::now();
        fn(iterations);
        double elapsed_ms = std::chrono::duration<double, std::milli>(clock::now() - start).count();
        if (elapsed_ms >= target_ms || iterations >= (std::uint64_t{1} << 34)) {
            double ns_per_op = elapsed_ms * 1.0e6 / static_cast<double>(iterations);
            return Result{std::move(name), ns_per_op, iterations};
        }
        iterations *= 2;
    }
}

inline void print_result(const Result& r) {
    std::printf("%-55s %12.2f ns/op  (%llu iters)\n", r.name.c_str(), r.ns_per_op,
                static_cast<unsigned long long>(r.iterations));
}

}  // namespace acn::bench
