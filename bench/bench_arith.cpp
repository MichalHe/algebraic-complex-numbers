// PLAN.md §8.1 bullet 1: add, mul, fma, canonicalize, ==, hash for
// Acn<4|8|16|32, Checked<i64>>, Acn<4, BigInt>, AcnDyn at matching n.
//
// Each benchmarked operation is run against a fixed pair of pre-generated
// random operands (never mutated across iterations, except fma_into's
// accumulator, which is reset to a fixed start value on every iteration
// rather than left to accumulate -- see the comment at bench_fma) so that
// per-iteration cost stays representative instead of drifting as
// coefficient magnitude grows.

#include <array>
#include <cstdio>
#include <random>

#include "acn/acn.hpp"
#include "bench/support/micro_bench.hpp"

using namespace acn::bench;

namespace {

template <std::size_t N, class Int>
acn::Acn<N, Int> random_value(std::mt19937_64& rng, acn::i64 bound) {
    std::uniform_int_distribution<acn::i64> dist(-bound, bound);
    std::array<acn::i64, N> coeffs{};
    for (auto& v : coeffs)
        v = dist(rng);
    return acn::Acn<N, Int>::from_ints(coeffs, 0);
}

template <std::size_t N, class Int>
void bench_precision(const char* backend_name) {
    std::mt19937_64 rng(1000 + N);
    auto a = random_value<N, Int>(rng, 5);
    auto b = random_value<N, Int>(rng, 5);
    auto start_acc = acn::Acn<N, Int>::zero();

    char name[128];

    std::snprintf(name, sizeof(name), "add<%zu,%s>", N, backend_name);
    print_result(run(name, [&](std::uint64_t iters) {
        for (std::uint64_t i = 0; i < iters; ++i) {
            auto r = a + b;
            do_not_optimize(r);
        }
    }));

    std::snprintf(name, sizeof(name), "mul<%zu,%s>", N, backend_name);
    print_result(run(name, [&](std::uint64_t iters) {
        for (std::uint64_t i = 0; i < iters; ++i) {
            auto r = a * b;
            do_not_optimize(r);
        }
    }));

    std::snprintf(name, sizeof(name), "fma<%zu,%s>", N, backend_name);
    print_result(run(name, [&](std::uint64_t iters) {
        for (std::uint64_t i = 0; i < iters; ++i) {
            auto acc = start_acc;
            fma_into(acc, a, b);
            do_not_optimize(acc);
        }
    }));

    std::snprintf(name, sizeof(name), "canonicalize<%zu,%s> (via scale(0))", N, backend_name);
    print_result(run(name, [&](std::uint64_t iters) {
        for (std::uint64_t i = 0; i < iters; ++i) {
            auto r = a.scale(0);
            do_not_optimize(r);
        }
    }));

    std::snprintf(name, sizeof(name), "operator==<%zu,%s>", N, backend_name);
    print_result(run(name, [&](std::uint64_t iters) {
        bool acc_eq = false;
        for (std::uint64_t i = 0; i < iters; ++i) {
            acc_eq = (a == b);
            do_not_optimize(acc_eq);
        }
    }));

    std::snprintf(name, sizeof(name), "hash_value<%zu,%s>", N, backend_name);
    print_result(run(name, [&](std::uint64_t iters) {
        acn::u64 acc_hash = 0;
        for (std::uint64_t i = 0; i < iters; ++i) {
            acc_hash = acn::hash_value(a);
            do_not_optimize(acc_hash);
        }
    }));
}

void bench_dyn(acn::u32 n) {
    using Int = acn::CheckedI64;
    using AcnDyn = acn::AcnDyn<Int>;
    std::mt19937_64 rng(2000 + n);
    std::uniform_int_distribution<acn::i64> dist(-5, 5);
    std::vector<acn::i64> coeffs_a(n), coeffs_b(n);
    for (auto& v : coeffs_a)
        v = dist(rng);
    for (auto& v : coeffs_b)
        v = dist(rng);
    auto a = AcnDyn::from_ints(n, coeffs_a, 0);
    auto b = AcnDyn::from_ints(n, coeffs_b, 0);
    auto start_acc = AcnDyn::zero(n);

    char name[128];

    std::snprintf(name, sizeof(name), "add<AcnDyn,n=%u>", n);
    print_result(run(name, [&](std::uint64_t iters) {
        for (std::uint64_t i = 0; i < iters; ++i) {
            auto r = a + b;
            do_not_optimize(r);
        }
    }));

    std::snprintf(name, sizeof(name), "mul<AcnDyn,n=%u>", n);
    print_result(run(name, [&](std::uint64_t iters) {
        for (std::uint64_t i = 0; i < iters; ++i) {
            auto r = a * b;
            do_not_optimize(r);
        }
    }));

    std::snprintf(name, sizeof(name), "fma<AcnDyn,n=%u>", n);
    print_result(run(name, [&](std::uint64_t iters) {
        for (std::uint64_t i = 0; i < iters; ++i) {
            auto acc = start_acc;
            fma_into(acc, a, b);
            do_not_optimize(acc);
        }
    }));

    std::snprintf(name, sizeof(name), "canonicalize<AcnDyn,n=%u> (via scale(0))", n);
    print_result(run(name, [&](std::uint64_t iters) {
        for (std::uint64_t i = 0; i < iters; ++i) {
            auto r = a.scale(0);
            do_not_optimize(r);
        }
    }));

    std::snprintf(name, sizeof(name), "operator==<AcnDyn,n=%u>", n);
    print_result(run(name, [&](std::uint64_t iters) {
        bool acc_eq = false;
        for (std::uint64_t i = 0; i < iters; ++i) {
            acc_eq = (a == b);
            do_not_optimize(acc_eq);
        }
    }));

    std::snprintf(name, sizeof(name), "hash_value<AcnDyn,n=%u>", n);
    print_result(run(name, [&](std::uint64_t iters) {
        acn::u64 acc_hash = 0;
        for (std::uint64_t i = 0; i < iters; ++i) {
            acc_hash = acn::hash_value(a);
            do_not_optimize(acc_hash);
        }
    }));
}

}  // namespace

int main() {
    std::printf("=== Acn<N, CheckedI64> ===\n");
    bench_precision<4, acn::CheckedI64>("i64");
    bench_precision<8, acn::CheckedI64>("i64");
    bench_precision<16, acn::CheckedI64>("i64");
    bench_precision<32, acn::CheckedI64>("i64");

#if defined(ACN_WITH_GMP)
    std::printf("\n=== Acn<4, BigInt> ===\n");
    bench_precision<4, acn::BigInt>("BigInt");
#endif

    std::printf("\n=== AcnDyn<CheckedI64> ===\n");
    bench_dyn(4);
    bench_dyn(8);
    bench_dyn(16);
    bench_dyn(32);

    return 0;
}
