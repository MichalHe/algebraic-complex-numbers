// Only built when ACN_STATS=ON (see tests/CMakeLists.txt) -- with the
// default ACN_STATS=OFF, every ACN_STAT_* site compiles to nothing, so
// there's nothing runtime-observable to test in that configuration.

#include <random>
#include <sstream>

#include <catch2/catch.hpp>

#include "acn/acn.hpp"

using acn::Acn;
using acn::AcnDyn;
using acn::CheckedI64;
using acn::i64;
using acn::stats::counters;

using Acn4 = Acn<4, CheckedI64>;

namespace {

Acn4 random_acn(std::mt19937_64& rng, i64 bound = 6) {
    std::uniform_int_distribution<i64> dist(-bound, bound);
    std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
    return Acn4::from_ints(coeffs, 0);
}

}  // namespace

TEST_CASE("add/sub/mul/fma counters increment exactly once per call", "[stats]") {
    counters().reset();
    std::mt19937_64 rng(1);
    auto a = random_acn(rng);
    auto b = random_acn(rng);

    auto sum = a + b;
    REQUIRE(counters().add_count == 1);
    auto diff = a - b;
    REQUIRE(counters().sub_count == 1);
    auto prod = a * b;
    REQUIRE(counters().mul_count == 1);
    Acn4 acc = Acn4::zero();
    fma_into(acc, a, b);
    REQUIRE(counters().fma_count == 1);

    (void)sum;
    (void)diff;
    (void)prod;
}

TEST_CASE("canonicalize_calls counts every canonicalizing operation", "[stats]") {
    counters().reset();
    // from_ints (x2) + one multiply below: exactly 3 canonicalizing calls.
    auto c1 = Acn4::from_ints({1, 2, 3, 4}, 0);
    auto c2 = Acn4::from_ints({5, 6, 7, 8}, 0);
    auto prod = c1 * c2;
    (void)prod;
    REQUIRE(counters().canonicalize_calls == 3);
}

TEST_CASE("mul_by_sqrt2_calls and canonicalize_iterations increment on an odd-k rescale", "[stats]") {
    counters().reset();
    // sqrt2() itself needs one odd-reduction step to canonicalize (0,1,0,-1)
    // down to (1,0,0,0),k=-1, so it alone exercises the odd-step path.
    auto s = Acn4::sqrt2();
    REQUIRE(s.scale_exp() == -1);
    REQUIRE(counters().mul_by_sqrt2_calls >= 1);
    REQUIRE(counters().canonicalize_iterations >= 1);
}

TEST_CASE("scale_alignment counters track add/sub's scale-alignment work", "[stats]") {
    counters().reset();
    auto sqrt2 = Acn4::sqrt2();       // k = -1
    auto inv_sqrt2 = Acn4::inv_sqrt2();  // k = 1
    auto combined = sqrt2 + inv_sqrt2;
    (void)combined;
    REQUIRE(counters().scale_alignment_calls == 1);
    // target_k = max(-1,1) = 1; distance = (1-(-1)) + (1-1) = 2.
    REQUIRE(counters().scale_alignment_total_distance == 2);
}

TEST_CASE("widen/narrow counters", "[stats]") {
    counters().reset();
    auto a = Acn4::from_ints({1, -2, 3, 0}, 0);
    auto widened = a.widen<8>();
    REQUIRE(counters().widen_calls == 1);
    auto narrowed = widened.narrow<4>();
    REQUIRE(counters().narrow_calls == 1);
    REQUIRE(narrowed.has_value());
}

TEST_CASE("AcnDyn counters mirror Acn<N>'s", "[stats][dynamic]") {
    counters().reset();
    std::array<i64, 4> coeffs{1, -2, 3, 0};
    auto a = AcnDyn<CheckedI64>::from_ints(4, coeffs, 0);
    auto b = AcnDyn<CheckedI64>::from_ints(4, coeffs, 0);
    auto sum = a + b;
    REQUIRE(counters().add_count == 1);
    auto prod = a * b;
    REQUIRE(counters().mul_count == 1);
    auto widened = a.widen(8);
    REQUIRE(counters().widen_calls == 1);
    (void)sum;
    (void)prod;
    (void)widened;
}

TEST_CASE("overflow_flags_seen increments on Checked<i64> overflow", "[stats]") {
    counters().reset();
    acn::CheckedI64::clear_overflow();
    volatile i64 huge = 9223372036854775807LL;  // i64 max
    auto overflowed = acn::CheckedI64(huge) + acn::CheckedI64(huge);
    (void)overflowed;
    REQUIRE(acn::CheckedI64::overflow_seen());
    REQUIRE(counters().overflow_flags_seen >= 1);
    acn::CheckedI64::clear_overflow();
}

#if defined(ACN_WITH_GMP)

#include "acn/bigint.hpp"

TEST_CASE("bignum_promotions increments on promote()", "[stats][bigint]") {
    counters().reset();
    auto a = Acn4::from_ints({1, 2, 3, 4}, 0);
    auto promoted = acn::promote(a);
    (void)promoted;
    REQUIRE(counters().bignum_promotions == 1);
}

#endif
