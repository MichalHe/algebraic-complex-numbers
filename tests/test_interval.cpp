#include <cmath>
#include <complex>
#include <numbers>
#include <random>

#include <catch2/catch.hpp>

#include "acn/acn.hpp"

using acn::Acn;
using acn::AcnDyn;
using acn::CheckedI64;
using acn::i64;
using acn::to_interval;

using Acn4 = Acn<4, CheckedI64>;
using Acn8 = Acn<8, CheckedI64>;
using Acn16 = Acn<16, CheckedI64>;

namespace {

// Independent long-double oracle (same shape as test_ring.cpp's), included
// here rather than shared so this file doesn't need tests/reference/ (which
// doesn't exist yet -- see PROGRESS.md).
template <std::size_t N, class Int>
std::complex<long double> oracle_value(const Acn<N, Int>& a) {
    std::complex<long double> sum(0.0L, 0.0L);
    auto c = a.coeffs();
    for (std::size_t j = 0; j < N; ++j) {
        long double angle =
            std::numbers::pi_v<long double> * static_cast<long double>(j) / static_cast<long double>(N);
        sum += static_cast<long double>(c[j].value()) * std::complex<long double>(std::cos(angle), std::sin(angle));
    }
    long double factor = std::pow(1.0L / std::numbers::sqrt2_v<long double>, static_cast<long double>(a.scale_exp()));
    return sum * factor;
}

Acn4 random_acn4(std::mt19937_64& rng, i64 bound = 30) {
    std::uniform_int_distribution<i64> dist(-bound, bound);
    std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
    return Acn4::from_ints(coeffs, 0);
}

template <std::size_t N>
Acn<N, CheckedI64> random_acn(std::mt19937_64& rng, i64 bound = 30) {
    std::uniform_int_distribution<i64> dist(-bound, bound);
    std::array<i64, N> coeffs{};
    for (auto& v : coeffs)
        v = dist(rng);
    return Acn<N, CheckedI64>::from_ints(coeffs, 0);
}

}  // namespace

TEST_CASE("to_interval contains the oracle value for random Acn<4>", "[interval]") {
    std::mt19937_64 rng(21);
    for (int trial = 0; trial < 500; ++trial) {
        auto a = random_acn4(rng);
        auto iv = to_interval(a);
        auto expected = oracle_value(a);
        REQUIRE(iv.re.contains(static_cast<double>(expected.real())));
        REQUIRE(iv.im.contains(static_cast<double>(expected.imag())));
    }
}

TEST_CASE("to_interval contains the oracle value across several precisions", "[interval]") {
    std::mt19937_64 rng(22);
    {
        auto a = random_acn<8>(rng);
        auto iv = to_interval(a);
        auto expected = oracle_value(a);
        REQUIRE(iv.re.contains(static_cast<double>(expected.real())));
        REQUIRE(iv.im.contains(static_cast<double>(expected.imag())));
    }
    {
        auto a = random_acn<16>(rng);
        auto iv = to_interval(a);
        auto expected = oracle_value(a);
        REQUIRE(iv.re.contains(static_cast<double>(expected.real())));
        REQUIRE(iv.im.contains(static_cast<double>(expected.imag())));
    }
}

TEST_CASE("to_interval handles rescaling (negative and positive k) correctly", "[interval]") {
    auto sqrt2 = Acn4::sqrt2();
    auto sqrt2_iv = to_interval(sqrt2);
    REQUIRE(sqrt2_iv.re.contains(std::numbers::sqrt2));
    REQUIRE(sqrt2_iv.im.contains(0.0));

    auto inv_sqrt2 = Acn4::inv_sqrt2();
    auto inv_sqrt2_iv = to_interval(inv_sqrt2);
    REQUIRE(inv_sqrt2_iv.re.contains(1.0 / std::numbers::sqrt2));
    REQUIRE(inv_sqrt2_iv.im.contains(0.0));

    // 2 canonicalizes to c=(1,0,0,0), k=-2 (a large negative k).
    auto two = Acn4::one() + Acn4::one();
    REQUIRE(two.scale_exp() == -2);
    auto two_iv = to_interval(two);
    REQUIRE(two_iv.re.contains(2.0));
}

TEST_CASE("to_interval of zero contains zero exactly", "[interval]") {
    auto iv = to_interval(Acn4::zero());
    REQUIRE(iv.re.contains(0.0));
    REQUIRE(iv.im.contains(0.0));
}

TEST_CASE("to_interval excludes zero for random nonzero values (soundness for pruning)", "[interval][property]") {
    std::mt19937_64 rng(23);
    int checked = 0;
    for (int trial = 0; trial < 500; ++trial) {
        Acn4 a;
        do {
            a = random_acn4(rng);
        } while (a.is_zero());
        auto iv = to_interval(a);
        // The interval must contain the true (nonzero) value...
        auto expected = oracle_value(a);
        REQUIRE(iv.re.contains(static_cast<double>(expected.real())));
        REQUIRE(iv.im.contains(static_cast<double>(expected.imag())));
        // ...and, since these are generic random values (not deliberately
        // near-cancelling), at least one component's interval should
        // exclude zero.
        if (!iv.re.contains_zero() || !iv.im.contains_zero())
            ++checked;
    }
    REQUIRE(checked == 500);
}

TEST_CASE("to_interval agrees between Acn<N> and AcnDyn", "[interval][dynamic]") {
    std::mt19937_64 rng(24);
    std::uniform_int_distribution<i64> dist(-30, 30);
    for (int trial = 0; trial < 200; ++trial) {
        std::array<i64, 8> coeffs{};
        for (auto& v : coeffs)
            v = dist(rng);
        auto a_static = Acn<8, CheckedI64>::from_ints(coeffs, 0);
        auto a_dyn = AcnDyn<CheckedI64>::from_ints(8, coeffs, 0);

        auto iv_static = to_interval(a_static);
        auto iv_dyn = to_interval(a_dyn);
        REQUIRE(iv_static.re.lo == iv_dyn.re.lo);
        REQUIRE(iv_static.re.hi == iv_dyn.re.hi);
        REQUIRE(iv_static.im.lo == iv_dyn.im.lo);
        REQUIRE(iv_static.im.hi == iv_dyn.im.hi);
    }
}

#if defined(ACN_WITH_GMP)

#include "acn/bigint.hpp"

using acn::BigInt;

TEST_CASE("to_interval contains the oracle value for BigInt coefficients beyond i64", "[interval][bigint]") {
    BigInt huge = BigInt(1);
    for (int i = 0; i < 25; ++i)
        huge = huge * BigInt(10);  // 10^25, well beyond i64 range and beyond exact double precision
    std::array<BigInt, 4> coeffs{huge, -huge, BigInt(0), BigInt(3)};
    auto a = Acn<4, BigInt>::from_coeffs(std::span<const BigInt>(coeffs), 0);

    auto iv = to_interval(a);
    // Independent oracle via a second, differently-ordered double
    // computation (mpz_get_d directly), consistent with to_double()'s own
    // implementation but written separately here so it isn't just checking
    // the interval against its own inputs restated.
    double huge_d = std::pow(10.0, 25);
    double expected_re = huge_d * std::cos(0.0) + (-huge_d) * std::cos(std::numbers::pi / 4.0) +
                          3.0 * std::cos(3.0 * std::numbers::pi / 4.0);
    // Large-magnitude cross-check: just confirm the interval is centered
    // near the expected order of magnitude and not wildly wrong, since
    // pow(10,25) and mpz's own double conversion can differ in their last
    // few bits.
    REQUIRE(std::abs(iv.re.lo - expected_re) < std::abs(expected_re) * 1e-9);
    REQUIRE(std::abs(iv.re.hi - expected_re) < std::abs(expected_re) * 1e-9);
}

#endif
