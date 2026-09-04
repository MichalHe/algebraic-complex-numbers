#include <random>

#include <catch2/catch.hpp>

#include "acn/acn.hpp"

using acn::Acn;
using acn::AcnDyn;
using acn::BigInt;
using acn::CheckedI64;
using acn::i64;

TEST_CASE("BigInt basic arithmetic", "[bigint]") {
    BigInt a(3), b(5);
    REQUIRE((a + b) == BigInt(8));
    REQUIRE((a - b) == BigInt(-2));
    REQUIRE((a * b) == BigInt(15));
    REQUIRE(-a == BigInt(-3));
    REQUIRE(BigInt::overflow_seen() == false);
}

TEST_CASE("BigInt actually exceeds i64 without overflowing", "[bigint]") {
    // 2^100, far outside i64 range: this is the whole point of the backend.
    BigInt value(1);
    BigInt two(2);
    for (int i = 0; i < 100; ++i)
        value = value * two;
    REQUIRE_FALSE(value.fits_i64());
    REQUIRE(value.is_even());
    BigInt halved = value.halve_exact();
    REQUIRE((halved * two) == value);
}

TEST_CASE("BigInt parity, ctz and halving agree with Checked<i64> in range", "[bigint]") {
    for (i64 x : {1, 2, 4, 6, 8, 12, 96, 1024, -6, -96}) {
        BigInt b(x);
        CheckedI64 c(x);
        REQUIRE(b.is_even() == c.is_even());
        if (x != 0)
            REQUIRE(b.ctz() == c.ctz());
    }
    REQUIRE(BigInt(6).halve_exact() == BigInt(3));
    REQUIRE(BigInt(-6).halve_exact() == BigInt(-3));
}

TEST_CASE("BigInt shift_left matches multiplication by a power of two", "[bigint]") {
    REQUIRE(BigInt(3).shift_left(4) == BigInt(48));
}

TEST_CASE("BigInt ordering and equality", "[bigint]") {
    REQUIRE(BigInt(3) == BigInt(3));
    REQUIRE(BigInt(3) != BigInt(4));
    REQUIRE(BigInt(3) < BigInt(4));
    REQUIRE(BigInt(-100000) < BigInt(3));
}

TEST_CASE("BigInt copy/move semantics", "[bigint]") {
    BigInt a(42);
    BigInt b = a;
    REQUIRE(b == a);
    BigInt c = std::move(b);
    REQUIRE(c == a);
    BigInt d(0);
    d = c;
    REQUIRE(d == a);
    BigInt e(0);
    e = std::move(d);
    REQUIRE(e == a);
}

TEST_CASE("Acn<4, BigInt> passes the same identities as Acn<4, Checked<i64>>", "[bigint][static]") {
    using Acn4Big = Acn<4, BigInt>;
    REQUIRE((Acn4Big::inv_sqrt2() * Acn4Big::sqrt2()) == Acn4Big::one());
    REQUIRE((Acn4Big::imag() * Acn4Big::imag()) == -Acn4Big::one());
    REQUIRE(Acn4Big::omega(8) == Acn4Big::one());

    auto half = Acn4Big::inv_sqrt2();
    REQUIRE((half + half) == Acn4Big::sqrt2());

    auto two = Acn4Big::sqrt2() * Acn4Big::sqrt2();
    REQUIRE(two == (Acn4Big::one() + Acn4Big::one()));
    REQUIRE(two.is_integer());
    REQUIRE(two.scale_exp() == -2);  // same canonical shape as the i64 backend
}

TEST_CASE("Acn<4, BigInt> and Acn<4, Checked<i64>> agree for in-range values", "[bigint][differential]") {
    std::mt19937_64 rng(2026);
    std::uniform_int_distribution<i64> dist(-20, 20);
    for (int trial = 0; trial < 300; ++trial) {
        std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
        auto small = Acn<4, CheckedI64>::from_ints(coeffs, 0);
        auto big = Acn<4, BigInt>::from_ints(coeffs, 0);
        REQUIRE(small.scale_exp() == big.scale_exp());
        auto sc = small.coeffs();
        auto bc = big.coeffs();
        for (std::size_t j = 0; j < 4; ++j)
            REQUIRE(sc[j].value() == bc[j].to_i64_unchecked());
    }
}

TEST_CASE("AcnDyn<BigInt> works the same as AcnDyn<Checked<i64>> for in-range values", "[bigint][dynamic]") {
    auto small = AcnDyn<CheckedI64>::from_ints(4, std::array<i64, 4>{1, -2, 3, 0}, 0);
    auto big = AcnDyn<BigInt>::from_ints(4, std::array<i64, 4>{1, -2, 3, 0}, 0);
    auto small_sum = small + small;
    auto big_sum = big + big;
    REQUIRE(small_sum.scale_exp() == big_sum.scale_exp());
    for (std::size_t j = 0; j < 4; ++j)
        REQUIRE(small_sum.coeffs()[j].value() == big_sum.coeffs()[j].to_i64_unchecked());
}

TEST_CASE("hash_value agrees between equal Checked<i64> and BigInt values only via same-type comparison",
          "[bigint][hash]") {
    // hash_value is only meaningful within one backend (the two backends
    // have different hash_component() implementations by design); this
    // just checks BigInt-backed hashing is well-defined and stable.
    auto a = Acn<4, BigInt>::from_ints({1, 2, 3, 4}, 0);
    auto b = Acn<4, BigInt>::from_ints({1, 2, 3, 4}, 0);
    REQUIRE(acn::hash_value(a) == acn::hash_value(b));
}
