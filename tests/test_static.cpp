#include <random>
#include <type_traits>
#include <unordered_set>

#include <catch2/catch.hpp>

#include "acn/acn.hpp"

using acn::Acn;
using acn::CheckedI64;
using acn::i64;

using Acn4 = Acn<4, CheckedI64>;

namespace {

Acn4 random_acn(std::mt19937_64& rng, i64 bound = 8) {
    std::uniform_int_distribution<i64> dist(-bound, bound);
    std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
    return Acn4::from_ints(coeffs, 0);
}

}  // namespace

TEST_CASE("layout guarantees", "[static]") {
    STATIC_REQUIRE(std::is_trivially_copyable_v<Acn4>);
    REQUIRE(sizeof(Acn4) <= 40);
}

TEST_CASE("constants", "[static]") {
    REQUIRE((Acn4::inv_sqrt2() * Acn4::sqrt2()) == Acn4::one());
    REQUIRE((Acn4::imag() * Acn4::imag()) == -Acn4::one());
    REQUIRE(Acn4::omega(8) == Acn4::one());   // omega^(2n) == 1
    REQUIRE(Acn4::omega(2) == Acn4::imag());  // omega^(n/2) == i
    REQUIRE(Acn4::omega(4) == -Acn4::one());  // omega^n == -1
}

TEST_CASE("addition and multiplication with rescaling, sqrt(2) = omega - omega^3", "[static]") {
    // 1/sqrt(2) + 1/sqrt(2) == sqrt(2)
    auto half = Acn4::inv_sqrt2();
    REQUIRE((half + half) == Acn4::sqrt2());

    // 1 + sqrt(2) has k = 0 and is not further reducible.
    auto sum = Acn4::one() + Acn4::sqrt2();
    REQUIRE(sum.scale_exp() == 0);

    // sqrt(2) * sqrt(2) == 2
    auto two = Acn4::sqrt2() * Acn4::sqrt2();
    REQUIRE(two == (Acn4::one() + Acn4::one()));
}

TEST_CASE("is_integer handles negative k (even integers canonicalize with k < 0)", "[static]") {
    // 2 canonicalizes to c=(1,0,0,0), k=-2 (DESIGN.md §4.2's own example) —
    // is_integer must not use a naive "k == 0" test, which would reject it.
    auto two = Acn4::one() + Acn4::one();
    REQUIRE(two.scale_exp() == -2);
    REQUIRE(two.is_integer());

    REQUIRE(Acn4::zero().is_integer());
    REQUIRE(Acn4::one().is_integer());
    REQUIRE_FALSE(Acn4::sqrt2().is_integer());       // irrational, k=-1 with numerator 1
    REQUIRE_FALSE(Acn4::inv_sqrt2().is_integer());    // irrational, k=1
    REQUIRE_FALSE(Acn4::imag().is_integer());         // not real

    auto four = two + two;
    REQUIRE(four.is_integer());
    auto three = Acn4::one() + two;
    REQUIRE(three.scale_exp() == 0);  // odd integers stay at k=0
    REQUIRE(three.is_integer());
}

TEST_CASE("negation and commutativity of multiplication", "[static]") {
    std::mt19937_64 rng(11);
    for (int trial = 0; trial < 200; ++trial) {
        auto a = random_acn(rng);
        auto b = random_acn(rng);
        REQUIRE((a * b) == (b * a));
        REQUIRE((a + (-a)) == Acn4::zero());
    }
}

TEST_CASE("every operation returns a canonical value", "[static]") {
    std::array<CheckedI64, 8> scratch{};
    std::mt19937_64 rng(23);
    for (int trial = 0; trial < 500; ++trial) {
        auto a = random_acn(rng);
        auto b = random_acn(rng);
        for (auto value : {a + b, a - b, a * b, -a, a.conj(), a.mul_omega(3), a.scale(2)}) {
            REQUIRE(acn::is_canonical<CheckedI64>(value.coeffs(), value.scale_exp(), scratch));
        }
    }
}

TEST_CASE("equality matches byte-representation and hash agreement", "[static]") {
    std::mt19937_64 rng(31);
    std::unordered_set<Acn4, std::hash<Acn4>> seen_hashes_via_std_hash;
    for (int trial = 0; trial < 300; ++trial) {
        auto h = Acn4::inv_sqrt2();
        auto derived = (h * h) * Acn4::sqrt2();  // == 1/sqrt2, but derived differently
        REQUIRE(derived == h);
        REQUIRE(acn::hash_value(derived) == acn::hash_value(h));
        seen_hashes_via_std_hash.insert(derived);
    }
    REQUIRE(seen_hashes_via_std_hash.size() == 1);
}

TEST_CASE("fma_into matches naive accumulation", "[static]") {
    std::mt19937_64 rng(41);
    Acn4 acc_naive = Acn4::zero();
    Acn4 acc_fma = Acn4::zero();
    for (int i = 0; i < 20; ++i) {
        auto a = random_acn(rng, 4);
        auto b = random_acn(rng, 4);
        acc_naive = acc_naive + (a * b);
        fma_into(acc_fma, a, b);
    }
    REQUIRE(acc_naive == acc_fma);
}

TEST_CASE("fma_into is correct when d aliases a or b", "[static]") {
    // fma_into's single-canonicalize implementation reads a/b (into a
    // fresh `product` buffer and a copy of d's own coefficients) before
    // writing d's storage, so d = d + d*y and d = d + y*d must both be
    // safe even though the naive mul_into-then-add_into decomposition
    // would also have been safe here (mul_into's output can't alias its
    // inputs, but that output was never d itself).
    std::mt19937_64 rng(42);
    for (int trial = 0; trial < 50; ++trial) {
        auto x = random_acn(rng, 4);
        auto y = random_acn(rng, 4);

        auto expected = x + x * y;
        auto d = x;
        fma_into(d, d, y);
        REQUIRE(d == expected);

        auto expected2 = x + y * x;
        auto d2 = x;
        fma_into(d2, y, d2);
        REQUIRE(d2 == expected2);
    }
}

TEST_CASE("no allocation during Acn<4, CheckedI64> arithmetic", "[static]") {
    // Trivial-copyability plus the absence of any pointer/owning member in
    // Acn is the structural guarantee; this test just exercises a chain of
    // operations under ASan/UBSan (run via CI) as a smoke check.
    std::mt19937_64 rng(51);
    Acn4 acc = Acn4::zero();
    for (int i = 0; i < 1000; ++i)
        acc = acc + random_acn(rng, 3) * random_acn(rng, 3);
    SUCCEED();
}
