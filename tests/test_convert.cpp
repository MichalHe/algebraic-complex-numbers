#include <random>

#include <catch2/catch.hpp>

#include "acn/acn.hpp"

using acn::Acn;
using acn::AcnDyn;
using acn::CheckedI64;
using acn::i64;

using Acn4 = Acn<4, CheckedI64>;
using Acn8 = Acn<8, CheckedI64>;
using Acn16 = Acn<16, CheckedI64>;

TEST_CASE("widen is a ring homomorphism: preserves +, *, 1", "[convert]") {
    auto a = Acn4::from_ints({1, -2, 3, 0}, 0);
    auto b = Acn4::from_ints({0, 1, -1, 2}, 0);

    auto wa = a.widen<8>();
    auto wb = b.widen<8>();

    REQUIRE((a + b).widen<8>() == (wa + wb));
    REQUIRE((a * b).widen<8>() == (wa * wb));
    REQUIRE(Acn4::one().widen<8>() == Acn8::one());
    REQUIRE(Acn4::zero().widen<8>() == Acn8::zero());
}

TEST_CASE("widen then narrow is the identity", "[convert]") {
    std::mt19937_64 rng(5);
    std::uniform_int_distribution<i64> dist(-10, 10);
    for (int trial = 0; trial < 200; ++trial) {
        std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
        auto a = Acn4::from_ints(coeffs, 0);
        auto widened = a.widen<16>();
        auto narrowed = widened.narrow<4>();
        REQUIRE(narrowed.has_value());
        REQUIRE(*narrowed == a);
    }
}

TEST_CASE("narrow fails when off-sublattice coefficients are nonzero", "[convert]") {
    // omega_8 itself (index 1 at n=8) is not in the image of R_4 -> R_8
    // (which only ever populates even indices), so it must not narrow.
    auto not_narrowable = Acn8::omega(1);
    REQUIRE_FALSE(not_narrowable.narrow<4>().has_value());
}

TEST_CASE("AcnDyn widen/narrow agree with Acn<N>", "[convert][dynamic]") {
    std::mt19937_64 rng(6);
    std::uniform_int_distribution<i64> dist(-10, 10);
    for (int trial = 0; trial < 100; ++trial) {
        std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
        auto static_a = Acn4::from_ints(coeffs, 0);
        auto dyn_a = AcnDyn<CheckedI64>::from_ints(4, coeffs, 0);

        auto static_widened = static_a.widen<16>();
        auto dyn_widened = dyn_a.widen(16);
        REQUIRE(static_widened.scale_exp() == dyn_widened.scale_exp());
        for (std::size_t j = 0; j < 16; ++j)
            REQUIRE(static_widened.coeffs()[j].value() == dyn_widened.coeffs()[j].value());

        auto dyn_narrowed = dyn_widened.narrow(4);
        REQUIRE(dyn_narrowed.has_value());
        REQUIRE(*dyn_narrowed == dyn_a);
    }
}

TEST_CASE("is_rational and is_unit", "[convert]") {
    REQUIRE(Acn4::zero().is_rational());
    REQUIRE(Acn4::one().is_rational());
    REQUIRE((Acn4::one() + Acn4::one()).is_rational());  // 2, canonical k=-2
    REQUIRE_FALSE(Acn4::sqrt2().is_rational());
    REQUIRE_FALSE(Acn4::inv_sqrt2().is_rational());
    REQUIRE_FALSE(Acn4::imag().is_rational());  // rational reals only, i is not real

    REQUIRE(Acn4::one().is_unit());
    REQUIRE((-Acn4::one()).is_unit());
    REQUIRE(Acn4::imag().is_unit());
    REQUIRE(Acn4::sqrt2().is_unit());
    REQUIRE(Acn4::inv_sqrt2().is_unit());
    REQUIRE(Acn4::omega(3).is_unit());
    // 2 == sqrt(2)^2 *is* a unit here: allowing negative k adjoins 1/sqrt(2)
    // (DESIGN.md §4.2), so 1/2 = (1/sqrt(2))^2 is representable too.
    REQUIRE((Acn4::one() + Acn4::one()).is_unit());
    // 3 is a genuine non-unit: its canonical form stays c=(3,0,0,0), k=0.
    auto three = Acn4::one() + Acn4::one() + Acn4::one();
    REQUIRE(three.scale_exp() == 0);
    REQUIRE_FALSE(three.is_unit());
    REQUIRE_FALSE(Acn4::zero().is_unit());
    // 1+i *is* a unit (== sqrt(2)*omega, canonicalizes to a single
    // coefficient) so it's not a counterexample; 2+i has norm 5, not a
    // unit, and stays genuinely two-coefficient after canonicalizing.
    REQUIRE((Acn4::one() + Acn4::imag()).is_unit());
    auto two_plus_i = Acn4::one() + Acn4::one() + Acn4::imag();
    REQUIRE_FALSE(two_plus_i.is_unit());
}
