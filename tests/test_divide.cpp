#include <random>

#include <catch2/catch.hpp>

#include "acn/acn.hpp"

using acn::Acn;
using acn::AcnDyn;
using acn::CheckedI64;
using acn::divide_by_unit;
using acn::divide_exact;
using acn::i64;

using Acn4 = Acn<4, CheckedI64>;

namespace {

Acn4 random_acn(std::mt19937_64& rng, i64 bound = 6) {
    std::uniform_int_distribution<i64> dist(-bound, bound);
    std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
    return Acn4::from_ints(coeffs, 0);
}

}  // namespace

TEST_CASE("divide_exact(a*b, b) == a for random nonzero a, b", "[divide]") {
    std::mt19937_64 rng(101);
    int successes = 0;
    for (int trial = 0; trial < 500; ++trial) {
        Acn4 a, b;
        do {
            a = random_acn(rng, 5);
        } while (a.is_zero());
        do {
            b = random_acn(rng, 5);
        } while (b.is_zero());

        auto product = a * b;
        auto quotient = divide_exact(product, b);
        REQUIRE(quotient.has_value());
        REQUIRE(*quotient == a);
        ++successes;
    }
    REQUIRE(successes == 500);
}

TEST_CASE("divide_exact returns nullopt for division by zero", "[divide]") {
    auto a = Acn4::one();
    REQUIRE_FALSE(divide_exact(a, Acn4::zero()).has_value());
}

TEST_CASE("divide_exact detects genuine non-divisibility", "[divide]") {
    // 2 == sqrt(2)^2 is a unit here (negative k adjoins 1/sqrt(2), hence
    // 1/2), so it's not a counterexample. 3 genuinely is not a unit and
    // does not divide 1: its canonical form stays c=(3,0,0,0), k=0.
    auto one = Acn4::one();
    auto three = one + one + one;
    REQUIRE_FALSE(three.is_unit());
    REQUIRE_FALSE(divide_exact(one, three).has_value());
}

TEST_CASE("divide_exact of self is one", "[divide]") {
    std::mt19937_64 rng(202);
    for (int trial = 0; trial < 100; ++trial) {
        Acn4 a;
        do {
            a = random_acn(rng, 8);
        } while (a.is_zero());
        auto q = divide_exact(a, a);
        REQUIRE(q.has_value());
        REQUIRE(*q == Acn4::one());
    }
}

TEST_CASE("divide_by_unit agrees with divide_exact whenever the divisor is a unit", "[divide]") {
    std::mt19937_64 rng(303);
    std::vector<Acn4> units = {Acn4::one(),      -Acn4::one(),      Acn4::imag(),     -Acn4::imag(),
                                Acn4::sqrt2(),    Acn4::inv_sqrt2(), Acn4::omega(1),   Acn4::omega(3),
                                Acn4::omega(5),   Acn4::omega(7)};
    for (const auto& u : units)
        REQUIRE(u.is_unit());

    for (int trial = 0; trial < 200; ++trial) {
        auto a = random_acn(rng, 6);
        const auto& u = units[static_cast<std::size_t>(trial) % units.size()];
        auto via_unit = divide_by_unit(a, u);
        auto via_exact = divide_exact(a, u);
        REQUIRE(via_exact.has_value());
        REQUIRE(via_unit == *via_exact);
        // And it really is a's inverse-of-multiplication: via_unit * u == a.
        REQUIRE((via_unit * u) == a);
    }
}

TEST_CASE("divide_exact matches Acn<N> and AcnDyn identically", "[divide][dynamic]") {
    std::mt19937_64 rng(404);
    for (int trial = 0; trial < 200; ++trial) {
        std::array<i64, 4> a_coeffs{}, b_coeffs{};
        std::uniform_int_distribution<i64> dist(-6, 6);
        for (auto& v : a_coeffs)
            v = dist(rng);
        for (auto& v : b_coeffs)
            v = dist(rng);

        auto a_static = Acn4::from_ints(a_coeffs, 0);
        auto b_static = Acn4::from_ints(b_coeffs, 0);
        if (b_static.is_zero())
            continue;
        auto product_static = a_static * b_static;

        auto a_dyn = AcnDyn<CheckedI64>::from_ints(4, a_coeffs, 0);
        auto b_dyn = AcnDyn<CheckedI64>::from_ints(4, b_coeffs, 0);
        auto product_dyn = a_dyn * b_dyn;

        auto q_static = divide_exact(product_static, b_static);
        auto q_dyn = divide_exact(product_dyn, b_dyn);
        REQUIRE(q_static.has_value() == q_dyn.has_value());
        if (q_static) {
            REQUIRE(q_static->scale_exp() == q_dyn->scale_exp());
            for (std::size_t j = 0; j < 4; ++j)
                REQUIRE(q_static->coeffs()[j].value() == q_dyn->coeffs()[j].value());
        }
    }
}
