#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <vector>

#include <catch2/catch.hpp>

#include "acn/int_backend.hpp"
#include "acn/ring.hpp"

using acn::CheckedI64;
using acn::i64;

namespace {

template <std::size_t N>
std::complex<long double> evaluate(const std::array<CheckedI64, N>& c) {
    std::complex<long double> sum(0.0L, 0.0L);
    for (std::size_t j = 0; j < N; ++j) {
        long double angle = std::numbers::pi_v<long double> * static_cast<long double>(j) /
                             static_cast<long double>(N);
        sum += static_cast<long double>(c[j].value()) *
               std::complex<long double>(std::cos(angle), std::sin(angle));
    }
    return sum;
}

template <std::size_t N>
std::array<CheckedI64, N> random_vector(std::mt19937_64& rng, i64 bound = 20) {
    std::uniform_int_distribution<i64> dist(-bound, bound);
    std::array<CheckedI64, N> out{};
    for (auto& v : out)
        v = CheckedI64(dist(rng));
    return out;
}

}  // namespace

TEST_CASE("negacyclic multiply matches a hand-computed table for n=4", "[ring]") {
    // omega^j * omega^t == omega^(j+t) if j+t<4, else -omega^(j+t-4).
    for (int j = 0; j < 4; ++j) {
        for (int t = 0; t < 4; ++t) {
            std::array<CheckedI64, 4> a{}, b{}, out{};
            a[static_cast<std::size_t>(j)] = CheckedI64(1);
            b[static_cast<std::size_t>(t)] = CheckedI64(1);
            acn::ring::mul<CheckedI64>(out, a, b);

            std::array<CheckedI64, 4> expected{};
            int idx = j + t;
            if (idx < 4)
                expected[static_cast<std::size_t>(idx)] = CheckedI64(1);
            else
                expected[static_cast<std::size_t>(idx - 4)] = CheckedI64(-1);
            REQUIRE(out == expected);
        }
    }
}

TEST_CASE("mul_by_sqrt2(1) == omega - omega^3 for n=4", "[ring]") {
    std::array<CheckedI64, 4> one{CheckedI64(1), CheckedI64(0), CheckedI64(0), CheckedI64(0)};
    std::array<CheckedI64, 4> out{}, scratch{};
    acn::ring::mul_by_sqrt2<CheckedI64>(out, one, scratch);
    std::array<CheckedI64, 4> expected{CheckedI64(0), CheckedI64(1), CheckedI64(0), CheckedI64(-1)};
    REQUIRE(out == expected);
}

TEST_CASE("mul_by_sqrt2 twice equals doubling", "[ring][property]") {
    std::mt19937_64 rng(12345);
    for (int n : {4, 8, 16, 32}) {
        for (int trial = 0; trial < 2000; ++trial) {
            std::vector<CheckedI64> a(static_cast<std::size_t>(n)), once(static_cast<std::size_t>(n)),
                twice(static_cast<std::size_t>(n)), scratch(static_cast<std::size_t>(n));
            std::uniform_int_distribution<i64> dist(-50, 50);
            for (auto& v : a)
                v = CheckedI64(dist(rng));

            acn::ring::mul_by_sqrt2<CheckedI64>(once, a, scratch);
            acn::ring::mul_by_sqrt2<CheckedI64>(twice, once, scratch);

            for (std::size_t j = 0; j < a.size(); ++j)
                REQUIRE(twice[j].value() == 2 * a[j].value());
            REQUIRE_FALSE(CheckedI64::overflow_seen());
        }
    }
}

TEST_CASE("conj is an involution and matches norm2 via the general path", "[ring]") {
    std::mt19937_64 rng(7);
    auto a = random_vector<4>(rng);
    std::array<CheckedI64, 4> conjugated{}, roundtrip{};
    acn::ring::conj<CheckedI64>(conjugated, a);
    acn::ring::conj<CheckedI64>(roundtrip, conjugated);
    REQUIRE(roundtrip == a);

    std::array<CheckedI64, 4> norm{}, norm_scratch{}, via_mul{};
    acn::ring::norm2<CheckedI64>(norm, a, norm_scratch);
    acn::ring::mul<CheckedI64>(via_mul, a, conjugated);
    REQUIRE(norm == via_mul);

    // The result of a * conj(a) is real: its imaginary part, evaluated
    // numerically, is (numerically) zero.
    auto value = evaluate<4>(norm);
    REQUIRE(std::abs(value.imag()) < 1e-6L);
}

TEST_CASE("rotate identities", "[ring]") {
    std::mt19937_64 rng(99);
    auto a = random_vector<8>(rng);

    std::array<CheckedI64, 8> rotated_full{};
    acn::ring::rotate<CheckedI64>(rotated_full, a, 16);  // omega^(2n) == 1
    REQUIRE(rotated_full == a);

    std::array<CheckedI64, 8> rotated_half_turn{};
    acn::ring::rotate<CheckedI64>(rotated_half_turn, a, 8);  // omega^n == -1
    std::array<CheckedI64, 8> negated{};
    acn::ring::neg<CheckedI64>(negated, a);
    REQUIRE(rotated_half_turn == negated);

    // omega^(n/2) == i: rotating by n/2 matches multiplying the numeric
    // evaluation by i.
    std::array<CheckedI64, 8> rotated_quarter{};
    acn::ring::rotate<CheckedI64>(rotated_quarter, a, 4);
    auto original_value = evaluate<8>(a);
    auto rotated_value = evaluate<8>(rotated_quarter);
    auto expected = original_value * std::complex<long double>(0.0L, 1.0L);
    REQUIRE(std::abs(rotated_value - expected) < 1e-6L);
}

TEST_CASE("ring axioms hold on random inputs", "[ring][property]") {
    std::mt19937_64 rng(2024);
    for (int trial = 0; trial < 500; ++trial) {
        auto a = random_vector<4>(rng, 10);
        auto b = random_vector<4>(rng, 10);
        auto c = random_vector<4>(rng, 10);

        std::array<CheckedI64, 4> ab{}, ba{};
        acn::ring::mul<CheckedI64>(ab, a, b);
        acn::ring::mul<CheckedI64>(ba, b, a);
        REQUIRE(ab == ba);  // commutativity

        std::array<CheckedI64, 4> ab_c{}, bc{}, a_bc{};
        acn::ring::mul<CheckedI64>(bc, b, c);
        acn::ring::mul<CheckedI64>(ab_c, ab, c);
        acn::ring::mul<CheckedI64>(a_bc, a, bc);
        REQUIRE(ab_c == a_bc);  // associativity

        std::array<CheckedI64, 4> b_plus_c{}, a_b{}, a_c{}, a_b_plus_a_c{}, a_times_bc{};
        acn::ring::add<CheckedI64>(b_plus_c, b, c);
        acn::ring::mul<CheckedI64>(a_times_bc, a, b_plus_c);
        acn::ring::mul<CheckedI64>(a_b, a, b);
        acn::ring::mul<CheckedI64>(a_c, a, c);
        acn::ring::add<CheckedI64>(a_b_plus_a_c, a_b, a_c);
        REQUIRE(a_times_bc == a_b_plus_a_c);  // distributivity
    }
    REQUIRE_FALSE(CheckedI64::overflow_seen());
}

TEST_CASE("numeric cross-check of mul across several precisions", "[ring][property]") {
    std::mt19937_64 rng(42);
    for (int n : {1, 2, 4, 8, 16, 32}) {
        for (int trial = 0; trial < 200; ++trial) {
            std::size_t un = static_cast<std::size_t>(n);
            std::vector<CheckedI64> a(un), b(un), out(un);
            std::uniform_int_distribution<i64> dist(-15, 15);
            for (auto& v : a)
                v = CheckedI64(dist(rng));
            for (auto& v : b)
                v = CheckedI64(dist(rng));
            acn::ring::mul<CheckedI64>(out, a, b);

            std::complex<long double> ea(0.0L, 0.0L), eb(0.0L, 0.0L), eo(0.0L, 0.0L);
            for (std::size_t j = 0; j < un; ++j) {
                long double angle =
                    std::numbers::pi_v<long double> * static_cast<long double>(j) / static_cast<long double>(un);
                std::complex<long double> root(std::cos(angle), std::sin(angle));
                ea += static_cast<long double>(a[j].value()) * root;
                eb += static_cast<long double>(b[j].value()) * root;
                eo += static_cast<long double>(out[j].value()) * root;
            }
            REQUIRE(std::abs(eo - ea * eb) < 1e-3L);
        }
    }
}
