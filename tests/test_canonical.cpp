#include <array>
#include <random>

#include <catch2/catch.hpp>

#include "acn/canonical.hpp"
#include "acn/int_backend.hpp"
#include "acn/ring.hpp"

using acn::CheckedI64;
using acn::i64;
using acn::ScaleExp;

namespace {

template <std::size_t N>
std::array<CheckedI64, N> random_nonzero(std::mt19937_64& rng, i64 bound = 30) {
    std::uniform_int_distribution<i64> dist(-bound, bound);
    std::array<CheckedI64, N> out{};
    do {
        for (auto& v : out)
            v = CheckedI64(dist(rng));
    } while (acn::all_zero<CheckedI64>(out));
    return out;
}

}  // namespace

TEST_CASE("canonicalize idempotence", "[canonical]") {
    std::mt19937_64 rng(1);
    for (int trial = 0; trial < 500; ++trial) {
        auto c = random_nonzero<4>(rng);
        ScaleExp k = 3;
        std::array<CheckedI64, 8> scratch{};
        acn::canonicalize<CheckedI64>(c, k, scratch);

        auto c2 = c;
        ScaleExp k2 = k;
        acn::canonicalize<CheckedI64>(c2, k2, scratch);
        REQUIRE(c2 == c);
        REQUIRE(k2 == k);
    }
}

TEST_CASE("canonicalize produces a canonical value", "[canonical]") {
    std::mt19937_64 rng(2);
    for (int trial = 0; trial < 500; ++trial) {
        auto c = random_nonzero<4>(rng);
        ScaleExp k = 0;
        std::array<CheckedI64, 8> scratch{};
        acn::canonicalize<CheckedI64>(c, k, scratch);
        REQUIRE(acn::is_canonical<CheckedI64>(c, k, scratch));
    }
}

TEST_CASE("uniqueness: multiplying numerator and k by matching sqrt2 power canonicalizes to the same value",
          "[canonical]") {
    std::mt19937_64 rng(3);
    for (int trial = 0; trial < 300; ++trial) {
        auto x = random_nonzero<4>(rng, 10);
        ScaleExp kx = 0;
        std::array<CheckedI64, 8> scratch{};
        acn::canonicalize<CheckedI64>(x, kx, scratch);

        std::uniform_int_distribution<int> d_dist(0, 6);
        int d = d_dist(rng);

        std::array<CheckedI64, 4> scaled = x;
        for (int i = 0; i < d; ++i) {
            std::array<CheckedI64, 4> next{}, sq_scratch{};
            acn::ring::mul_by_sqrt2<CheckedI64>(next, scaled, sq_scratch);
            scaled = next;
        }
        ScaleExp k_scaled = static_cast<ScaleExp>(kx + d);
        acn::canonicalize<CheckedI64>(scaled, k_scaled, scratch);

        REQUIRE(scaled == x);
        REQUIRE(k_scaled == kx);
    }
}

TEST_CASE("known canonical scale exponents", "[canonical]") {
    std::array<CheckedI64, 8> scratch{};

    // 1/sqrt(2): numerator 1, k = 1.
    {
        std::array<CheckedI64, 4> c{CheckedI64(1), CheckedI64(0), CheckedI64(0), CheckedI64(0)};
        ScaleExp k = 1;
        acn::canonicalize<CheckedI64>(c, k, scratch);
        REQUIRE(k == 1);
        REQUIRE(c[0].value() == 1);
    }

    // sqrt(2) = omega - omega^3 at k=0 is itself divisible by sqrt(2) (quotient 1),
    // so it must canonicalize down to numerator 1 at k = -1.
    {
        std::array<CheckedI64, 4> c{CheckedI64(0), CheckedI64(1), CheckedI64(0), CheckedI64(-1)};
        ScaleExp k = 0;
        acn::canonicalize<CheckedI64>(c, k, scratch);
        REQUIRE(k == -1);
        REQUIRE(c[0].value() == 1);
        REQUIRE(c[1].is_zero());
        REQUIRE(c[2].is_zero());
        REQUIRE(c[3].is_zero());
    }

    // 2 = (1,0,0,0) at k=-2.
    {
        std::array<CheckedI64, 4> c{CheckedI64(2), CheckedI64(0), CheckedI64(0), CheckedI64(0)};
        ScaleExp k = 0;
        acn::canonicalize<CheckedI64>(c, k, scratch);
        REQUIRE(k == -2);
        REQUIRE(c[0].value() == 1);
    }

    // zero always canonicalizes to k = 0.
    {
        std::array<CheckedI64, 4> c{};
        ScaleExp k = 17;
        acn::canonicalize<CheckedI64>(c, k, scratch);
        REQUIRE(k == 0);
    }
}

TEST_CASE("is_canonical rejects constructed non-canonical inputs", "[canonical]") {
    std::array<CheckedI64, 8> scratch{};
    // (2,0,0,0) at k=0 is not canonical: 2 = sqrt2 * sqrt2, so sqrt2 divides it.
    std::array<CheckedI64, 4> c{CheckedI64(2), CheckedI64(0), CheckedI64(0), CheckedI64(0)};
    REQUIRE_FALSE(acn::is_canonical<CheckedI64>(c, 0, scratch));
}
