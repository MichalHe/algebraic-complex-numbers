#include <random>
#include <sstream>

#include <catch2/catch.hpp>

#include "acn/acn.hpp"

using acn::Acn;
using acn::AcnDyn;
using acn::CheckedI64;
using acn::i64;

using Acn4 = Acn<4, CheckedI64>;

namespace {

Acn4 random_acn(std::mt19937_64& rng, i64 bound = 20) {
    std::uniform_int_distribution<i64> dist(-bound, bound);
    std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
    return Acn4::from_ints(coeffs, 0);
}

}  // namespace

TEST_CASE("operator<< then parse round-trips for Acn<4, CheckedI64>", "[format]") {
    std::mt19937_64 rng(11);
    for (int trial = 0; trial < 300; ++trial) {
        auto a = random_acn(rng);
        std::ostringstream out;
        out << a;
        auto parsed = acn::parse<4, CheckedI64>(out.str());
        REQUIRE(parsed.has_value());
        REQUIRE(*parsed == a);
    }
}

TEST_CASE("parse ignores the trailing approximation comment", "[format]") {
    auto a = Acn4::from_ints({1, -2, 3, 0}, 0);
    std::ostringstream out;
    out << a;
    std::string text = out.str();
    REQUIRE(text.find('#') != std::string::npos);
    auto parsed = acn::parse<4, CheckedI64>(text);
    REQUIRE(parsed.has_value());
    REQUIRE(*parsed == a);
}

TEST_CASE("parse rejects malformed text", "[format]") {
    REQUIRE_FALSE(acn::parse<4, CheckedI64>("not a value").has_value());
    REQUIRE_FALSE(acn::parse<4, CheckedI64>("(1, 2, 3)/sqrt2^0").has_value());  // wrong coeff count
    REQUIRE_FALSE(acn::parse<4, CheckedI64>("(1, 2, 3, 4)/sqrt2^").has_value());  // missing exponent
    REQUIRE_FALSE(acn::parse<4, CheckedI64>("(1, 2, 3, x)/sqrt2^0").has_value());  // bad coefficient
}

TEST_CASE("operator<< then parse_dyn round-trips for AcnDyn<CheckedI64>", "[format][dynamic]") {
    std::mt19937_64 rng(12);
    std::uniform_int_distribution<i64> dist(-20, 20);
    for (int trial = 0; trial < 100; ++trial) {
        std::array<i64, 8> coeffs{};
        for (auto& v : coeffs)
            v = dist(rng);
        auto a = AcnDyn<CheckedI64>::from_ints(8, coeffs, 0);
        std::ostringstream out;
        out << a;
        auto parsed = acn::parse_dyn<CheckedI64>(out.str());
        REQUIRE(parsed.has_value());
        REQUIRE(*parsed == a);
    }
}

TEST_CASE("binary serialize/deserialize round-trips for Acn<4, CheckedI64>", "[format]") {
    std::mt19937_64 rng(13);
    for (int trial = 0; trial < 300; ++trial) {
        auto a = random_acn(rng);
        auto bytes = acn::serialize(a);
        auto restored = acn::deserialize<4, CheckedI64>(bytes);
        REQUIRE(restored.has_value());
        REQUIRE(*restored == a);
    }
}

TEST_CASE("binary deserialize rejects truncated or mismatched-precision input", "[format]") {
    std::mt19937_64 rng(14);
    auto a = random_acn(rng);
    auto bytes = acn::serialize(a);

    REQUIRE_FALSE(acn::deserialize<8, CheckedI64>(bytes).has_value());  // precision mismatch

    auto truncated = bytes;
    truncated.resize(truncated.size() - 1);
    REQUIRE_FALSE(acn::deserialize<4, CheckedI64>(truncated).has_value());
}

TEST_CASE("binary serialize_dyn/deserialize_dyn round-trips for AcnDyn<CheckedI64>", "[format][dynamic]") {
    std::mt19937_64 rng(15);
    std::uniform_int_distribution<i64> dist(-20, 20);
    for (int trial = 0; trial < 100; ++trial) {
        std::array<i64, 8> coeffs{};
        for (auto& v : coeffs)
            v = dist(rng);
        auto a = AcnDyn<CheckedI64>::from_ints(8, coeffs, 0);
        auto bytes = acn::serialize_dyn(a);
        auto restored = acn::deserialize_dyn<CheckedI64>(bytes);
        REQUIRE(restored.has_value());
        REQUIRE(*restored == a);
    }
}

#if defined(ACN_WITH_GMP)

#include "acn/bigint.hpp"

using acn::BigInt;
using Acn4Big = Acn<4, BigInt>;

TEST_CASE("format round-trips for BigInt beyond i64 range", "[format][bigint]") {
    // 10^30 doesn't fit in an i64; from_coeffs (not from_ints) is required
    // to construct it directly as BigInt.
    BigInt huge = BigInt(1);
    for (int i = 0; i < 30; ++i)
        huge = huge * BigInt(10);
    std::array<BigInt, 4> coeffs{huge, -huge, BigInt(0), BigInt(1)};
    auto a = Acn4Big::from_coeffs(std::span<const BigInt>(coeffs), 0);

    std::ostringstream out;
    out << a;
    auto parsed = acn::parse<4, BigInt>(out.str());
    REQUIRE(parsed.has_value());
    REQUIRE(*parsed == a);

    auto bytes = acn::serialize(a);
    auto restored = acn::deserialize<4, BigInt>(bytes);
    REQUIRE(restored.has_value());
    REQUIRE(*restored == a);
}

#endif
