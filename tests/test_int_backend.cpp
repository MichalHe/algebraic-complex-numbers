#include <catch2/catch.hpp>

#include "acn/int_backend.hpp"

using acn::CheckedI64;
using acn::i64;

TEST_CASE("Checked<i64> basic arithmetic", "[int_backend]") {
    CheckedI64::clear_overflow();
    CheckedI64 a(3), b(5);
    REQUIRE((a + b).value() == 8);
    REQUIRE((a - b).value() == -2);
    REQUIRE((a * b).value() == 15);
    REQUIRE((-a).value() == -3);
    REQUIRE_FALSE(CheckedI64::overflow_seen());
}

TEST_CASE("Checked<i64> overflow is flagged, not UB", "[int_backend]") {
    CheckedI64::clear_overflow();
    CheckedI64 huge(std::numeric_limits<i64>::max());
    auto result = huge + CheckedI64(1);
    REQUIRE(CheckedI64::overflow_seen());
    (void)result;
    CheckedI64::clear_overflow();
}

TEST_CASE("Checked<i64> parity and halving", "[int_backend]") {
    REQUIRE(CheckedI64(4).is_even());
    REQUIRE_FALSE(CheckedI64(3).is_even());
    REQUIRE(CheckedI64(6).halve_exact().value() == 3);
    REQUIRE(CheckedI64(-6).halve_exact().value() == -3);
}

TEST_CASE("Checked<i64> ctz agrees for x and -x", "[int_backend]") {
    for (i64 x : {1, 2, 4, 6, 8, 12, 96, 1024}) {
        REQUIRE(CheckedI64(x).ctz() == CheckedI64(-x).ctz());
    }
    REQUIRE(CheckedI64(12).ctz() == 2);
    REQUIRE(CheckedI64(1).ctz() == 0);
}

TEST_CASE("Checked<i64> shift_left matches multiplication by a power of two", "[int_backend]") {
    CheckedI64::clear_overflow();
    REQUIRE(CheckedI64(3).shift_left(4).value() == 48);
    REQUIRE_FALSE(CheckedI64::overflow_seen());
}

TEST_CASE("Checked<i64> ordering and equality", "[int_backend]") {
    REQUIRE(CheckedI64(3) == CheckedI64(3));
    REQUIRE(CheckedI64(3) != CheckedI64(4));
    REQUIRE(CheckedI64(3) < CheckedI64(4));
}
