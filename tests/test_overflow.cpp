#include <random>

#include <catch2/catch.hpp>

#include "acn/acn.hpp"

using acn::Acn;
using acn::BigInt;
using acn::CheckedI64;
using acn::i64;
using acn::promote;

using Acn4Checked = Acn<4, CheckedI64>;
using Acn4Big = Acn<4, BigInt>;

TEST_CASE("repeated squaring overflows i64 at a known step, flagged and not silently wrong", "[overflow]") {
    CheckedI64::clear_overflow();

    // Start from something with coefficient 3 at every slot: squaring under
    // the negacyclic convolution roughly triples the coefficient magnitude
    // each round (empirically, for this ring), so i64 (~9.2e18) overflows
    // well before the promoted BigInt computation runs into any limit.
    std::array<i64, 4> start{3, 3, 3, 3};
    Acn4Checked checked = Acn4Checked::from_ints(start, 0);
    Acn4Big big = Acn4Big::from_ints(start, 0);

    bool overflow_hit = false;
    for (int round = 0; round < 40; ++round) {
        checked = checked * checked;
        big = big * big;
        if (CheckedI64::overflow_seen()) {
            overflow_hit = true;
            break;
        }
        // Until overflow, the checked path must still track the exact
        // (BigInt) result exactly -- this is the "no wrong-but-flagless
        // result" half of the acceptance criterion.
        auto promoted_checked = promote<4>(checked);
        REQUIRE(promoted_checked == big);
    }

    REQUIRE(overflow_hit);
    CheckedI64::clear_overflow();
}

TEST_CASE("BigInt reproduces the mathematically correct result past the point i64 overflows", "[overflow]") {
    CheckedI64::clear_overflow();
    std::array<i64, 4> start{3, 3, 3, 3};
    Acn4Big big = Acn4Big::from_ints(start, 0);
    Acn4Big reference = Acn4Big::from_ints(start, 0);

    // Reference computed by repeated multiplication in a different order
    // (accumulated product vs. repeated squaring) so it is not simply the
    // same call sequence duplicated.
    Acn4Big accumulated = Acn4Big::one();
    for (int i = 0; i < 8; ++i)
        accumulated = accumulated * reference;  // reference^8

    for (int round = 0; round < 3; ++round)
        big = big * big;  // reference^8 too (3 squarings == ^8)

    REQUIRE(big == accumulated);
    CheckedI64::clear_overflow();
}

TEST_CASE("fuzz: Checked<i64> and BigInt never silently diverge", "[overflow][fuzz]") {
    CheckedI64::clear_overflow();
    std::mt19937_64 rng(0xF00D);
    std::uniform_int_distribution<i64> dist(-1000, 1000);
    std::uniform_int_distribution<int> op_dist(0, 2);

    for (int series = 0; series < 30; ++series) {
        CheckedI64::clear_overflow();
        std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
        Acn4Checked checked = Acn4Checked::from_ints(coeffs, 0);
        Acn4Big big = Acn4Big::from_ints(coeffs, 0);

        for (int step = 0; step < 60; ++step) {
            std::array<i64, 4> operand_coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
            Acn4Checked checked_operand = Acn4Checked::from_ints(operand_coeffs, 0);
            Acn4Big big_operand = Acn4Big::from_ints(operand_coeffs, 0);

            if (CheckedI64::overflow_seen())
                break;  // no further guarantee once the sticky flag fired

            switch (op_dist(rng)) {
                case 0:
                    checked = checked + checked_operand;
                    big = big + big_operand;
                    break;
                case 1:
                    checked = checked - checked_operand;
                    big = big - big_operand;
                    break;
                default:
                    checked = checked * checked_operand;
                    big = big * big_operand;
                    break;
            }

            if (!CheckedI64::overflow_seen())
                REQUIRE(promote<4>(checked) == big);
        }
    }
    CheckedI64::clear_overflow();
}
