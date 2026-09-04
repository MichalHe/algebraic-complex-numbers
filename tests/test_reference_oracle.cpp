// Differential test between the real library and tests/reference's
// independent oracle (PLAN.md §9.5). See reference.hpp for why this is a
// meaningfully different check than the numeric (long double / interval)
// cross-checks used elsewhere: those catch a wrong *value*; this catches
// the specific failure mode PLAN.md calls out (ring.hpp/canonical.hpp
// having a shared bug that "independent" test code accidentally
// reproduces) by using genuinely separately-written arithmetic.

#include <random>

#include <catch2/catch.hpp>

#include "acn/acn.hpp"
#include "reference.hpp"

using acn::Acn;
using acn::BigInt;
using acn::CheckedI64;
using acn::i64;
using acn::reference::canonicalize_ref;
using acn::reference::ref_add;
using acn::reference::ref_conj;
using acn::reference::ref_mul;
using acn::reference::ref_neg;
using acn::reference::ref_sub;
using acn::reference::RefValue;

using Acn4 = Acn<4, CheckedI64>;

namespace {

void expect_matches(const RefValue& ref, const Acn4& value) {
    auto [canon_coeffs, canon_k] = canonicalize_ref(ref);
    REQUIRE(canon_k == value.scale_exp());
    auto vc = value.coeffs();
    REQUIRE(canon_coeffs.size() == vc.size());
    for (std::size_t j = 0; j < vc.size(); ++j)
        REQUIRE(canon_coeffs[j] == BigInt(vc[j].value()));
}

Acn4 random_acn(std::mt19937_64& rng, i64 bound = 5) {
    std::uniform_int_distribution<i64> dist(-bound, bound);
    std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
    return Acn4::from_ints(coeffs, 0);
}

}  // namespace

TEST_CASE("reference oracle agrees with Acn<4,CheckedI64> on individual operations", "[reference]") {
    std::mt19937_64 rng(501);
    for (int trial = 0; trial < 200; ++trial) {
        auto ints_a = random_acn(rng);
        auto ints_b = random_acn(rng);
        std::array<i64, 4> a_raw{ints_a.coeffs()[0].value(), ints_a.coeffs()[1].value(),
                                  ints_a.coeffs()[2].value(), ints_a.coeffs()[3].value()};
        std::array<i64, 4> b_raw{ints_b.coeffs()[0].value(), ints_b.coeffs()[1].value(),
                                  ints_b.coeffs()[2].value(), ints_b.coeffs()[3].value()};
        // Rebuild both operands from the *same* raw ints at k=0 (not the
        // already-canonicalized Acn4 above) so the oracle and the real
        // value start from an identical, uncanonicalized representation.
        RefValue ref_a = RefValue::from_ints(4, a_raw, 0);
        RefValue ref_b = RefValue::from_ints(4, b_raw, 0);
        Acn4 a = Acn4::from_ints(a_raw, 0);
        Acn4 b = Acn4::from_ints(b_raw, 0);

        expect_matches(ref_add(ref_a, ref_b), a + b);
        expect_matches(ref_sub(ref_a, ref_b), a - b);
        expect_matches(ref_mul(ref_a, ref_b), a * b);
        expect_matches(ref_neg(ref_a), -a);
        expect_matches(ref_conj(ref_a), a.conj());
    }
}

TEST_CASE("reference oracle agrees with Acn<4,CheckedI64> across a chained random sequence", "[reference]") {
    std::mt19937_64 rng(502);
    auto start = random_acn(rng, 4);
    std::array<i64, 4> start_raw{start.coeffs()[0].value(), start.coeffs()[1].value(), start.coeffs()[2].value(),
                                  start.coeffs()[3].value()};

    Acn4 acc = Acn4::from_ints(start_raw, 0);
    RefValue ref_acc = RefValue::from_ints(4, start_raw, 0);

    std::uniform_int_distribution<int> op_dist(0, 4);
    for (int step = 0; step < 40; ++step) {
        int op = op_dist(rng);
        switch (op) {
            case 0: {  // add a small random term (keeps magnitude linear, not squared)
                auto term = random_acn(rng, 3);
                std::array<i64, 4> term_raw{term.coeffs()[0].value(), term.coeffs()[1].value(),
                                             term.coeffs()[2].value(), term.coeffs()[3].value()};
                acc = acc + Acn4::from_ints(term_raw, 0);
                ref_acc = ref_add(ref_acc, RefValue::from_ints(4, term_raw, 0));
                break;
            }
            case 1: {
                auto term = random_acn(rng, 3);
                std::array<i64, 4> term_raw{term.coeffs()[0].value(), term.coeffs()[1].value(),
                                             term.coeffs()[2].value(), term.coeffs()[3].value()};
                acc = acc - Acn4::from_ints(term_raw, 0);
                ref_acc = ref_sub(ref_acc, RefValue::from_ints(4, term_raw, 0));
                break;
            }
            case 2: {
                acc = -acc;
                ref_acc = ref_neg(ref_acc);
                break;
            }
            case 3: {
                acc = acc.conj();
                ref_acc = ref_conj(ref_acc);
                break;
            }
            case 4:
            default: {
                // Small fixed unit-ish multiplier to bound growth across
                // the 40-step chain (this is the only magnitude-squaring
                // op in the mix).
                std::array<i64, 4> unit_raw{1, 1, 0, 0};
                auto unit = Acn4::from_ints(unit_raw, 0);
                acc = acc * unit;
                ref_acc = ref_mul(ref_acc, RefValue::from_ints(4, unit_raw, 0));
                break;
            }
        }
        expect_matches(ref_acc, acc);
    }
    REQUIRE_FALSE(CheckedI64::overflow_seen());
}

TEST_CASE("reference oracle catches rescaling correctly (DESIGN.md's sqrt2 worked example)", "[reference]") {
    // sqrt(2) = omega^(n/4) - omega^(3n/4), built here the same raw,
    // uncanonicalized way Acn4::sqrt2() does internally (c[1]=1, c[3]=-1,
    // k=0) -- exactly the computation whose sign convention the prototype
    // got wrong (DESIGN.md §2.2/§4.2). Canonicalizing the oracle's version
    // independently must land on the same (c=(1,0,0,0), k=-1) the real
    // type produces.
    RefValue ref_sqrt2 = ref_sub(RefValue::monomial(4, 1), RefValue::monomial(4, 3));
    expect_matches(ref_sqrt2, Acn4::sqrt2());

    RefValue ref_two = ref_add(RefValue::monomial(4, 0), RefValue::monomial(4, 0));  // omega^0 + omega^0 = 2
    expect_matches(ref_two, Acn4::one() + Acn4::one());
}
