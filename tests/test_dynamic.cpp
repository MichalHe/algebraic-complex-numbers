#include <array>
#include <random>
#include <vector>

#include <catch2/catch.hpp>

#include "acn/acn.hpp"
#include "acn/scratch.hpp"

using acn::Acn;
using acn::AcnDyn;
using acn::CheckedI64;
using acn::i64;
using acn::NumberContext;

namespace {

template <std::size_t N>
void run_differential(std::mt19937_64& rng) {
    using Static = Acn<N, CheckedI64>;
    using Dyn = AcnDyn<CheckedI64>;

    std::uniform_int_distribution<i64> dist(-6, 6);
    std::uniform_int_distribution<int> op_dist(0, N >= 4 ? 6 : 4);

    auto random_value = [&]() {
        std::array<i64, N> coeffs{};
        for (auto& v : coeffs)
            v = dist(rng);
        Static s = Static::from_ints(coeffs, 0);
        Dyn d = Dyn::from_ints(static_cast<acn::u32>(N), std::span<const i64>(coeffs), 0);
        return std::pair{s, d};
    };

    auto require_equal = [](const Static& s, const Dyn& d) {
        REQUIRE(s.scale_exp() == d.scale_exp());
        auto sc = s.coeffs();
        auto dc = d.coeffs();
        REQUIRE(sc.size() == dc.size());
        for (std::size_t i = 0; i < sc.size(); ++i)
            REQUIRE(sc[i].value() == dc[i].value());
    };

    auto [as, ad] = random_value();
    require_equal(as, ad);

    for (int step = 0; step < 200; ++step) {
        auto [bs, bd] = random_value();
        require_equal(bs, bd);

        int op = op_dist(rng);
        switch (op) {
            case 0:
                as = as + bs;
                ad = ad + bd;
                break;
            case 1:
                as = as - bs;
                ad = ad - bd;
                break;
            case 2:
                as = as * bs;
                ad = ad * bd;
                break;
            case 3:
                as = -as;
                ad = -ad;
                break;
            case 4:
                as = as.conj();
                ad = ad.conj();
                break;
            case 5:
                as = as.mul_omega(3);
                ad = ad.mul_omega(3);
                break;
            case 6:
                as = as.scale(1);
                ad = ad.scale(1);
                break;
            default:
                break;
        }
        require_equal(as, ad);
    }
}

}  // namespace

TEST_CASE("AcnDyn is bit-identical to Acn<N> across precisions", "[dynamic][differential]") {
    // Long chains of multiplication legitimately overflow i64 (values grow
    // roughly exponentially); this test's contract is that Acn<N> and
    // AcnDyn overflow identically and stay bit-identical up to that point,
    // not that they never overflow.
    CheckedI64::clear_overflow();
    std::mt19937_64 rng(0xD17FEED);
    run_differential<1>(rng);
    run_differential<2>(rng);
    run_differential<4>(rng);
    run_differential<8>(rng);
    run_differential<16>(rng);
    run_differential<32>(rng);
    CheckedI64::clear_overflow();
}

TEST_CASE("AcnDyn copy/move/self-assign semantics", "[dynamic]") {
    auto a = AcnDyn<CheckedI64>::from_ints(4, std::array<i64, 4>{1, 2, 3, 4}, 0);

    // Copy.
    AcnDyn<CheckedI64> b = a;
    REQUIRE(b == a);
    // Self-assign via an alias, so the compiler can't flag the source line
    // as an obvious self-assignment; this is exactly what operator=(const&)
    // must guard against.
    AcnDyn<CheckedI64>& b_ref = b;
    b = b_ref;
    REQUIRE(b == a);

    // Move.
    AcnDyn<CheckedI64> c = std::move(b);
    REQUIRE(c == a);

    AcnDyn<CheckedI64> d = AcnDyn<CheckedI64>::zero(4);
    d = std::move(c);
    REQUIRE(d == a);

    // Move a value into itself (via an alias, so the compiler can't turn
    // this into a no-op self-move warning) must not corrupt state.
    AcnDyn<CheckedI64>* self_ptr = &d;
    AcnDyn<CheckedI64>& self_ref = *self_ptr;
    d = std::move(self_ref);
    REQUIRE(d == a);
}

TEST_CASE("AcnDyn small-buffer-optimization boundary: n=8 inline, n=16 heap", "[dynamic]") {
    auto inline_value = AcnDyn<CheckedI64>::one(8);
    auto heap_value = AcnDyn<CheckedI64>::one(16);
    REQUIRE(inline_value.precision() == 8);
    REQUIRE(heap_value.precision() == 16);

    // one() + one() == 2, which canonicalizes to numerator 1 at k=-2 (the
    // same "2 = (1,0,...), k=-2" shape as the static-type canonical test).
    auto inline_sum = inline_value + inline_value;
    auto heap_sum = heap_value + heap_value;
    REQUIRE(inline_sum.coeffs()[0].value() == 1);
    REQUIRE(inline_sum.scale_exp() == -2);
    REQUIRE(heap_sum.coeffs()[0].value() == 1);
    REQUIRE(heap_sum.scale_exp() == -2);
    REQUIRE(inline_sum.is_integer());
    REQUIRE(heap_sum.is_integer());

    // Copies at both sizes round-trip correctly.
    auto inline_copy = inline_value;
    auto heap_copy = heap_value;
    REQUIRE(inline_copy == inline_value);
    REQUIRE(heap_copy == heap_value);
}

TEST_CASE("NumberContext factories agree with AcnDyn free factories", "[dynamic]") {
    NumberContext<CheckedI64> ctx(4);
    REQUIRE(ctx.zero() == AcnDyn<CheckedI64>::zero(4));
    REQUIRE(ctx.one() == AcnDyn<CheckedI64>::one(4));
    REQUIRE(ctx.sqrt2() == AcnDyn<CheckedI64>::sqrt2(4));
    REQUIRE((ctx.inv_sqrt2() * ctx.sqrt2()) == ctx.one());
}

TEST_CASE("AcnDyn fma_into matches naive accumulation", "[dynamic]") {
    std::mt19937_64 rng(77);
    std::uniform_int_distribution<i64> dist(-4, 4);
    auto random_value = [&]() {
        std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
        return AcnDyn<CheckedI64>::from_ints(4, coeffs, 0);
    };

    AcnDyn<CheckedI64> acc_naive = AcnDyn<CheckedI64>::zero(4);
    AcnDyn<CheckedI64> acc_fma = AcnDyn<CheckedI64>::zero(4);
    for (int i = 0; i < 20; ++i) {
        auto a = random_value();
        auto b = random_value();
        acc_naive = acc_naive + (a * b);
        fma_into(acc_fma, a, b);
    }
    REQUIRE(acc_naive == acc_fma);
}

TEST_CASE("AcnDyn fma_into is correct when d aliases a or b", "[dynamic]") {
    std::mt19937_64 rng(78);
    std::uniform_int_distribution<i64> dist(-4, 4);
    auto random_value = [&]() {
        std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
        return AcnDyn<CheckedI64>::from_ints(4, coeffs, 0);
    };

    for (int trial = 0; trial < 50; ++trial) {
        auto x = random_value();
        auto y = random_value();

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

TEST_CASE("AcnDyn scratch buffers reach a stable capacity (amortized zero allocation)",
          "[dynamic][scratch]") {
    using acn::detail::scratch_capacity;
    using acn::detail::ScratchTag;

    // n = 32 is well above kInlineN (8), so every operation below goes
    // through the heap/scratch path this test is checking.
    constexpr acn::u32 n = 32;
    std::mt19937_64 rng(88);
    std::uniform_int_distribution<i64> dist(-6, 6);
    auto random_value = [&]() {
        std::array<i64, n> coeffs{};
        for (auto& v : coeffs)
            v = dist(rng);
        return AcnDyn<CheckedI64>::from_ints(n, std::span<const i64>(coeffs), 0);
    };

    auto run_workload = [&](int iterations) {
        AcnDyn<CheckedI64> acc = AcnDyn<CheckedI64>::zero(n);
        for (int i = 0; i < iterations; ++i) {
            auto a = random_value();
            auto b = random_value();
            acc = acc + a - b;
            acc = acc * a;
            fma_into(acc, a, b);
            auto normed = acc.norm2();
            (void)normed;
            // sqrt2(n) has an odd canonical k (-1), so adding it forces
            // align_and_combine's odd-delta branch -- and therefore the
            // Sqrt2Rotated/Sqrt2Scratch slots -- to actually run at least
            // once per iteration, not just the even (shift_left-only) path.
            acc = acc + AcnDyn<CheckedI64>::sqrt2(n);
        }
        return acc;
    };

    // Warm up every scratch Tag this workload touches to its steady-state
    // size, then record capacities.
    auto acc = run_workload(5);

    std::size_t cap_canon = scratch_capacity<CheckedI64, ScratchTag::Canonicalize>();
    std::size_t cap_ca = scratch_capacity<CheckedI64, ScratchTag::AlignCa>();
    std::size_t cap_cb = scratch_capacity<CheckedI64, ScratchTag::AlignCb>();
    std::size_t cap_rot = scratch_capacity<CheckedI64, ScratchTag::Sqrt2Rotated>();
    std::size_t cap_s2s = scratch_capacity<CheckedI64, ScratchTag::Sqrt2Scratch>();
    std::size_t cap_fp = scratch_capacity<CheckedI64, ScratchTag::FmaProduct>();
    std::size_t cap_fa = scratch_capacity<CheckedI64, ScratchTag::FmaAccumulator>();
    std::size_t cap_n2 = scratch_capacity<CheckedI64, ScratchTag::Norm2>();

    REQUIRE(cap_canon >= 2 * n);
    REQUIRE(cap_ca >= n);
    REQUIRE(cap_cb >= n);
    REQUIRE(cap_rot >= n);
    REQUIRE(cap_s2s >= n);
    REQUIRE(cap_fp >= n);
    REQUIRE(cap_fa >= n);
    REQUIRE(cap_n2 >= n);

    // Run substantially more of the same workload (same n, so no slot
    // should ever need to grow again) and confirm every capacity is
    // unchanged -- the amortized-zero-allocation property.
    (void)run_workload(500);
    (void)acc;

    REQUIRE(scratch_capacity<CheckedI64, ScratchTag::Canonicalize>() == cap_canon);
    REQUIRE(scratch_capacity<CheckedI64, ScratchTag::AlignCa>() == cap_ca);
    REQUIRE(scratch_capacity<CheckedI64, ScratchTag::AlignCb>() == cap_cb);
    REQUIRE(scratch_capacity<CheckedI64, ScratchTag::Sqrt2Rotated>() == cap_rot);
    REQUIRE(scratch_capacity<CheckedI64, ScratchTag::Sqrt2Scratch>() == cap_s2s);
    REQUIRE(scratch_capacity<CheckedI64, ScratchTag::FmaProduct>() == cap_fp);
    REQUIRE(scratch_capacity<CheckedI64, ScratchTag::FmaAccumulator>() == cap_fa);
    REQUIRE(scratch_capacity<CheckedI64, ScratchTag::Norm2>() == cap_n2);
}
