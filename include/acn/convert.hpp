#pragma once

// The two-tier promotion driver (DESIGN.md §3.4): run a computation with
// Acn<N, Checked<i64>>; if Checked<i64>::overflow_seen() ever fires, the
// caller restarts the affected computation with Acn<N, BigInt> built from
// promote(). try_narrow the other way once the caller believes it's safe to
// drop back to the cheap backend.
//
// This is the "one worked example, not a policy" PLAN.md §6/§5 asks for —
// callers own the restart decision.

#include <array>
#include <optional>
#include <vector>

#include "acn/acn_dynamic.hpp"
#include "acn/acn_static.hpp"
#include "acn/bigint.hpp"
#include "acn/int_backend.hpp"
#include "acn/stats.hpp"

namespace acn {

/// Widens a Checked<i64>-backed value to BigInt (e.g. after overflow).
template <std::size_t N>
Acn<N, BigInt> promote(const Acn<N, CheckedI64>& a) {
    ACN_STAT_INC(bignum_promotions);
    std::array<i64, N> coeffs{};
    auto src = a.coeffs();
    for (std::size_t j = 0; j < N; ++j)
        coeffs[j] = src[j].value();
    return Acn<N, BigInt>::from_ints(coeffs, a.scale_exp());
}

/// Narrows a BigInt-backed value to Checked<i64>; nullopt if it doesn't fit.
template <std::size_t N>
std::optional<Acn<N, CheckedI64>> try_narrow(const Acn<N, BigInt>& a) {
    std::array<i64, N> coeffs{};
    auto src = a.coeffs();
    for (std::size_t j = 0; j < N; ++j) {
        if (!src[j].fits_i64())
            return std::nullopt;
        coeffs[j] = src[j].to_i64_unchecked();
    }
    return Acn<N, CheckedI64>::from_ints(coeffs, a.scale_exp());
}

/// Widens an AcnDyn<CheckedI64> to AcnDyn<BigInt>.
inline AcnDyn<BigInt> promote(const AcnDyn<CheckedI64>& a) {
    ACN_STAT_INC(bignum_promotions);
    std::vector<i64> coeffs(a.precision());
    auto src = a.coeffs();
    for (std::size_t j = 0; j < coeffs.size(); ++j)
        coeffs[j] = src[j].value();
    return AcnDyn<BigInt>::from_ints(a.precision(), coeffs, a.scale_exp());
}

/// Narrows an AcnDyn<BigInt> to AcnDyn<CheckedI64>; nullopt if it doesn't fit.
inline std::optional<AcnDyn<CheckedI64>> try_narrow(const AcnDyn<BigInt>& a) {
    std::vector<i64> coeffs(a.precision());
    auto src = a.coeffs();
    for (std::size_t j = 0; j < coeffs.size(); ++j) {
        if (!src[j].fits_i64())
            return std::nullopt;
        coeffs[j] = src[j].to_i64_unchecked();
    }
    return AcnDyn<CheckedI64>::from_ints(a.precision(), coeffs, a.scale_exp());
}

}  // namespace acn
