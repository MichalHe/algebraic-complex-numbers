#pragma once

// The canonical-form invariant (DESIGN.md §4): a value (c, k) denoting
// u / sqrt(2)^k is canonical iff either all coefficients are zero and k==0,
// or sqrt(2) does not divide the numerator u. Canonicalizing never clamps k
// at zero; k may go negative.

#include <span>

#include "acn/config.hpp"
#include "acn/ring.hpp"
#include "acn/stats.hpp"
#include "acn/types.hpp"

namespace acn {

template <class Int>
bool all_zero(std::span<const Int> c) {
    for (const auto& value : c)
        if (!value.is_zero())
            return false;
    return true;
}

template <class Int>
bool all_even(std::span<const Int> c) {
    for (const auto& value : c)
        if (!value.is_even())
            return false;
    return true;
}

/// Reduces (c, k) to the unique canonical form in place. scratch must
/// have size 2*c.size() and must not alias c.
// canonicalize(c, k): in place. `scratch` must have size 2 * c.size() and
// must not alias c.
template <class Int>
void canonicalize(std::span<Int> c, ScaleExp& k, std::span<Int> scratch) {
    const std::size_t n = c.size();
    ACN_ASSERT(scratch.size() == 2 * n);
    ACN_ASSERT(scratch.data() != c.data());
    ACN_STAT_INC(canonicalize_calls);

    if (all_zero<Int>(c)) {
        k = 0;
        return;
    }

    // Bulk even reduction: divide out every shared factor of two at once,
    // rather than one sqrt(2) step at a time.
    int shared_ctz = -1;
    for (const auto& value : c) {
        if (value.is_zero())
            continue;
        int t = value.ctz();
        if (shared_ctz < 0 || t < shared_ctz)
            shared_ctz = t;
    }
    if (shared_ctz > 0) {
        for (auto& value : c)
            if (!value.is_zero())
                for (int i = 0; i < shared_ctz; ++i)
                    value = value.halve_exact();
        k = static_cast<ScaleExp>(k - 2 * shared_ctz);
    }

    if (n < 4)
        return;  // no omega^(n/4); the bulk-even path above is all there is.

    std::span<Int> candidate = scratch.subspan(0, n);
    std::span<Int> sqrt2_scratch = scratch.subspan(n, n);

    for (;;) {
        ring::mul_by_sqrt2<Int>(candidate, std::span<const Int>(c), sqrt2_scratch);
        if (!all_even<Int>(candidate))
            break;
        for (std::size_t j = 0; j < n; ++j)
            c[j] = candidate[j].halve_exact();
        k = static_cast<ScaleExp>(k - 1);
        ACN_STAT_INC(canonicalize_iterations);

        // Bulk-reduce again: the odd step can uncover further shared
        // factors of two.
        shared_ctz = -1;
        for (const auto& value : c) {
            if (value.is_zero())
                continue;
            int t = value.ctz();
            if (shared_ctz < 0 || t < shared_ctz)
                shared_ctz = t;
        }
        if (shared_ctz > 0) {
            for (auto& value : c)
                if (!value.is_zero())
                    for (int i = 0; i < shared_ctz; ++i)
                        value = value.halve_exact();
            k = static_cast<ScaleExp>(k - 2 * shared_ctz);
        }
    }
}

/// True iff (c, k) is already in canonical form.
template <class Int>
bool is_canonical(std::span<const Int> c, ScaleExp k, std::span<Int> scratch) {
    const std::size_t n = c.size();
    if (all_zero<Int>(c))
        return k == 0;
    if (n < 4)
        return !all_even<Int>(c);
    std::span<Int> candidate = scratch.subspan(0, n);
    std::span<Int> sqrt2_scratch = scratch.subspan(n, n);
    ring::mul_by_sqrt2<Int>(candidate, c, sqrt2_scratch);
    return !all_even<Int>(candidate);
}

}  // namespace acn
