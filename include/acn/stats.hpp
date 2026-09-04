#pragma once

// Instrumentation counters (PLAN.md §8.3): operation counts,
// canonicalization iterations, mul_by_sqrt2 invocations, scale-alignment
// distance, widenings/narrowings, overflow flags seen, bignum promotions.
// Off by default (ACN_STATS=0, set in config.hpp); every ACN_STAT_* site
// compiles to nothing at all when off, not just "cheap" -- the macros
// themselves are `((void)0)`, so there's no counters() call, no branch, no
// thread_local access to elide.
//
// Deliberately does not track "arena bytes" from PLAN.md's list: there is
// no arena yet (see PROGRESS.md -- AcnDyn's own coefficient storage is
// still plain new/delete; only its *scratch* temporaries were moved to
// acn/scratch.hpp's thread-local reuse, which isn't itself an allocator
// with a byte-budget concept to report).

#include <cstddef>
#include <ostream>

#include "acn/config.hpp"
#include "acn/types.hpp"

namespace acn::stats {

struct Counters {
    u64 add_count = 0;
    u64 sub_count = 0;
    u64 mul_count = 0;
    u64 fma_count = 0;

    u64 canonicalize_calls = 0;
    // Total odd (mul_by_sqrt2) reduction steps taken across all
    // canonicalize() calls -- the number PLAN.md's canonicalization-cost
    // risk mitigation wants watched for regressions.
    u64 canonicalize_iterations = 0;
    u64 mul_by_sqrt2_calls = 0;

    u64 scale_alignment_calls = 0;
    // Sum, over every add/sub/fma scale-alignment, of how far the smaller
    // operand's k had to be raised (DESIGN.md's align_scales distance).
    u64 scale_alignment_total_distance = 0;

    u64 widen_calls = 0;
    u64 narrow_calls = 0;

    u64 overflow_flags_seen = 0;
    u64 bignum_promotions = 0;

    void reset() { *this = Counters{}; }
};

#if ACN_STATS

inline Counters& counters() {
    thread_local Counters c;
    return c;
}

#define ACN_STAT_INC(field) (::acn::stats::counters().field += 1)
#define ACN_STAT_ADD(field, amount) (::acn::stats::counters().field += static_cast<::acn::u64>(amount))

#else

// `counters()` still exists when ACN_STATS is off, returning a value that
// is always zero and never written to by any instrumentation site (every
// ACN_STAT_* call compiles to nothing), so code that unconditionally reads
// stats::counters() for reporting purposes still links either way.
inline Counters& counters() {
    static Counters c;
    return c;
}

#define ACN_STAT_INC(field) ((void)0)
#define ACN_STAT_ADD(field, amount) ((void)0)

#endif

inline std::ostream& operator<<(std::ostream& os, const Counters& c) {
    os << "acn::stats::Counters{"
       << "add=" << c.add_count << ", sub=" << c.sub_count << ", mul=" << c.mul_count
       << ", fma=" << c.fma_count << ", canonicalize_calls=" << c.canonicalize_calls
       << ", canonicalize_iterations=" << c.canonicalize_iterations
       << ", mul_by_sqrt2_calls=" << c.mul_by_sqrt2_calls
       << ", scale_alignment_calls=" << c.scale_alignment_calls
       << ", scale_alignment_total_distance=" << c.scale_alignment_total_distance
       << ", widen_calls=" << c.widen_calls << ", narrow_calls=" << c.narrow_calls
       << ", overflow_flags_seen=" << c.overflow_flags_seen << ", bignum_promotions=" << c.bignum_promotions
       << "}";
    return os;
}

}  // namespace acn::stats
