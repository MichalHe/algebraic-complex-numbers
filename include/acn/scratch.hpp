#pragma once

// Thread-local reusable scratch buffers for AcnDyn's internal temporaries
// (PLAN.md §5.1/§5.3's "context-owned ScratchBuffer" motivation).
// AcnDyn's operations for n > kInlineN previously allocated a fresh
// std::vector<Int> on every call for canonicalize's 2n scratch, the
// scale-aligned copies in align_and_combine, and fma_into's product/
// accumulator copies -- exactly the per-term allocation cost DESIGN.md
// §5.4/§8 says the _into forms exist to avoid (flagged in PROGRESS.md
// across several sessions).
//
// This is a lighter-weight stand-in for PLAN.md's full per-NumberContext
// ScratchBuffer: a plain thread_local growable vector per named role
// rather than something owned by (and scoped to the lifetime of) a
// NumberContext. It gets the actual performance property PLAN.md is after
// (no repeated heap allocation once a thread's largest-seen n has been
// reached for a role) without threading a context through every
// operation. See PROGRESS.md for the fuller PLAN.md-vs-this-session
// comparison.
//
// A distinct Tag is required per role that can be *concurrently* live
// within one call (e.g. align_and_combine needs its `ca` and `cb` copies
// at the same time) -- this is deliberately not one shared buffer, since
// that would let two live borrows alias. Every call site here was
// checked by inspection to use its buffer only for the duration of one
// function call, with no reentrant call back into the same Tag while that
// borrow is still live, so a thread_local (rather than some kind of
// checked-out/returned handle) is sufficient.
//
// std::vector::resize only reallocates when the requested size exceeds
// current capacity, so this is amortized zero-allocation: after a thread
// has processed its largest-seen n for a given Tag, later calls at that
// size or smaller allocate nothing. scratch_capacity() exposes that
// backing capacity so tests can verify "no growth after warmup" directly,
// without a global operator new/delete interposer (which risks
// interacting badly with the sanitizer builds' own allocator hooks).

#include <cstddef>
#include <span>
#include <vector>

namespace acn::detail {

enum class ScratchTag {
    Canonicalize,
    Sqrt2Rotated,
    Sqrt2Scratch,
    AlignCa,
    AlignCb,
    FmaProduct,
    FmaAccumulator,
    Norm2,
};

template <class Int, ScratchTag Tag>
struct ScratchSlot {
    static std::vector<Int>& storage() {
        thread_local std::vector<Int> buf;
        return buf;
    }
};

template <class Int, ScratchTag Tag>
std::span<Int> scratch_span(std::size_t n) {
    auto& buf = ScratchSlot<Int, Tag>::storage();
    if (buf.size() < n)
        buf.resize(n);
    // Elements past index n may hold stale values from a previous, larger
    // call at this Tag; every call site here fully overwrites [0, n)
    // before reading it back, so that's never observed.
    return std::span<Int>(buf.data(), n);
}

template <class Int, ScratchTag Tag>
std::size_t scratch_capacity() {
    return ScratchSlot<Int, Tag>::storage().capacity();
}

}  // namespace acn::detail
