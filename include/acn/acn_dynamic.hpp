#pragma once

// AcnDyn<Int>: same semantics as Acn<N, Int> (see acn_static.hpp) but with
// the precision n chosen at construction rather than at compile time.
// n <= kInlineN coefficients live inline in the object (no allocation);
// larger n falls back to a heap array owned by the value.
//
// DESIGN.md §11 open question 1 is answered here the way PLAN.md §5.1
// commits to: NumberContext is a thin factory that fixes n for the values
// it creates, so ordinary construction/copy/assignment of AcnDyn itself
// does not need a context. (PLAN.md §5.1/§5.3's full bump-allocator Arena
// for a value's *own* coefficient storage is still deferred — see
// PROGRESS.md. Internal *scratch* temporaries -- canonicalize's 2n
// buffer, the scale-aligned copies add/sub/fma need, fma_into's product
// and accumulator copies -- now come from acn/scratch.hpp's thread-local
// reusable slots instead of a fresh std::vector<Int> per call, which is
// the concrete performance property PLAN.md's ScratchBuffer was after.)

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <complex>
#include <cstddef>
#include <new>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

#include "acn/canonical.hpp"
#include "acn/config.hpp"
#include "acn/ring.hpp"
#include "acn/scratch.hpp"
#include "acn/stats.hpp"
#include "acn/types.hpp"

namespace acn {

/// Runtime-precision element of Z[x]/(x^n+1)[1/sqrt2]; same semantics as
/// Acn<N,Int> with n chosen at construction. n <= kInlineN is inline
/// (no allocation).
template <class Int>
class AcnDyn {
public:
    static constexpr u32 kInlineN = 8;
    using int_type = Int;

private:
    alignas(Int) unsigned char inline_storage_[sizeof(Int) * kInlineN];
    Int* c_ = nullptr;
    u32 n_ = 0;
    ScaleExp k_ = 0;

    Int* inline_ptr() { return reinterpret_cast<Int*>(inline_storage_); }
    const Int* inline_ptr() const { return reinterpret_cast<const Int*>(inline_storage_); }
    bool is_inline() const { return c_ == inline_ptr(); }

    // The inline buffer is raw bytes, not an array of live Int objects, so
    // (unlike the heap path, where new Int[n]/delete[] manage lifetime)
    // inline elements must be placement-constructed/destroyed explicitly.
    // This matters for non-trivial backends like BigInt: assigning through
    // an Int* that was never constructed invokes operator= on an mpz_t with
    // garbage limb pointers, which segfaults.
    void release() {
        if (!c_)
            return;
        if (is_inline()) {
            for (u32 i = 0; i < n_; ++i)
                c_[i].~Int();
        } else {
            delete[] c_;
        }
        c_ = nullptr;
    }

    void allocate_zeroed(u32 n) {
        n_ = n;
        if (n <= kInlineN) {
            c_ = inline_ptr();
            for (u32 i = 0; i < n; ++i)
                ::new (static_cast<void*>(&c_[i])) Int();
        } else {
            c_ = new Int[n];  // Int's own default constructor already zeros.
        }
    }

    void copy_from(const AcnDyn& other) {
        allocate_zeroed(other.n_);
        for (u32 i = 0; i < n_; ++i)
            c_[i] = other.c_[i];
        k_ = other.k_;
    }

    std::span<Int> coeff_span() { return std::span<Int>(c_, n_); }
    std::span<const Int> coeff_span() const { return std::span<const Int>(c_, n_); }

    void canonicalize_self() {
        auto scratch =
            detail::scratch_span<Int, detail::ScratchTag::Canonicalize>(2 * static_cast<std::size_t>(n_));
        acn::canonicalize<Int>(coeff_span(), k_, scratch);
    }

public:
    AcnDyn() { allocate_zeroed(1); }

    explicit AcnDyn(u32 n) { allocate_zeroed(n == 0 ? 1 : n); }

    AcnDyn(const AcnDyn& other) { copy_from(other); }

    AcnDyn(AcnDyn&& other) noexcept {
        if (other.is_inline()) {
            copy_from(other);
        } else {
            c_ = other.c_;
            n_ = other.n_;
            k_ = other.k_;
            other.c_ = nullptr;
            other.n_ = 0;
            other.allocate_zeroed(1);
        }
    }

    AcnDyn& operator=(const AcnDyn& other) {
        if (this == &other)
            return *this;
        release();
        copy_from(other);
        return *this;
    }

    AcnDyn& operator=(AcnDyn&& other) noexcept {
        if (this == &other)
            return *this;
        release();
        if (other.is_inline()) {
            copy_from(other);
        } else {
            c_ = other.c_;
            n_ = other.n_;
            k_ = other.k_;
            other.c_ = nullptr;
            other.n_ = 0;
            other.allocate_zeroed(1);
        }
        return *this;
    }

    ~AcnDyn() { release(); }

    u32 precision() const { return n_; }
    std::span<const Int> coeffs() const { return coeff_span(); }
    ScaleExp scale_exp() const { return k_; }

    /// Additive identity at precision n.
    static AcnDyn zero(u32 n) { return AcnDyn(n); }

    /// Multiplicative identity at precision n.
    static AcnDyn one(u32 n) {
        AcnDyn result(n);
        result.c_[0] = Int(1);
        return result;
    }

    /// i = omega^(n/2).
    static AcnDyn imag(u32 n) {
        ACN_ASSERT(n >= 2);
        AcnDyn result(n);
        result.c_[n / 2] = Int(1);
        return result;
    }

    /// sqrt(2) = omega^(n/4) - omega^(3n/4).
    static AcnDyn sqrt2(u32 n) {
        ACN_ASSERT(n >= 4);
        AcnDyn result(n);
        result.c_[n / 4] = Int(1);
        result.c_[3 * n / 4] = Int(-1);
        result.canonicalize_self();
        return result;
    }

    /// 1/sqrt(2).
    static AcnDyn inv_sqrt2(u32 n) {
        ACN_ASSERT(n >= 4);
        AcnDyn result(n);
        result.c_[0] = Int(1);
        result.k_ = 1;
        return result;
    }

    /// omega^j, j taken mod 2n.
    static AcnDyn omega(u32 n, long long j) {
        AcnDyn basis = one(n);
        AcnDyn result(n);
        ring::rotate<Int>(result.coeff_span(), basis.coeff_span(), j);
        return result;
    }

    /// Builds sum(coeffs[j]*omega^j) / sqrt2^k from machine-word coefficients.
    static AcnDyn from_ints(u32 n, std::span<const i64> coeffs, ScaleExp k) {
        ACN_ASSERT(coeffs.size() == n);
        AcnDyn result(n);
        for (u32 i = 0; i < n; ++i)
            result.c_[i] = Int(coeffs[i]);
        result.k_ = k;
        result.canonicalize_self();
        return result;
    }

    /// Like from_ints(), but takes already-Int-typed coefficients.
    // Like from_ints, but takes already-Int-typed coefficients (needed when
    // Int is BigInt and a coefficient may not fit in i64).
    static AcnDyn from_coeffs(u32 n, std::span<const Int> coeffs, ScaleExp k) {
        ACN_ASSERT(coeffs.size() == n);
        AcnDyn result(n);
        for (u32 i = 0; i < n; ++i)
            result.c_[i] = coeffs[i];
        result.k_ = k;
        result.canonicalize_self();
        return result;
    }

    // Same as above with n taken from coeffs.size(), for generic code
    // (divide.hpp) that's written once against both Acn<N,Int> and AcnDyn.
    static AcnDyn from_coeffs(std::span<const Int> coeffs, ScaleExp k) {
        return from_coeffs(static_cast<u32>(coeffs.size()), coeffs, k);
    }

    /// True iff this value is exactly zero.
    bool is_zero() const { return all_zero<Int>(coeff_span()); }

    /// True iff this value is exactly one.
    bool is_one() const {
        if (k_ != 0 || !(c_[0] == Int(1)))
            return false;
        for (u32 j = 1; j < n_; ++j)
            if (!c_[j].is_zero())
                return false;
        return true;
    }

    /// True iff this value is a rational integer.
    // See Acn<N,Int>::is_integer for why k==0 alone is not the right test.
    bool is_integer() const {
        if (is_zero())
            return true;
        if (k_ > 0)
            return false;
        std::vector<Int> scaled(c_, c_ + n_);
        if (k_ < 0)
            apply_sqrt2_power(scaled, -static_cast<long long>(k_));
        for (u32 j = 1; j < n_; ++j)
            if (!scaled[j].is_zero())
                return false;
        return true;
    }

    /// True iff this value is a (dyadic) rational number.
    // See Acn<N,Int>::is_rational for the reasoning.
    bool is_rational() const {
        if (is_zero())
            return true;
        if (k_ % 2 != 0)
            return false;
        for (u32 j = 1; j < n_; ++j)
            if (!c_[j].is_zero())
                return false;
        return true;
    }

    /// True iff this value is a unit (+-omega^j * sqrt(2)^m).
    // See Acn<N,Int>::is_unit for the reasoning.
    bool is_unit() const {
        bool seen_nonzero = false;
        for (u32 j = 0; j < n_; ++j) {
            if (c_[j].is_zero())
                continue;
            if (seen_nonzero)
                return false;
            if (!(c_[j] == Int(1) || c_[j] == Int(-1)))
                return false;
            seen_nonzero = true;
        }
        return seen_nonzero;
    }

    friend bool operator==(const AcnDyn& a, const AcnDyn& b) {
        ACN_ASSERT(a.n_ == b.n_);
        if (a.k_ != b.k_)
            return false;
        for (u32 j = 0; j < a.n_; ++j)
            if (!(a.c_[j] == b.c_[j]))
                return false;
        return true;
    }

    friend std::strong_ordering operator<=>(const AcnDyn& a, const AcnDyn& b) {
        ACN_ASSERT(a.n_ == b.n_);
        if (auto cmp = a.k_ <=> b.k_; cmp != 0)
            return cmp;
        for (u32 j = 0; j < a.n_; ++j)
            if (auto cmp = a.c_[j] <=> b.c_[j]; cmp != 0)
                return cmp;
        return std::strong_ordering::equal;
    }

    AcnDyn operator-() const {
        AcnDyn result(n_);
        ring::neg<Int>(result.coeff_span(), coeff_span());
        result.k_ = k_;
        return result;
    }

    /// Complex conjugate.
    AcnDyn conj() const {
        AcnDyn result(n_);
        ring::conj<Int>(result.coeff_span(), coeff_span());
        result.k_ = k_;
        return result;
    }

    /// Multiplies by omega^j (index/sign work only, exact, no canonicalize).
    AcnDyn mul_omega(long long j) const {
        AcnDyn result(n_);
        ring::rotate<Int>(result.coeff_span(), coeff_span(), j);
        result.k_ = k_;
        return result;
    }

    /// Multiplies by sqrt(2)^(-d); d may be negative.
    AcnDyn scale(ScaleExp d) const {
        AcnDyn result = *this;
        result.k_ = static_cast<ScaleExp>(result.k_ - d);
        result.canonicalize_self();
        return result;
    }

    AcnDyn operator+(const AcnDyn& other) const {
        AcnDyn result(n_);
        add_into(result, *this, other);
        return result;
    }

    AcnDyn operator-(const AcnDyn& other) const {
        AcnDyn result(n_);
        sub_into(result, *this, other);
        return result;
    }

    AcnDyn operator*(const AcnDyn& other) const {
        AcnDyn result(n_);
        mul_into(result, *this, other);
        return result;
    }

    AcnDyn& operator+=(const AcnDyn& other) { return *this = *this + other; }
    AcnDyn& operator-=(const AcnDyn& other) { return *this = *this - other; }
    AcnDyn& operator*=(const AcnDyn& other) { return *this = *this * other; }

    /// this * conj(this); fixed by conjugation (real-valued).
    AcnDyn norm2() const {
        AcnDyn result(n_);
        auto scratch = detail::scratch_span<Int, detail::ScratchTag::Norm2>(n_);
        ring::norm2<Int>(result.coeff_span(), coeff_span(), scratch);
        result.k_ = static_cast<ScaleExp>(2 * k_);
        result.canonicalize_self();
        return result;
    }

    /// Heuristic (rounded) double-precision approximation; not sound --
    /// use to_interval() for soundness-critical comparisons.
    std::complex<double> to_complex() const {
        std::complex<double> sum(0.0, 0.0);
        for (u32 j = 0; j < n_; ++j) {
            double angle = std::numbers::pi * static_cast<double>(j) / static_cast<double>(n_);
            sum += c_[j].to_double() * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        double factor = std::pow(1.0 / std::numbers::sqrt2, static_cast<double>(k_));
        return sum * factor;
    }

    /// d = a + b, allocation-free (n <= kInlineN). Performance API.
    friend void add_into(AcnDyn& d, const AcnDyn& a, const AcnDyn& b) {
        ACN_STAT_INC(add_count);
        align_and_combine(d, a, b, /*subtract=*/false);
    }

    /// d = a - b, allocation-free (n <= kInlineN).
    friend void sub_into(AcnDyn& d, const AcnDyn& a, const AcnDyn& b) {
        ACN_STAT_INC(sub_count);
        align_and_combine(d, a, b, /*subtract=*/true);
    }

    /// d = a * b, allocation-free (n <= kInlineN); d must not alias a or b.
    friend void mul_into(AcnDyn& d, const AcnDyn& a, const AcnDyn& b) {
        ACN_ASSERT(&d != &a && &d != &b);
        ACN_ASSERT(a.n_ == b.n_);
        ACN_STAT_INC(mul_count);
        if (d.n_ != a.n_) {
            d.release();
            d.allocate_zeroed(a.n_);
        }
        ring::mul<Int>(d.coeff_span(), a.coeff_span(), b.coeff_span());
        d.k_ = static_cast<ScaleExp>(a.k_ + b.k_);
        d.canonicalize_self();
    }

    /// d += a*b, single canonicalize, allocation-free (n <= kInlineN);
    /// safe if d aliases a or b.
    // See Acn<N,Int>::fma_into: one canonicalize instead of mul_into's-
    // then-add_into's-two, by combining the convolution and the
    // scale-alignment before ever calling canonicalize_self. Safe even
    // when d aliases a or b (only read, via `product`/`dc`, before d's
    // storage is resized/written).
    friend void fma_into(AcnDyn& d, const AcnDyn& a, const AcnDyn& b) {
        ACN_ASSERT(a.n_ == b.n_);
        ACN_STAT_INC(fma_count);
        // Unlike mul_into (where d is a pure output, freely resized to
        // a.n_), d here is an accumulator whose existing value must
        // survive, so a real precision mismatch is a caller bug, not a
        // resize -- matching add_into/align_and_combine's own
        // ACN_ASSERT(a.n_ == b.n_) contract.
        ACN_ASSERT(d.n_ == a.n_);
        u32 n = a.n_;
        auto product = detail::scratch_span<Int, detail::ScratchTag::FmaProduct>(n);
        ring::mul<Int>(product, a.coeff_span(), b.coeff_span());
        ScaleExp product_k = static_cast<ScaleExp>(a.k_ + b.k_);

        auto dc = detail::scratch_span<Int, detail::ScratchTag::FmaAccumulator>(n);
        for (u32 i = 0; i < n; ++i)
            dc[i] = d.c_[i];
        ScaleExp target_k = std::max(d.k_, product_k);
        apply_sqrt2_power(dc, static_cast<long long>(target_k) - static_cast<long long>(d.k_));
        apply_sqrt2_power(product, static_cast<long long>(target_k) - static_cast<long long>(product_k));

        ring::add<Int>(d.coeff_span(), std::span<const Int>(dc), std::span<const Int>(product));
        d.k_ = target_k;
        d.canonicalize_self();
    }

    /// Ring embedding R_n -> R_target_n (n | target_n); a ring homomorphism.
    // See Acn<N,Int>::widen/narrow. target_n must be a multiple of n_.
    AcnDyn widen(u32 target_n) const {
        ACN_ASSERT(target_n % n_ == 0);
        ACN_STAT_INC(widen_calls);
        std::vector<Int> dst(target_n, Int{});
        u32 stride = target_n / n_;
        for (u32 j = 0; j < n_; ++j)
            dst[j * stride] = c_[j];
        return AcnDyn::from_coeffs(target_n, dst, k_);
    }

    /// Inverse of widen(); nullopt if not in R_target_n's image.
    // n_ must be a multiple of target_n. Succeeds iff every coefficient
    // outside the sublattice (index not a multiple of n_/target_n) is zero.
    std::optional<AcnDyn> narrow(u32 target_n) const {
        ACN_ASSERT(n_ % target_n == 0);
        ACN_STAT_INC(narrow_calls);
        u32 stride = n_ / target_n;
        for (u32 j = 0; j < n_; ++j)
            if (j % stride != 0 && !c_[j].is_zero())
                return std::nullopt;
        std::vector<Int> dst(target_n, Int{});
        for (u32 j = 0; j < target_n; ++j)
            dst[j] = c_[j * stride];
        return AcnDyn::from_coeffs(target_n, dst, k_);
    }

private:
    // Multiplies coefficients by sqrt(2)^delta, delta >= 0, staying in R_n.
    // `coeffs` is mutated in place (via the Sqrt2Rotated/Sqrt2Scratch
    // scratch slots) rather than reassigned wholesale, since it's now a
    // span borrowed from some other scratch slot (or from is_integer's own
    // local vector) rather than an owned std::vector.
    static void apply_sqrt2_power(std::span<Int> coeffs, long long delta) {
        if (delta == 0)
            return;
        ACN_ASSERT(delta > 0);
        std::size_t n = coeffs.size();
        long long even_steps = delta / 2;
        if (even_steps > 0) {
            for (auto& value : coeffs)
                for (long long i = 0; i < even_steps; ++i)
                    value = value.shift_left(1);
        }
        if (delta % 2 != 0) {
            ACN_ASSERT(n >= 4);
            auto rotated = detail::scratch_span<Int, detail::ScratchTag::Sqrt2Rotated>(n);
            auto scratch = detail::scratch_span<Int, detail::ScratchTag::Sqrt2Scratch>(n);
            ring::mul_by_sqrt2<Int>(rotated, std::span<const Int>(coeffs), scratch);
            for (std::size_t j = 0; j < n; ++j)
                coeffs[j] = rotated[j];
        }
    }

    static void align_and_combine(AcnDyn& d, const AcnDyn& a, const AcnDyn& b, bool subtract) {
        ACN_ASSERT(a.n_ == b.n_);
        u32 n = a.n_;
        auto ca = detail::scratch_span<Int, detail::ScratchTag::AlignCa>(n);
        auto cb = detail::scratch_span<Int, detail::ScratchTag::AlignCb>(n);
        for (u32 i = 0; i < n; ++i) {
            ca[i] = a.c_[i];
            cb[i] = b.c_[i];
        }
        ScaleExp target_k = std::max(a.k_, b.k_);
        ACN_STAT_INC(scale_alignment_calls);
        ACN_STAT_ADD(scale_alignment_total_distance,
                     (static_cast<long long>(target_k) - static_cast<long long>(a.k_)) +
                         (static_cast<long long>(target_k) - static_cast<long long>(b.k_)));
        apply_sqrt2_power(ca, static_cast<long long>(target_k) - static_cast<long long>(a.k_));
        apply_sqrt2_power(cb, static_cast<long long>(target_k) - static_cast<long long>(b.k_));

        if (d.n_ != n) {
            d.release();
            d.allocate_zeroed(n);
        }
        if (subtract)
            ring::sub<Int>(d.coeff_span(), std::span<const Int>(ca), std::span<const Int>(cb));
        else
            ring::add<Int>(d.coeff_span(), std::span<const Int>(ca), std::span<const Int>(cb));
        d.k_ = target_k;
        d.canonicalize_self();
    }
};

/// Thin factory fixing the precision n for the AcnDyn<Int> values it
/// creates. Not yet arena-backed (each AcnDyn still owns its own
/// storage) -- see PROGRESS.md.
// A thin factory that fixes n for the values it creates. PLAN.md §5.1
// commits to context-owned precision as the primary model; the bump
// allocator / arena it also describes is not implemented yet (each AcnDyn
// still owns its own storage) — see PROGRESS.md.
template <class Int>
class NumberContext {
    u32 n_;

public:
    explicit NumberContext(u32 n) : n_(n) { ACN_ASSERT(n > 0); }

    u32 precision() const { return n_; }

    AcnDyn<Int> zero() const { return AcnDyn<Int>::zero(n_); }
    AcnDyn<Int> one() const { return AcnDyn<Int>::one(n_); }
    AcnDyn<Int> imag() const { return AcnDyn<Int>::imag(n_); }
    AcnDyn<Int> sqrt2() const { return AcnDyn<Int>::sqrt2(n_); }
    AcnDyn<Int> inv_sqrt2() const { return AcnDyn<Int>::inv_sqrt2(n_); }
    AcnDyn<Int> from_ints(std::span<const i64> coeffs, ScaleExp k) const {
        return AcnDyn<Int>::from_ints(n_, coeffs, k);
    }
};

}  // namespace acn
