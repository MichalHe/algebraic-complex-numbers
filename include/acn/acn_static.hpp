#pragma once

// Acn<N, Int>: the compile-time-precision algebraic complex number type.
// A value (c, k) with c an array of N coefficients (coefficient j
// multiplies omega^j, omega = e^(i*pi/N), omega^N = -1) and k a scale
// exponent denotes u / sqrt(2)^k where u = sum c_j * omega^j. Every public
// operation returns a canonical value (DESIGN.md §4), so operator== is a
// componentwise compare and hash_value may hash the raw representation.
//
// Operators are convenience; the _into forms are the performance API and
// are what the library's own internals (matrix, echelon) use.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <compare>
#include <complex>
#include <cstddef>
#include <numbers>
#include <optional>
#include <span>

#include "acn/canonical.hpp"
#include "acn/config.hpp"
#include "acn/ring.hpp"
#include "acn/stats.hpp"
#include "acn/types.hpp"

namespace acn {

/// Compile-time-precision element of Z[x]/(x^N+1)[1/sqrt2]: canonical
/// (coeffs, k) pair denoting sum(coeffs[j]*omega^j) / sqrt2^k.
template <std::size_t N, class Int>
class Acn {
    static_assert(std::has_single_bit(N), "N must be a power of two");

    std::array<Int, N> c_{};
    ScaleExp k_ = 0;

    std::span<Int> coeff_span() { return std::span<Int>(c_); }
    std::span<const Int> coeff_span() const { return std::span<const Int>(c_); }

    void canonicalize_self() {
        std::array<Int, 2 * N> scratch{};
        acn::canonicalize<Int>(coeff_span(), k_, std::span<Int>(scratch));
    }

public:
    static constexpr std::size_t precision = N;
    using int_type = Int;

    Acn() = default;

    /// Additive identity.
    static Acn zero() { return Acn(); }

    /// Multiplicative identity.
    static Acn one() {
        Acn result;
        result.c_[0] = Int(1);
        return result;
    }

    /// i = omega^(N/2).
    static Acn imag() {
        static_assert(N >= 2, "omega^(N/2) requires N >= 2");
        Acn result;
        result.c_[N / 2] = Int(1);
        return result;
    }

    /// sqrt(2) = omega^(N/4) - omega^(3N/4).
    static Acn sqrt2() {
        static_assert(N >= 4, "sqrt(2) = omega^(N/4) - omega^(3N/4) requires N >= 4");
        Acn result;
        result.c_[N / 4] = Int(1);
        result.c_[3 * N / 4] = Int(-1);
        result.canonicalize_self();
        return result;
    }

    /// 1/sqrt(2).
    static Acn inv_sqrt2() {
        static_assert(N >= 4, "1/sqrt(2) requires N >= 4");
        Acn result;
        result.c_[0] = Int(1);
        result.k_ = 1;
        return result;
    }

    /// omega^j, j taken mod 2N.
    static Acn omega(long long j) {
        Acn result;
        std::array<Int, N> basis{};
        basis[0] = Int(1);
        ring::rotate<Int>(result.coeff_span(), std::span<const Int>(basis), j);
        return result;
    }

    /// Builds sum(coeffs[j]*omega^j) / sqrt2^k from machine-word coefficients.
    static Acn from_ints(const std::array<i64, N>& coeffs, ScaleExp k) {
        Acn result;
        for (std::size_t j = 0; j < N; ++j)
            result.c_[j] = Int(coeffs[j]);
        result.k_ = k;
        result.canonicalize_self();
        return result;
    }

    // Like from_ints, but takes already-Int-typed coefficients (needed when
    // Int is BigInt and a coefficient may not fit in i64, e.g. divide.hpp's
    // intermediate products).
    /// Like from_ints(), but takes already-Int-typed coefficients (for
    /// values, e.g. BigInt, that don't fit in i64).
    static Acn from_coeffs(std::span<const Int> coeffs, ScaleExp k) {
        ACN_ASSERT(coeffs.size() == N);
        Acn result;
        for (std::size_t j = 0; j < N; ++j)
            result.c_[j] = coeffs[j];
        result.k_ = k;
        result.canonicalize_self();
        return result;
    }

    std::span<const Int> coeffs() const { return coeff_span(); }
    ScaleExp scale_exp() const { return k_; }

    /// True iff this value is exactly zero.
    bool is_zero() const { return all_zero<Int>(coeff_span()); }

    /// True iff this value is exactly one.
    bool is_one() const {
        if (k_ != 0)
            return false;
        if (!(c_[0] == Int(1)))
            return false;
        for (std::size_t j = 1; j < N; ++j)
            if (!c_[j].is_zero())
                return false;
        return true;
    }

    /// True iff this value equals its own conjugate.
    // Fixed by conjugation: c_j == -c_{N-j} for 0 < j < N/2 style symmetry.
    bool is_real() const {
        std::array<Int, N> conjugate{};
        ring::conj<Int>(std::span<Int>(conjugate), coeff_span());
        return *this == from_raw(conjugate, k_);
    }

    /// True iff this value is a rational integer.
    // Correct for negative k: canonical form pushes k as negative as
    // possible (e.g. the integer 2 canonicalizes to c=(1,0,...), k=-2), so
    // an integer does not generally have k==0. k>0 in canonical form can
    // never be an integer (that would mean sqrt(2) divides the numerator,
    // contradicting canonicality) except for zero, handled separately.
    bool is_integer() const {
        if (is_zero())
            return true;
        if (k_ > 0)
            return false;
        std::array<Int, N> scaled = c_;
        if (k_ < 0)
            apply_sqrt2_power(scaled, -static_cast<long long>(k_));
        for (std::size_t j = 1; j < N; ++j)
            if (!scaled[j].is_zero())
                return false;
        return true;
    }

    /// True iff this value is a (dyadic) rational number.
    // A value is rational iff it's a dyadic rational: since {1, omega, ...,
    // omega^(N-1)} is a Z-basis of R_N, the only combination that is
    // rational is c_0 alone (the basis is linearly independent over Q, and
    // omega^j for j != 0 is irrational). sqrt(2)^k is itself rational iff k
    // is even (a power of 2); an odd k in canonical form means sqrt(2)
    // genuinely doesn't divide out of the numerator, so the value is
    // irrational regardless of which coefficient is nonzero.
    bool is_rational() const {
        if (is_zero())
            return true;
        if (k_ % 2 != 0)
            return false;
        for (std::size_t j = 1; j < N; ++j)
            if (!c_[j].is_zero())
                return false;
        return true;
    }

    /// True iff this value is a unit (+-omega^j * sqrt(2)^m).
    // Units are +-omega^j * sqrt(2)^m (DESIGN.md §5.3): exactly one
    // coefficient nonzero, with value +-1. k is unconstrained (sqrt(2) is a
    // unit, so any power of it composes with any rotation).
    bool is_unit() const {
        bool seen_nonzero = false;
        for (std::size_t j = 0; j < N; ++j) {
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

    friend bool operator==(const Acn& a, const Acn& b) {
        if (a.k_ != b.k_)
            return false;
        for (std::size_t j = 0; j < N; ++j)
            if (!(a.c_[j] == b.c_[j]))
                return false;
        return true;
    }

    friend std::strong_ordering operator<=>(const Acn& a, const Acn& b) {
        if (auto cmp = a.k_ <=> b.k_; cmp != 0)
            return cmp;
        for (std::size_t j = 0; j < N; ++j)
            if (auto cmp = a.c_[j] <=> b.c_[j]; cmp != 0)
                return cmp;
        return std::strong_ordering::equal;
    }

    Acn operator-() const {
        Acn result;
        ring::neg<Int>(result.coeff_span(), coeff_span());
        result.k_ = k_;
        return result;
    }

    /// Complex conjugate.
    Acn conj() const {
        Acn result;
        ring::conj<Int>(result.coeff_span(), coeff_span());
        result.k_ = k_;
        return result;
    }

    /// Multiplies by omega^j (index/sign work only, exact, no canonicalize).
    // Rotation by omega^j: index/sign work only, no coefficient multiplies
    // and no canonicalization needed (it preserves sqrt(2)-valuation).
    Acn mul_omega(long long j) const {
        Acn result;
        ring::rotate<Int>(result.coeff_span(), coeff_span(), j);
        result.k_ = k_;
        return result;
    }

    /// Multiplies by sqrt(2)^(-d); d may be negative.
    // Multiply by sqrt(2)^(-d): k -= d, then canonicalize. d may be negative.
    Acn scale(ScaleExp d) const {
        Acn result = *this;
        result.k_ = static_cast<ScaleExp>(result.k_ - d);
        result.canonicalize_self();
        return result;
    }

    Acn operator+(const Acn& other) const {
        Acn result;
        add_into(result, *this, other);
        return result;
    }

    Acn operator-(const Acn& other) const {
        Acn result;
        sub_into(result, *this, other);
        return result;
    }

    Acn operator*(const Acn& other) const {
        Acn result;
        mul_into(result, *this, other);
        return result;
    }

    Acn& operator+=(const Acn& other) { return *this = *this + other; }
    Acn& operator-=(const Acn& other) { return *this = *this - other; }
    Acn& operator*=(const Acn& other) { return *this = *this * other; }

    /// this * conj(this); fixed by conjugation (real-valued).
    Acn norm2() const {
        Acn result;
        std::array<Int, N> scratch{};
        ring::norm2<Int>(result.coeff_span(), coeff_span(), std::span<Int>(scratch));
        result.k_ = static_cast<ScaleExp>(2 * k_);
        result.canonicalize_self();
        return result;
    }

    /// Heuristic (rounded) double-precision approximation; not sound --
    /// use to_interval() for soundness-critical comparisons.
    std::complex<double> to_complex() const {
        std::complex<double> sum(0.0, 0.0);
        for (std::size_t j = 0; j < N; ++j) {
            double angle = std::numbers::pi * static_cast<double>(j) / static_cast<double>(N);
            sum += c_[j].to_double() * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        double factor = std::pow(1.0 / std::numbers::sqrt2, static_cast<double>(k_));
        return sum * factor;
    }

    /// d = a + b, allocation-free. Performance API; see class docs.
    friend void add_into(Acn& d, const Acn& a, const Acn& b) {
        ACN_STAT_INC(add_count);
        align_and_combine(d, a, b, /*subtract=*/false);
    }

    /// d = a - b, allocation-free.
    friend void sub_into(Acn& d, const Acn& a, const Acn& b) {
        ACN_STAT_INC(sub_count);
        align_and_combine(d, a, b, /*subtract=*/true);
    }

    /// d = a * b, allocation-free; d must not alias a or b.
    friend void mul_into(Acn& d, const Acn& a, const Acn& b) {
        ACN_ASSERT(&d != &a && &d != &b);
        ACN_STAT_INC(mul_count);
        ring::mul<Int>(d.coeff_span(), a.coeff_span(), b.coeff_span());
        d.k_ = static_cast<ScaleExp>(a.k_ + b.k_);
        d.canonicalize_self();
    }

    /// d += a*b, single canonicalize, allocation-free; safe if d aliases a or b.
    // d += a*b with a single canonicalize (not mul_into's canonicalize
    // followed by add_into's), by combining mul_into's convolution and
    // add_into's scale-alignment into one pass before ever calling
    // canonicalize_self. Safe even when d aliases a or b: a/b are only
    // read (into the freshly computed `product`, and via the copy `dc` of
    // d's own coefficients) before d's storage is written.
    friend void fma_into(Acn& d, const Acn& a, const Acn& b) {
        ACN_STAT_INC(fma_count);
        std::array<Int, N> product{};
        ring::mul<Int>(std::span<Int>(product), a.coeff_span(), b.coeff_span());
        ScaleExp product_k = static_cast<ScaleExp>(a.k_ + b.k_);

        std::array<Int, N> dc = d.c_;
        ScaleExp target_k = std::max(d.k_, product_k);
        apply_sqrt2_power(dc, static_cast<long long>(target_k) - static_cast<long long>(d.k_));
        apply_sqrt2_power(product, static_cast<long long>(target_k) - static_cast<long long>(product_k));

        ring::add<Int>(d.coeff_span(), std::span<const Int>(dc), std::span<const Int>(product));
        d.k_ = target_k;
        d.canonicalize_self();
    }

    // R_N embeds into R_M when N | M via omega_N -> omega_M^(M/N)
    // (DESIGN.md §2.2/§6): coefficient j moves to index j*(M/N), the rest
    // stay zero, and k is unchanged. A ring homomorphism, so it commutes
    // with +, *, and the additive/multiplicative identities. Canonicalized
    // defensively via from_coeffs (embeddings are expected to preserve
    // canonicity, but re-deriving it here costs little and removes any
    // need to trust that across ring extensions).
    /// Ring embedding R_N -> R_M (N | M); a ring homomorphism.
    template <std::size_t M>
    Acn<M, Int> widen() const {
        static_assert(M % N == 0, "widen<M>() requires N | M");
        ACN_STAT_INC(widen_calls);
        std::array<Int, M> dst{};
        constexpr std::size_t stride = M / N;
        for (std::size_t j = 0; j < N; ++j)
            dst[j * stride] = c_[j];
        return Acn<M, Int>::from_coeffs(std::span<const Int>(dst), k_);
    }

    /// Inverse of widen(); nullopt if not in R_M's image.
    // Inverse of widen<N>(): succeeds iff every coefficient outside the
    // sublattice (index not a multiple of N/M) is zero.
    template <std::size_t M>
    std::optional<Acn<M, Int>> narrow() const {
        static_assert(N % M == 0, "narrow<M>() requires M | N");
        ACN_STAT_INC(narrow_calls);
        constexpr std::size_t stride = N / M;
        for (std::size_t j = 0; j < N; ++j)
            if (j % stride != 0 && !c_[j].is_zero())
                return std::nullopt;
        std::array<Int, M> dst{};
        for (std::size_t j = 0; j < M; ++j)
            dst[j] = c_[j * stride];
        return Acn<M, Int>::from_coeffs(std::span<const Int>(dst), k_);
    }

private:
    static Acn from_raw(const std::array<Int, N>& coeffs, ScaleExp k) {
        Acn result;
        result.c_ = coeffs;
        result.k_ = k;
        return result;
    }

    // Multiplies coefficients by sqrt(2)^delta, delta >= 0, staying in R_n.
    // Even part is a plain shift; an odd remainder costs one mul_by_sqrt2.
    static void apply_sqrt2_power(std::array<Int, N>& coeffs, long long delta) {
        if (delta == 0)
            return;
        ACN_ASSERT(delta > 0);
        long long even_steps = delta / 2;
        if (even_steps > 0) {
            for (auto& value : coeffs)
                for (long long i = 0; i < even_steps; ++i)
                    value = value.shift_left(1);
        }
        if (delta % 2 != 0) {
            ACN_ASSERT(N >= 4);
            std::array<Int, N> rotated{};
            std::array<Int, N> scratch{};
            if constexpr (N >= 4) {
                ring::mul_by_sqrt2<Int>(std::span<Int>(rotated), std::span<const Int>(coeffs),
                                         std::span<Int>(scratch));
            }
            coeffs = rotated;
        }
    }

    static void align_and_combine(Acn& d, const Acn& a, const Acn& b, bool subtract) {
        std::array<Int, N> ca = a.c_;
        std::array<Int, N> cb = b.c_;
        ScaleExp target_k = std::max(a.k_, b.k_);
        ACN_STAT_INC(scale_alignment_calls);
        ACN_STAT_ADD(scale_alignment_total_distance,
                     (static_cast<long long>(target_k) - static_cast<long long>(a.k_)) +
                         (static_cast<long long>(target_k) - static_cast<long long>(b.k_)));
        apply_sqrt2_power(ca, static_cast<long long>(target_k) - static_cast<long long>(a.k_));
        apply_sqrt2_power(cb, static_cast<long long>(target_k) - static_cast<long long>(b.k_));
        if (subtract)
            ring::sub<Int>(std::span<Int>(d.c_), std::span<const Int>(ca), std::span<const Int>(cb));
        else
            ring::add<Int>(std::span<Int>(d.c_), std::span<const Int>(ca), std::span<const Int>(cb));
        d.k_ = target_k;
        d.canonicalize_self();
    }
};

}  // namespace acn
