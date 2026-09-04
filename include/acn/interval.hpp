#pragma once

// PLAN.md §7.2: sound (non-heuristic) bounds on the value to_complex()
// only approximates. This is what a verifier's zero-pruning is allowed to
// trust: the exact value always lies inside the returned interval, and
// to_interval(x).contains(0) is false whenever x.is_zero() is false and the
// interval is tight enough.
//
// Rounding strategy: PLAN.md prefers explicit nextafter widening over
// std::fesetround ("it does not fight the optimizer"). Every primitive
// interval operation here computes the double result the normal
// (round-to-nearest) way and then widens outward by one ULP on each side.
// That single-ULP margin is enough to cover round-to-nearest's own <=0.5
// ULP error; two extra sources need their own accounting, handled where
// they're introduced:
//   - std::cos/std::sin are not guaranteed correctly rounded by the
//     standard (unlike the required-correctly-rounded arithmetic
//     operators), so the root table widens by a larger fixed margin
//     (kLibmUlpMargin) instead of one ULP.
//   - converting an exact Int coefficient to double (Int::to_double()) can
//     lose precision for BigInt coefficients wider than 53 bits. GMP's
//     mpz_get_d truncates rather than rounds, but truncating mantissa bits
//     below the 53rd changes the value by strictly less than the value's
//     own ULP, so the same one-ULP margin used elsewhere still covers it.
//
// Deviation from PLAN.md's std::complex<Interval<double>> sketch: returns
// a small ComplexInterval{re, im} struct instead. std::complex<T> is only
// specified by the standard for T in {float, double, long double}; while
// libstdc++/libc++ happen to instantiate the template for other T, a
// soundness-critical type shouldn't lean on that.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

#include "acn/types.hpp"

namespace acn {

/// Sound directed-rounding interval [lo, hi]; every op widens by
/// enough ULPs to guarantee the true value stays enclosed.
template <class T>
class Interval {
public:
    T lo;
    T hi;

    constexpr Interval() : lo(T(0)), hi(T(0)) {}
    constexpr Interval(T lo_, T hi_) : lo(lo_), hi(hi_) {}

    // A degenerate interval around an exactly-known value.
    static constexpr Interval exact(T v) { return Interval(v, v); }

    // A value known only up to a directed-rounding computation: widen the
    // computed double by one ULP on each side.
    static Interval rounded(T v) { return Interval(round_down(v), round_up(v)); }

    // A value computed by a function without a correct-rounding guarantee
    // (e.g. std::cos/std::sin): widen by `ulps` steps on each side.
    static Interval rounded(T v, int ulps) {
        T lo_bound = v, hi_bound = v;
        for (int i = 0; i < ulps; ++i) {
            lo_bound = round_down(lo_bound);
            hi_bound = round_up(hi_bound);
        }
        return Interval(lo_bound, hi_bound);
    }

    bool contains(T x) const { return lo <= x && x <= hi; }
    bool contains_zero() const { return contains(T(0)); }

    friend Interval operator+(const Interval& a, const Interval& b) {
        return Interval(round_down(a.lo + b.lo), round_up(a.hi + b.hi));
    }

    friend Interval operator-(const Interval& a) { return Interval(-a.hi, -a.lo); }

    friend Interval operator-(const Interval& a, const Interval& b) { return a + (-b); }

    friend Interval operator*(const Interval& a, const Interval& b) {
        T candidates[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
        T lo_val = *std::min_element(candidates, candidates + 4);
        T hi_val = *std::max_element(candidates, candidates + 4);
        return Interval(round_down(lo_val), round_up(hi_val));
    }

private:
    static T round_down(T v) { return std::nextafter(v, -std::numeric_limits<T>::infinity()); }
    static T round_up(T v) { return std::nextafter(v, std::numeric_limits<T>::infinity()); }
};

/// Real/imaginary parts as sound Interval<double> bounds.
struct ComplexInterval {
    Interval<double> re;
    Interval<double> im;
};

namespace interval_detail {

// libm's cos/sin are typically accurate to <1 ULP in practice but the
// standard gives no correct-rounding guarantee; a few ULPs of margin is a
// conservative, cheap way to not depend on that.
inline constexpr int kLibmUlpMargin = 4;

inline Interval<double> interval_pow(Interval<double> base, long long exponent) {
    Interval<double> result = Interval<double>::exact(1.0);
    Interval<double> power = base;
    while (exponent > 0) {
        if (exponent & 1)
            result = result * power;
        power = power * power;
        exponent >>= 1;
    }
    return result;
}

// sqrt2^k as an interval, k of either sign, computed by repeated interval
// multiplication of a directly-bounded sqrt(2) (or 1/sqrt(2)) rather than
// std::pow, so every intermediate rounding step is accounted for instead
// of trusting pow()'s own error bound.
inline Interval<double> sqrt2_power_interval(ScaleExp k) {
    if (k == 0)
        return Interval<double>::exact(1.0);
    Interval<double> sqrt2 = Interval<double>::rounded(std::numbers::sqrt2, kLibmUlpMargin);
    Interval<double> inv_sqrt2 = Interval<double>::rounded(1.0 / std::numbers::sqrt2, kLibmUlpMargin);
    // value = sqrt2^(-k) (DESIGN.md's (c,k) ~ c/sqrt2^k convention).
    return k > 0 ? interval_pow(inv_sqrt2, k) : interval_pow(sqrt2, -static_cast<long long>(k));
}

}  // namespace interval_detail

/// Sound complex-interval evaluation of any Acn\<N,Int\>/AcnDyn\<Int\>:
/// the exact value is always contained in the returned bounds.
// Sound complex-interval evaluation of any AcnT exposing coeffs() (a span
// of Int with is_zero()/to_double()) and scale_exp(). Works identically
// for Acn<N,Int> and AcnDyn<Int>.
template <class AcnT>
ComplexInterval to_interval(const AcnT& value) {
    auto c = value.coeffs();
    const std::size_t n = c.size();

    Interval<double> re_sum = Interval<double>::exact(0.0);
    Interval<double> im_sum = Interval<double>::exact(0.0);

    for (std::size_t j = 0; j < n; ++j) {
        if (c[j].is_zero())
            continue;
        Interval<double> coeff = Interval<double>::rounded(c[j].to_double());
        double angle = std::numbers::pi * static_cast<double>(j) / static_cast<double>(n);
        Interval<double> cos_iv = Interval<double>::rounded(std::cos(angle), interval_detail::kLibmUlpMargin);
        Interval<double> sin_iv = Interval<double>::rounded(std::sin(angle), interval_detail::kLibmUlpMargin);
        re_sum = re_sum + coeff * cos_iv;
        im_sum = im_sum + coeff * sin_iv;
    }

    Interval<double> factor = interval_detail::sqrt2_power_interval(value.scale_exp());
    return ComplexInterval{re_sum * factor, im_sum * factor};
}

}  // namespace acn
