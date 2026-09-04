#pragma once

// The mathematical core: index/sign arithmetic on Z[x]/(x^n+1), the ring
// with basis 1, omega, ..., omega^(n-1) where omega = e^(i*pi/n) and
// omega^n = -1. Coefficient j is the multiplier of omega^j. Nothing here
// knows about the scale exponent or canonical form; those live one layer up
// in canonical.hpp.
//
// n must be a power of two. Every function requires all spans passed to it
// to have the same size n, and (unless documented otherwise) requires the
// output span not to alias any input span.

#include <cstddef>
#include <span>

#include "acn/config.hpp"
#include "acn/stats.hpp"

namespace acn::ring {

/// out = a + b, componentwise.
template <class Int>
void add(std::span<Int> out, std::span<const Int> a, std::span<const Int> b) {
    ACN_ASSERT(out.size() == a.size() && a.size() == b.size());
    for (std::size_t j = 0; j < out.size(); ++j)
        out[j] = a[j] + b[j];
}

/// out = a - b, componentwise.
template <class Int>
void sub(std::span<Int> out, std::span<const Int> a, std::span<const Int> b) {
    ACN_ASSERT(out.size() == a.size() && a.size() == b.size());
    for (std::size_t j = 0; j < out.size(); ++j)
        out[j] = a[j] - b[j];
}

/// out = -a, componentwise.
template <class Int>
void neg(std::span<Int> out, std::span<const Int> a) {
    ACN_ASSERT(out.size() == a.size());
    for (std::size_t j = 0; j < out.size(); ++j)
        out[j] = -a[j];
}

/// Complex conjugate: out[0] = a[0], out[j] = -a[n-j] for 0 < j < n.
template <class Int>
void conj(std::span<Int> out, std::span<const Int> a) {
    ACN_ASSERT(out.size() == a.size());
    ACN_ASSERT(out.data() != a.data());
    const std::size_t n = a.size();
    out[0] = a[0];
    for (std::size_t j = 1; j < n; ++j)
        out[j] = -a[n - j];
}

/// out = omega^j * a (index/sign work only, exact).
// out = omega^j * a, for j in [0, 2n). Coefficient t of a lands at index
// (t + j) mod n, with a sign flip each time the index wraps past n (since
// omega^n = -1). out must not alias a.
template <class Int>
void rotate(std::span<Int> out, std::span<const Int> a, long long j) {
    ACN_ASSERT(out.size() == a.size());
    ACN_ASSERT(out.data() != a.data());
    const long long n = static_cast<long long>(a.size());
    const long long two_n = 2 * n;
    j = ((j % two_n) + two_n) % two_n;
    bool extra_negate = false;
    if (j >= n) {
        j -= n;
        extra_negate = true;
    }
    for (long long t = 0; t < n; ++t) {
        long long idx = t + j;
        bool negate = extra_negate;
        if (idx >= n) {
            idx -= n;
            negate = !negate;
        }
        out[static_cast<std::size_t>(idx)] =
            negate ? -a[static_cast<std::size_t>(t)] : a[static_cast<std::size_t>(t)];
    }
}

/// out = sigma_j(a), the Galois automorphism omega -> omega^j (j odd).
// out = sigma_j(a), the Galois automorphism omega -> omega^j (j odd, i.e.
// coprime to 2n since n is a power of two — the full unit group of Z/2nZ).
// Unlike rotate (which multiplies a by a fixed omega^j), this substitutes
// omega^j for omega *inside* a's own expansion: term c_t*omega^t becomes
// c_t*omega^(t*j). j=1 is the identity; j=2n-1 (i.e. j=-1 mod 2n) is
// conjugation (matches conj() above). out must not alias a.
template <class Int>
void apply_automorphism(std::span<Int> out, std::span<const Int> a, long long j) {
    ACN_ASSERT(out.size() == a.size());
    ACN_ASSERT(out.data() != a.data());
    const long long n = static_cast<long long>(a.size());
    const long long two_n = 2 * n;
    long long jj = ((j % two_n) + two_n) % two_n;
    ACN_ASSERT(jj % 2 == 1);
    for (long long t = 0; t < n; ++t) {
        long long target = (t * jj) % two_n;
        bool negate = false;
        if (target >= n) {
            target -= n;
            negate = true;
        }
        out[static_cast<std::size_t>(target)] =
            negate ? -a[static_cast<std::size_t>(t)] : a[static_cast<std::size_t>(t)];
    }
}

/// Negacyclic convolution (ring multiplication).
// Negacyclic convolution: out[l] = sum_{t+u==l} a[t]*b[u]
//                                 - sum_{t+u==l+n} a[t]*b[u].
// out must not alias a or b.
template <class Int>
void mul(std::span<Int> out, std::span<const Int> a, std::span<const Int> b) {
    ACN_ASSERT(out.size() == a.size() && a.size() == b.size());
    ACN_ASSERT(out.data() != a.data() && out.data() != b.data());
    const std::size_t n = a.size();
    for (std::size_t l = 0; l < n; ++l)
        out[l] = Int{};
    for (std::size_t t = 0; t < n; ++t) {
        if (a[t].is_zero())
            continue;
        for (std::size_t u = 0; u < n; ++u) {
            std::size_t l = t + u;
            Int term = a[t] * b[u];
            if (l < n)
                out[l] = out[l] + term;
            else
                out[l - n] = out[l - n] - term;
        }
    }
}

/// out = a * scalar, componentwise.
template <class Int>
void mul_small(std::span<Int> out, std::span<const Int> a, Int scalar) {
    ACN_ASSERT(out.size() == a.size());
    for (std::size_t j = 0; j < out.size(); ++j)
        out[j] = a[j] * scalar;
}

/// out = sqrt(2) * a; requires n >= 4.
// sqrt(2) = omega^(n/4) - omega^(3n/4), valid only for n >= 4 (there is no
// omega^(n/4) for smaller n). scratch must have the same size as out/a and
// must not alias out or a.
template <class Int>
void mul_by_sqrt2(std::span<Int> out, std::span<const Int> a, std::span<Int> scratch) {
    ACN_ASSERT(out.size() == a.size() && a.size() == scratch.size());
    ACN_ASSERT(out.data() != a.data());
    ACN_ASSERT(scratch.data() != a.data() && scratch.data() != out.data());
    const long long n = static_cast<long long>(a.size());
    ACN_ASSERT(n >= 4);
    ACN_STAT_INC(mul_by_sqrt2_calls);
    rotate<Int>(out, a, n / 4);
    rotate<Int>(scratch, a, 3 * n / 4);
    for (std::size_t j = 0; j < out.size(); ++j)
        out[j] = out[j] - scratch[j];
}

/// out = a * conj(a); fixed by conjugation (real-valued).
// a * conj(a); the result is fixed by conjugation (its "imaginary part" is
// zero) but is not generally a single coefficient. scratch must have the
// same size as a and must not alias out or a.
template <class Int>
void norm2(std::span<Int> out, std::span<const Int> a, std::span<Int> scratch) {
    ACN_ASSERT(out.size() == a.size() && a.size() == scratch.size());
    ACN_ASSERT(scratch.data() != a.data());
    conj<Int>(scratch, a);
    mul<Int>(out, a, std::span<const Int>(scratch));
}

}  // namespace acn::ring
