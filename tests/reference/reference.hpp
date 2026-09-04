#pragma once

// PLAN.md §9.5: a deliberately naive, obviously-correct oracle for R_n =
// Z[x]/(x^n+1) arithmetic with an explicit (symbolic) sqrt(2) scale
// exponent, built independently of include/acn/ring.hpp and
// include/acn/canonical.hpp. The point is that a shared bug in those two
// (the prototype's actual failure mode: a sign error that showed up
// identically across code that only *looked* independent) can't hide from
// a differential test against genuinely separate code.
//
// The only real computational kernel here is ref_mul: a direct O(n^2)
// negacyclic convolution. Every other operation -- multiplication by a
// root of unity, multiplication by sqrt(2), the scale-alignment add/sub
// need -- is expressed in terms of ref_mul plus componentwise add/sub,
// rather than given its own index/sign-bookkeeping implementation (doing
// that would just be re-deriving ring::rotate's logic under a different
// name, which is not independent of it). This is slower and allocates
// freely; it is not meant to be fast, only obviously correct.
//
// RefValue never canonicalizes -- add/sub/mul just produce whatever
// (coeffs, k) the direct formulas give, exactly like DESIGN.md's
// definition of the ring operations before Phase 2's canonical-form layer
// is applied. To compare a RefValue against a value from the real
// library, canonicalize a *copy* of its raw coefficients through
// acn::canonicalize<BigInt> (reusing that one piece of the real library --
// its correctness is covered independently and thoroughly by Phase 2's
// uniqueness/idempotence/parity-fast-path tests, so treating it as a
// trusted primitive here is reasonable) and compare coefficient-by-
// coefficient against the real value's own coeffs()/scale_exp() (widened
// to BigInt first if the real value uses Checked<Word>).
//
// Only usable when GMP is available (RefValue is BigInt-based throughout);
// gated on ACN_WITH_GMP in tests/CMakeLists.txt, same as test_bigint.cpp.

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "acn/bigint.hpp"
#include "acn/types.hpp"

namespace acn::reference {

class RefValue {
public:
    std::vector<BigInt> coeffs;
    ScaleExp k = 0;

    std::size_t n() const { return coeffs.size(); }

    static RefValue zero(std::size_t n);
    static RefValue from_ints(std::size_t n, std::span<const i64> ints, ScaleExp k);

    // sign * omega^exponent, omega = e^(i*pi/n), exponent taken mod 2n.
    // The negacyclic sign rule (DESIGN.md's index convention) written
    // directly here: normalize exponent into [0, 2n), then the wrap past
    // n (if any) is the one sign flip.
    static RefValue monomial(std::size_t n, long long exponent, BigInt sign = BigInt(1));
};

RefValue ref_neg(const RefValue& a);
RefValue ref_conj(const RefValue& a);  // out[0]=a[0], out[j] = -a[n-j] for 0<j<n

// Direct O(n^2) negacyclic convolution:
//   out[l] = sum_{t+u==l} a[t]*b[u]  -  sum_{t+u==l+n} a[t]*b[u]
// `a` and `b` must have the same n; the result's k is a.k + b.k.
RefValue ref_mul(const RefValue& a, const RefValue& b);

// a * sqrt(2), via sqrt(2) = omega^(n/4) - omega^(3n/4) (DESIGN.md), built
// from `monomial` and `ref_mul` rather than a bespoke implementation.
// Requires n() >= 4 (no omega^(n/4) below that, same constraint ring.hpp
// documents).
RefValue ref_mul_by_sqrt2(const RefValue& a);

// Scale-aligned add/sub: whichever operand has the smaller k is raised to
// max(a.k, b.k) via repeated ref_mul_by_sqrt2 (one symbolic sqrt(2) step
// at a time -- no bulk/parity-mask shortcut, unlike canonical.hpp's), then
// combined componentwise. Requires n() >= 4 if a.k != b.k.
RefValue ref_add(const RefValue& a, const RefValue& b);
RefValue ref_sub(const RefValue& a, const RefValue& b);

// Canonicalizes a *copy* of ref's raw (coeffs, k) via the real library's
// acn::canonicalize<BigInt> and returns the result, for the caller to
// compare against a real Acn<N,Int>/AcnDyn<Int> value's own
// coeffs()/scale_exp().
std::pair<std::vector<BigInt>, ScaleExp> canonicalize_ref(const RefValue& ref);

}  // namespace acn::reference
