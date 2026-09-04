#pragma once

// Division is not total in R_n[1/sqrt(2)] (DESIGN.md §5.3): the API is
// explicit about it. divide_by_unit is the total, cheap fast path DD
// callers should actually use; divide_exact is the general (and much more
// expensive) test-and-divide, implemented via the field norm rather than
// any general-field-arithmetic machinery — no Frac<Acn> wrapper here, per
// DESIGN.md §5.3/§1.1.
//
// Works identically for Acn<N,Int> and AcnDyn<Int> (both expose coeffs(),
// scale_exp(), int_type, mul_omega(), scale(), from_coeffs()).

#include <optional>
#include <vector>

#include "acn/config.hpp"
#include "acn/ring.hpp"
#include "acn/types.hpp"

namespace acn {

/// a / u for a unit u; total, never fails (undefined if u isn't a unit --
/// check is_unit() first).
// a / u for a unit u = +-omega^j * sqrt(2)^m. Total (never fails) as long
// as u actually is a unit — check is_unit() first if that isn't already
// known. Implemented as the inverse of u applied directly: negate if the
// sign is -1, un-rotate by j, un-scale by m.
template <class AcnT>
AcnT divide_by_unit(const AcnT& a, const AcnT& u) {
    using Int = typename AcnT::int_type;
    auto uc = u.coeffs();
    long long unit_index = -1;
    Int sign{};
    for (std::size_t j = 0; j < uc.size(); ++j) {
        if (!uc[j].is_zero()) {
            unit_index = static_cast<long long>(j);
            sign = uc[j];
            break;
        }
    }
    ACN_ASSERT(unit_index >= 0);  // u must be nonzero (and, per contract, a unit)

    AcnT result = (sign == Int(-1)) ? -a : a;
    result = result.mul_omega(-unit_index);
    // u == sign*omega^unit_index * sqrt2^m, and since u's canonical numerator
    // is exactly the monomial sign*omega^unit_index, the (c,k)~c/sqrt2^k
    // convention gives u.scale_exp() == -m. So u^-1's sqrt2^-m factor equals
    // sqrt2^(u.scale_exp()); scale(d) multiplies the value by sqrt2^d, so this
    // is scale(u.scale_exp()) -- NOT scale(-u.scale_exp()), which was the bug.
    result = result.scale(u.scale_exp());
    return result;
}

/// a / b via the field norm; nullopt if b doesn't evenly divide a (or
/// b == 0). O(n^3), not for the innermost loop.
// a / b, or nullopt if b does not evenly divide a in the ring (b == 0
// always fails). General method: for n a power of two, the Galois group of
// Q(zeta_2n) over Q is exactly the odd residues mod 2n (n of them); the
// field norm N(b) = product over all such automorphisms sigma_j of
// sigma_j(b) is a rational integer, and a/b = a * (product over sigma_j,
// j!=1) / N(b), which is exact in the ring iff every coefficient of the
// numerator product divides N(b) evenly. O(n) automorphisms, each an O(n)
// substitution combined via O(n) ring multiplications, so O(n^3) total —
// deliberately not the innermost loop (DESIGN.md §5.3 says as much).
template <class AcnT>
std::optional<AcnT> divide_exact(const AcnT& a, const AcnT& b) {
    using Int = typename AcnT::int_type;

    if (b.is_zero())
        return std::nullopt;

    auto a_coeffs = a.coeffs();
    auto b_coeffs = b.coeffs();
    ACN_ASSERT(a_coeffs.size() == b_coeffs.size());
    const std::size_t n = b_coeffs.size();
    const long long two_n = static_cast<long long>(2 * n);

    std::vector<Int> u_b(b_coeffs.begin(), b_coeffs.end());
    std::vector<Int> co_norm(n, Int{});
    co_norm[0] = Int(1);  // multiplicative identity: the empty product

    std::vector<Int> automorphism_image(n), product_scratch(n);
    for (long long j = 3; j < two_n; j += 2) {
        ring::apply_automorphism<Int>(automorphism_image, u_b, j);
        ring::mul<Int>(product_scratch, co_norm, std::span<const Int>(automorphism_image));
        co_norm = product_scratch;
    }

    std::vector<Int> norm_vec(n);
    ring::mul<Int>(norm_vec, std::span<const Int>(u_b), std::span<const Int>(co_norm));
    // The field norm always lands in Z; a nonzero non-c_0 coefficient here
    // would mean a bug upstream (ring.hpp's automorphism/mul, or the field
    // theory assumption that n is a power of two), not a user error.
    for (std::size_t j = 1; j < n; ++j)
        ACN_ASSERT(norm_vec[j].is_zero());

    if (norm_vec[0].is_zero())
        return std::nullopt;  // shouldn't happen since b != 0, but stay defensive

    // norm_vec[0] need not evenly divide the numerator's coordinates even
    // when b genuinely divides a: sqrt(2)^2 == 2, so a factor of 2 in the
    // norm can be absorbed by moving it into k (an extra sqrt2-power
    // reduction) rather than requiring it to divide out coordinatewise.
    // Split norm = 2^e * odd_part; only odd_part needs to divide the
    // numerator's coordinates directly, and e feeds into the result's k.
    Int norm = norm_vec[0];
    int power_of_two = 0;
    while (norm.is_even()) {
        norm = norm.halve_exact();
        ++power_of_two;
    }

    std::vector<Int> numerator(n);
    ring::mul<Int>(numerator, a_coeffs, std::span<const Int>(co_norm));

    std::vector<Int> quotient(n);
    for (std::size_t j = 0; j < n; ++j) {
        auto q = numerator[j].divide_exact(norm);
        if (!q)
            return std::nullopt;
        quotient[j] = *q;
    }

    ScaleExp result_k = static_cast<ScaleExp>(a.scale_exp() - b.scale_exp() + 2 * power_of_two);
    return AcnT::from_coeffs(quotient, result_k);
}

}  // namespace acn
