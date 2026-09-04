#include "reference.hpp"

#include "acn/canonical.hpp"

namespace acn::reference {

RefValue RefValue::zero(std::size_t n) {
    RefValue result;
    result.coeffs.assign(n, BigInt(0));
    result.k = 0;
    return result;
}

RefValue RefValue::from_ints(std::size_t n, std::span<const i64> ints, ScaleExp k) {
    RefValue result;
    result.coeffs.reserve(n);
    for (i64 v : ints)
        result.coeffs.emplace_back(v);
    result.k = k;
    return result;
}

RefValue RefValue::monomial(std::size_t n, long long exponent, BigInt sign) {
    long long two_n = static_cast<long long>(2 * n);
    long long normalized = ((exponent % two_n) + two_n) % two_n;
    RefValue result = RefValue::zero(n);
    if (normalized >= static_cast<long long>(n))
        result.coeffs[static_cast<std::size_t>(normalized - static_cast<long long>(n))] = -sign;
    else
        result.coeffs[static_cast<std::size_t>(normalized)] = sign;
    return result;
}

RefValue ref_neg(const RefValue& a) {
    RefValue result = a;
    for (auto& c : result.coeffs)
        c = -c;
    return result;
}

RefValue ref_conj(const RefValue& a) {
    std::size_t n = a.n();
    RefValue result = RefValue::zero(n);
    result.k = a.k;
    result.coeffs[0] = a.coeffs[0];
    for (std::size_t j = 1; j < n; ++j)
        result.coeffs[j] = -a.coeffs[n - j];
    return result;
}

RefValue ref_mul(const RefValue& a, const RefValue& b) {
    std::size_t n = a.n();
    RefValue result = RefValue::zero(n);
    result.k = static_cast<ScaleExp>(a.k + b.k);
    for (std::size_t t = 0; t < n; ++t) {
        if (a.coeffs[t] == BigInt(0))
            continue;
        for (std::size_t u = 0; u < n; ++u) {
            if (b.coeffs[u] == BigInt(0))
                continue;
            BigInt term = a.coeffs[t] * b.coeffs[u];
            std::size_t l = t + u;
            if (l < n)
                result.coeffs[l] = result.coeffs[l] + term;
            else
                result.coeffs[l - n] = result.coeffs[l - n] - term;
        }
    }
    return result;
}

RefValue ref_mul_by_sqrt2(const RefValue& a) {
    std::size_t n = a.n();
    RefValue sqrt2 = RefValue::zero(n);
    // sqrt(2) = omega^(n/4) - omega^(3n/4); build it directly as a
    // two-term value rather than via ref_sub of two monomials, to avoid
    // routing through ref_sub's scale-alignment logic (both k's are 0
    // here, but this keeps the dependency direction one-way: ref_add/sub
    // depend on ref_mul_by_sqrt2, not the other way around).
    sqrt2.coeffs[n / 4] = BigInt(1);
    sqrt2.coeffs[3 * n / 4] = BigInt(-1);
    return ref_mul(a, sqrt2);
}

namespace {
RefValue align_up(const RefValue& a, ScaleExp target_k) {
    RefValue result = a;
    while (result.k < target_k) {
        result = ref_mul_by_sqrt2(result);
        result.k = static_cast<ScaleExp>(result.k - 1);
    }
    return result;
}
}  // namespace

RefValue ref_add(const RefValue& a, const RefValue& b) {
    ScaleExp target_k = std::max(a.k, b.k);
    RefValue aa = align_up(a, target_k);
    RefValue bb = align_up(b, target_k);
    RefValue result = RefValue::zero(a.n());
    result.k = target_k;
    for (std::size_t j = 0; j < result.n(); ++j)
        result.coeffs[j] = aa.coeffs[j] + bb.coeffs[j];
    return result;
}

RefValue ref_sub(const RefValue& a, const RefValue& b) {
    ScaleExp target_k = std::max(a.k, b.k);
    RefValue aa = align_up(a, target_k);
    RefValue bb = align_up(b, target_k);
    RefValue result = RefValue::zero(a.n());
    result.k = target_k;
    for (std::size_t j = 0; j < result.n(); ++j)
        result.coeffs[j] = aa.coeffs[j] - bb.coeffs[j];
    return result;
}

std::pair<std::vector<BigInt>, ScaleExp> canonicalize_ref(const RefValue& ref) {
    std::vector<BigInt> coeffs = ref.coeffs;
    ScaleExp k = ref.k;
    std::vector<BigInt> scratch(2 * coeffs.size());
    acn::canonicalize<BigInt>(std::span<BigInt>(coeffs), k, std::span<BigInt>(scratch));
    return {std::move(coeffs), k};
}

}  // namespace acn::reference
