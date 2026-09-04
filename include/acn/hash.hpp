#pragma once

// hash_value(a): hashes the raw canonical representation. Precondition:
// a is canonical, which every public operation guarantees. Generic over
// any coefficient backend that provides hash_component() (Checked<Word>,
// BigInt), not tied to a fixed-width .value().

#include <cstddef>

#include "acn/acn_dynamic.hpp"
#include "acn/acn_static.hpp"
#include "acn/hash_mix.hpp"
#include "acn/types.hpp"

namespace acn {

/// Hashes a's canonical representation; a must already be canonical.
template <class AcnT>
u64 hash_value(const AcnT& a) {
    u64 h = mix64(static_cast<u64>(a.scale_exp()));
    for (const auto& coeff : a.coeffs())
        h = mix64(h ^ coeff.hash_component());
    return h;
}

}  // namespace acn

namespace std {

template <std::size_t N, class Int>
struct hash<::acn::Acn<N, Int>> {
    std::size_t operator()(const ::acn::Acn<N, Int>& a) const noexcept {
        return static_cast<std::size_t>(::acn::hash_value(a));
    }
};

template <class Int>
struct hash<::acn::AcnDyn<Int>> {
    std::size_t operator()(const ::acn::AcnDyn<Int>& a) const noexcept {
        return static_cast<std::size_t>(::acn::hash_value(a));
    }
};

}  // namespace std
