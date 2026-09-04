#pragma once

// The bit mixer used to turn raw coefficient/limb words into a hash. Lives
// in its own header (rather than hash.hpp) because int backends (Checked,
// BigInt) need it to implement their own hash_component(), and hash.hpp
// itself depends on the Acn types, which depend on the backends.

#include "acn/types.hpp"

namespace acn {

// splitmix64, a fast fixed-point mixer with good avalanche properties.
inline u64 mix64(u64 x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

}  // namespace acn
