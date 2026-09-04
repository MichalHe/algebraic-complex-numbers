#pragma once

#include <cstdint>

namespace acn {

using i32 = std::int32_t;
using i64 = std::int64_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i128 = __int128;
using u128 = unsigned __int128;

#if defined(ACN_SCALE_EXP_64)
using ScaleExp = i64;
#else
using ScaleExp = i32;
#endif

enum class Status {
    ok,
    overflow,
    not_divisible,
    precision_mismatch,
};

}  // namespace acn
