#pragma once

#include <cassert>

#if !defined(ACN_DEBUG)
#if defined(NDEBUG)
#define ACN_DEBUG 0
#else
#define ACN_DEBUG 1
#endif
#endif

#if ACN_DEBUG
#define ACN_ASSERT(condition) assert(condition)
#else
#define ACN_ASSERT(condition) ((void)0)
#endif

#if !defined(ACN_STATS)
#define ACN_STATS 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ACN_INLINE inline __attribute__((always_inline))
#define ACN_ASSUME(condition) \
    do {                      \
        if (!(condition))     \
            __builtin_unreachable(); \
    } while (0)
#else
#define ACN_INLINE inline
#define ACN_ASSUME(condition) ((void)0)
#endif
