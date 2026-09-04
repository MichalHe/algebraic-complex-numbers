#pragma once

// BigInt: an RAII wrapper over GMP's mpz_t, conforming to the same
// IntBackend shape as Checked<Word> (see concepts.hpp), so Acn<N, BigInt>
// and AcnDyn<BigInt> are drop-in fallbacks when a Checked<i64> computation
// overflows (DESIGN.md §3.4's "two-tier execution"). mpz_t is never exposed
// in the public API (the prototype leaked mpz_t& out of at(), which is
// exactly what this wrapper exists to avoid).
//
// Requires ACN_WITH_GMP (the CMake option links GMP and defines this).

#include <compare>
#include <cstddef>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <gmp.h>

#include "acn/config.hpp"
#include "acn/hash_mix.hpp"
#include "acn/types.hpp"

namespace acn {

/// Arbitrary-precision integer backend (GMP mpz_t RAII wrapper);
/// IntBackend-conforming, drop-in for Checked<Word> when it overflows.
class BigInt {
    mpz_t value_;

public:
    using word_type = i64;

    BigInt() { mpz_init(value_); }
    BigInt(i64 v) { mpz_init_set_si(value_, v); }

    BigInt(const BigInt& other) { mpz_init_set(value_, other.value_); }

    BigInt(BigInt&& other) noexcept {
        // Standard GMP move idiom: steal the struct's contents, then
        // re-init the source so its destructor still has something valid
        // (and cheap) to mpz_clear.
        std::memcpy(value_, other.value_, sizeof(mpz_t));
        mpz_init(other.value_);
    }

    BigInt& operator=(const BigInt& other) {
        if (this != &other)
            mpz_set(value_, other.value_);
        return *this;
    }

    BigInt& operator=(BigInt&& other) noexcept {
        if (this != &other) {
            mpz_swap(value_, other.value_);
        }
        return *this;
    }

    ~BigInt() { mpz_clear(value_); }

    static bool overflow_seen() { return false; }
    static void clear_overflow() {}
    static void mark_overflow() { ACN_ASSERT(false && "BigInt arithmetic cannot overflow"); }

    friend BigInt operator+(const BigInt& a, const BigInt& b) {
        BigInt result;
        mpz_add(result.value_, a.value_, b.value_);
        return result;
    }

    friend BigInt operator-(const BigInt& a, const BigInt& b) {
        BigInt result;
        mpz_sub(result.value_, a.value_, b.value_);
        return result;
    }

    friend BigInt operator*(const BigInt& a, const BigInt& b) {
        BigInt result;
        mpz_mul(result.value_, a.value_, b.value_);
        return result;
    }

    BigInt operator-() const {
        BigInt result;
        mpz_neg(result.value_, value_);
        return result;
    }

    BigInt& operator+=(const BigInt& other) { return *this = *this + other; }
    BigInt& operator-=(const BigInt& other) { return *this = *this - other; }
    BigInt& operator*=(const BigInt& other) { return *this = *this * other; }

    friend bool operator==(const BigInt& a, const BigInt& b) { return mpz_cmp(a.value_, b.value_) == 0; }

    friend std::strong_ordering operator<=>(const BigInt& a, const BigInt& b) {
        int cmp = mpz_cmp(a.value_, b.value_);
        if (cmp < 0)
            return std::strong_ordering::less;
        if (cmp > 0)
            return std::strong_ordering::greater;
        return std::strong_ordering::equal;
    }

    bool is_zero() const { return mpz_sgn(value_) == 0; }
    bool is_even() const { return mpz_even_p(value_) != 0; }

    // Requires is_even(). Exact, so the rounding direction of mpz_tdiv_q_2exp
    // cannot matter.
    BigInt halve_exact() const {
        ACN_ASSERT(is_even());
        BigInt result;
        mpz_tdiv_q_2exp(result.value_, value_, 1);
        return result;
    }

    BigInt shift_left(int amount) const {
        ACN_ASSERT(amount >= 0);
        BigInt result;
        mpz_mul_2exp(result.value_, value_, static_cast<mp_bitcnt_t>(amount));
        return result;
    }

    // GMP considers negative operands in an infinite two's-complement
    // representation for scan1, so this is correct (and matches
    // Checked<Word>::ctz's contract) for negative values too.
    int ctz() const {
        ACN_ASSERT(!is_zero());
        return static_cast<int>(mpz_scan1(value_, 0));
    }

    // Bit-length of the magnitude (mpz_sizeinbase's own convention: 1 for
    // zero, ignores sign otherwise). Used by echelon.hpp's coefficient-
    // growth regression test to check fraction-free elimination doesn't
    // blow up exponentially; not needed by any exact-arithmetic path.
    std::size_t bit_length() const { return mpz_sizeinbase(value_, 2); }

    // General exact division (used by divide.hpp).
    std::optional<BigInt> divide_exact(const BigInt& divisor) const {
        ACN_ASSERT(!divisor.is_zero());
        if (mpz_divisible_p(value_, divisor.value_) == 0)
            return std::nullopt;
        BigInt result;
        mpz_divexact(result.value_, value_, divisor.value_);
        return result;
    }

    double to_double() const { return mpz_get_d(value_); }

    // For the promotion/narrowing driver (convert.hpp): whether this value
    // fits in an i64, and the (unchecked) conversion. `long` is 64 bits on
    // the platforms this library targets, so mpz_fits_slong_p is the right
    // fixed-width test.
    bool fits_i64() const { return mpz_fits_slong_p(value_) != 0; }
    i64 to_i64_unchecked() const {
        ACN_ASSERT(fits_i64());
        return static_cast<i64>(mpz_get_si(value_));
    }

    // Decimal text conversion for format.hpp, via GMP's own base-10
    // conversion (never mpz_get_si, which the prototype used and which
    // silently truncates for values outside a machine word).
    std::string to_string() const {
        char* buf = mpz_get_str(nullptr, 10, value_);
        std::string result(buf);
        // GMP's mpz_get_str allocates with its configured allocator
        // (the default malloc unless the caller overrides it), so a
        // matching free() is correct rather than freeing through the
        // library's own free-function hook.
        void (*freefn)(void*, std::size_t);
        mp_get_memory_functions(nullptr, nullptr, &freefn);
        freefn(buf, std::strlen(buf) + 1);
        return result;
    }

    static std::optional<BigInt> parse(std::string_view text) {
        if (text.empty())
            return std::nullopt;
        std::string owned(text);
        BigInt result;
        if (mpz_set_str(result.value_, owned.c_str(), 10) != 0)
            return std::nullopt;
        return result;
    }

    u64 hash_component() const {
        u64 h = mix64(static_cast<u64>(mpz_sgn(value_)) + 1);
        std::size_t limb_count = static_cast<std::size_t>(mpz_size(value_));
        for (std::size_t i = 0; i < limb_count; ++i)
            h = mix64(h ^ mix64(static_cast<u64>(mpz_getlimbn(value_, static_cast<mp_size_t>(i)))));
        return h;
    }
};

}  // namespace acn
