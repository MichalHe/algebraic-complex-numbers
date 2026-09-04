#pragma once

#include <algorithm>
#include <compare>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include "acn/config.hpp"
#include "acn/hash_mix.hpp"
#include "acn/stats.hpp"
#include "acn/types.hpp"

namespace acn {

// A machine-word integer wrapped with overflow-checked arithmetic. Overflow
// does not throw or abort; it sets a thread-local sticky flag that a driver
// polls at a checkpoint and reacts to (typically: restart the computation
// with a bignum backend). This keeps the checked type usable as a drop-in
// operand in expressions without threading Status through every operator.
template <class Word>
/// Machine-word integer with overflow-checked arithmetic; overflow sets a
/// thread-local sticky flag (see overflow_seen()) rather than trapping.
class Checked {
    static_assert(std::is_signed_v<Word>);

    Word word_ = 0;

    using Unsigned = std::make_unsigned_t<Word>;
    static constexpr int bit_width = static_cast<int>(sizeof(Word) * 8);

    static bool& overflow_flag() {
        thread_local bool flag = false;
        return flag;
    }

public:
    using word_type = Word;

    constexpr Checked() = default;
    constexpr Checked(Word value) : word_(value) {}

    constexpr Word value() const { return word_; }

    static bool overflow_seen() { return overflow_flag(); }
    static void clear_overflow() { overflow_flag() = false; }
    static void mark_overflow() {
        overflow_flag() = true;
        ACN_STAT_INC(overflow_flags_seen);
    }

    friend Checked operator+(Checked a, Checked b) {
        Word result;
        if (__builtin_add_overflow(a.word_, b.word_, &result))
            mark_overflow();
        return Checked(result);
    }

    friend Checked operator-(Checked a, Checked b) {
        Word result;
        if (__builtin_sub_overflow(a.word_, b.word_, &result))
            mark_overflow();
        return Checked(result);
    }

    friend Checked operator*(Checked a, Checked b) {
        Word result;
        if (__builtin_mul_overflow(a.word_, b.word_, &result))
            mark_overflow();
        return Checked(result);
    }

    Checked operator-() const {
        if (word_ == std::numeric_limits<Word>::min())
            mark_overflow();
        return Checked(static_cast<Word>(-word_));
    }

    Checked& operator+=(Checked other) { return *this = *this + other; }
    Checked& operator-=(Checked other) { return *this = *this - other; }
    Checked& operator*=(Checked other) { return *this = *this * other; }

    friend bool operator==(Checked a, Checked b) = default;
    friend std::strong_ordering operator<=>(Checked a, Checked b) = default;

    bool is_zero() const { return word_ == 0; }
    bool is_even() const { return (word_ & 1) == 0; }

    // Requires is_even(); C++20 guarantees arithmetic (sign-preserving)
    // right shift on signed integers, and division is exact here so the
    // rounding direction of >> cannot matter.
    Checked halve_exact() const {
        ACN_ASSERT(is_even());
        return Checked(static_cast<Word>(word_ >> 1));
    }

    Checked shift_left(int amount) const {
        ACN_ASSERT(amount >= 0);
        if (amount == 0)
            return *this;
        Unsigned magnitude = static_cast<Unsigned>(word_ < 0 ? -word_ : word_);
        int leading_zeros = 0;
        while (leading_zeros < bit_width &&
               ((magnitude >> (bit_width - 1 - leading_zeros)) & 1u) == 0)
            ++leading_zeros;
        if (magnitude != 0 && amount > leading_zeros - 1)
            mark_overflow();
        return Checked(static_cast<Word>(word_ << amount));
    }

    // Number of trailing zero bits of the two's-complement representation.
    // Equal for x and -x when x != 0, so the sign of word_ does not matter.
    int ctz() const {
        ACN_ASSERT(!is_zero());
        Unsigned bits = static_cast<Unsigned>(word_);
        int count = 0;
        while ((bits & 1u) == 0) {
            bits >>= 1;
            ++count;
        }
        return count;
    }

    // General exact division (used by divide.hpp): nullopt if divisor
    // does not evenly divide this value. Unlike halve_exact, the divisor
    // need not be 2.
    std::optional<Checked> divide_exact(Checked divisor) const {
        ACN_ASSERT(!divisor.is_zero());
        Word quotient = word_ / divisor.word_;
        Word remainder = word_ % divisor.word_;
        if (remainder != 0)
            return std::nullopt;
        return Checked(quotient);
    }

    double to_double() const { return static_cast<double>(word_); }

    u64 hash_component() const { return mix64(static_cast<u64>(word_)); }

    // Decimal text conversion for format.hpp. Written against plain +/-/%
    // on Word rather than std::to_chars/from_chars so it also works for
    // Word == i128, which those standard facilities don't support.
    std::string to_string() const {
        Unsigned magnitude = static_cast<Unsigned>(word_ < 0 ? -static_cast<Unsigned>(word_) : static_cast<Unsigned>(word_));
        std::string digits;
        if (magnitude == 0)
            digits = "0";
        else
            while (magnitude != 0) {
                digits.push_back(static_cast<char>('0' + static_cast<int>(magnitude % 10)));
                magnitude /= 10;
            }
        if (word_ < 0)
            digits.push_back('-');
        std::reverse(digits.begin(), digits.end());
        return digits;
    }

    static std::optional<Checked> parse(std::string_view text) {
        if (text.empty())
            return std::nullopt;
        bool negative = text.front() == '-';
        std::size_t start = negative ? 1 : 0;
        if (start >= text.size())
            return std::nullopt;
        Unsigned magnitude = 0;
        for (std::size_t i = start; i < text.size(); ++i) {
            char ch = text[i];
            if (ch < '0' || ch > '9')
                return std::nullopt;
            magnitude = static_cast<Unsigned>(magnitude * 10 + static_cast<Unsigned>(ch - '0'));
        }
        Word value = static_cast<Word>(magnitude);
        if (negative)
            value = static_cast<Word>(-value);
        return Checked(value);
    }
};

using CheckedI64 = Checked<i64>;
using CheckedI128 = Checked<i128>;

}  // namespace acn
