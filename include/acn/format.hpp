#pragma once

// PLAN.md §7.4: a readable canonical text form plus a round-tripping parser,
// and a binary serialization for caching verification results. Works for
// any Int backend that supplies to_string()/parse() (Checked<Word>,
// BigInt), so it's usable with Acn<N,Int> and AcnDyn<Int> alike.
//
// Text form: "(c0, c1, ..., c_{n-1})/sqrt2^k", optionally followed by a
// "  # ~a+bi" numeric-approximation comment that operator<< appends and
// parse() ignores (it stops at the closing "^<k>", so the comment is not
// part of the round-trip contract -- it's for humans reading logs).
//
// Binary form: varint(n), zigzag-varint(k), then per coefficient a
// varint-prefixed decimal-ASCII string (reusing Int::to_string()/parse()
// rather than a machine-word-width encoding, since BigInt coefficients are
// not bounded to a fixed width).

#include <array>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "acn/acn_dynamic.hpp"
#include "acn/acn_static.hpp"
#include "acn/types.hpp"

namespace acn {

namespace format_detail {

inline void write_varint(std::vector<std::uint8_t>& out, std::uint64_t v) {
    while (v >= 0x80) {
        out.push_back(static_cast<std::uint8_t>(v) | 0x80);
        v >>= 7;
    }
    out.push_back(static_cast<std::uint8_t>(v));
}

inline bool read_varint(const std::vector<std::uint8_t>& in, std::size_t& pos, std::uint64_t& out) {
    out = 0;
    int shift = 0;
    while (true) {
        if (pos >= in.size() || shift > 63)
            return false;
        std::uint8_t byte = in[pos++];
        out |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
        if ((byte & 0x80) == 0)
            return true;
        shift += 7;
    }
}

inline void write_zigzag(std::vector<std::uint8_t>& out, std::int64_t v) {
    write_varint(out, (static_cast<std::uint64_t>(v) << 1) ^ static_cast<std::uint64_t>(v >> 63));
}

inline bool read_zigzag(const std::vector<std::uint8_t>& in, std::size_t& pos, std::int64_t& out) {
    std::uint64_t raw = 0;
    if (!read_varint(in, pos, raw))
        return false;
    out = static_cast<std::int64_t>(raw >> 1) ^ -static_cast<std::int64_t>(raw & 1);
    return true;
}

struct ParsedFields {
    std::vector<std::string> coeff_tokens;
    ScaleExp k = 0;
};

// Parses the "(c0, c1, ..., c_{n-1})/sqrt2^k" prefix of text; anything after
// the exponent (e.g. an operator<<-appended comment) is ignored.
inline std::optional<ParsedFields> parse_fields(std::string_view text) {
    std::size_t i = 0;
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])))
        ++i;
    if (i >= text.size() || text[i] != '(')
        return std::nullopt;
    ++i;

    ParsedFields result;
    std::string current;
    bool closed = false;
    while (i < text.size()) {
        char ch = text[i];
        if (ch == ')') {
            result.coeff_tokens.push_back(current);
            ++i;
            closed = true;
            break;
        }
        if (ch == ',') {
            result.coeff_tokens.push_back(current);
            current.clear();
            ++i;
            while (i < text.size() && text[i] == ' ')
                ++i;
            continue;
        }
        current.push_back(ch);
        ++i;
    }
    if (!closed || result.coeff_tokens.empty())
        return std::nullopt;

    constexpr std::string_view marker = "/sqrt2^";
    if (text.substr(i, marker.size()) != marker)
        return std::nullopt;
    i += marker.size();

    std::size_t start = i;
    if (i < text.size() && text[i] == '-')
        ++i;
    while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i])))
        ++i;
    if (i == start || (i == start + 1 && text[start] == '-'))
        return std::nullopt;

    long long k_val = 0;
    auto [ptr, ec] = std::from_chars(text.data() + start, text.data() + i, k_val);
    if (ec != std::errc() || ptr != text.data() + i)
        return std::nullopt;
    result.k = static_cast<ScaleExp>(k_val);
    return result;
}

}  // namespace format_detail

/// Writes "(c0,...,c_{n-1})/sqrt2^k  # ~a+bi"; round-trips through parse().
template <std::size_t N, class Int>
std::ostream& operator<<(std::ostream& os, const Acn<N, Int>& value) {
    os << '(';
    auto c = value.coeffs();
    for (std::size_t j = 0; j < N; ++j) {
        if (j)
            os << ", ";
        os << c[j].to_string();
    }
    os << ")/sqrt2^" << value.scale_exp();
    auto approx = value.to_complex();
    os << "  # ~" << approx.real() << (approx.imag() < 0.0 ? "" : "+") << approx.imag() << 'i';
    return os;
}

/// Writes "(c0,...,c_{n-1})/sqrt2^k  # ~a+bi"; round-trips through parse_dyn().
template <class Int>
std::ostream& operator<<(std::ostream& os, const AcnDyn<Int>& value) {
    os << '(';
    auto c = value.coeffs();
    for (std::size_t j = 0; j < c.size(); ++j) {
        if (j)
            os << ", ";
        os << c[j].to_string();
    }
    os << ")/sqrt2^" << value.scale_exp();
    auto approx = value.to_complex();
    os << "  # ~" << approx.real() << (approx.imag() < 0.0 ? "" : "+") << approx.imag() << 'i';
    return os;
}

/// Inverse of operator<<; nullopt on malformed text or wrong precision.
template <std::size_t N, class Int>
std::optional<Acn<N, Int>> parse(std::string_view text) {
    auto fields = format_detail::parse_fields(text);
    if (!fields || fields->coeff_tokens.size() != N)
        return std::nullopt;
    std::array<Int, N> coeffs{};
    for (std::size_t j = 0; j < N; ++j) {
        auto v = Int::parse(fields->coeff_tokens[j]);
        if (!v)
            return std::nullopt;
        coeffs[j] = *v;
    }
    return Acn<N, Int>::from_coeffs(std::span<const Int>(coeffs), fields->k);
}

/// Inverse of operator<< for AcnDyn; nullopt on malformed text.
template <class Int>
std::optional<AcnDyn<Int>> parse_dyn(std::string_view text) {
    auto fields = format_detail::parse_fields(text);
    if (!fields)
        return std::nullopt;
    std::vector<Int> coeffs(fields->coeff_tokens.size());
    for (std::size_t j = 0; j < coeffs.size(); ++j) {
        auto v = Int::parse(fields->coeff_tokens[j]);
        if (!v)
            return std::nullopt;
        coeffs[j] = *v;
    }
    return AcnDyn<Int>::from_coeffs(std::span<const Int>(coeffs), fields->k);
}

/// Binary encoding: varint(N), zigzag-varint(k), varint-prefixed decimal
/// coefficients. See deserialize().
template <std::size_t N, class Int>
std::vector<std::uint8_t> serialize(const Acn<N, Int>& value) {
    std::vector<std::uint8_t> out;
    format_detail::write_varint(out, static_cast<std::uint64_t>(N));
    format_detail::write_zigzag(out, static_cast<std::int64_t>(value.scale_exp()));
    auto c = value.coeffs();
    for (std::size_t j = 0; j < N; ++j) {
        std::string text = c[j].to_string();
        format_detail::write_varint(out, static_cast<std::uint64_t>(text.size()));
        out.insert(out.end(), text.begin(), text.end());
    }
    return out;
}

/// Inverse of serialize(); nullopt on truncated/malformed/mismatched-N input.
template <std::size_t N, class Int>
std::optional<Acn<N, Int>> deserialize(const std::vector<std::uint8_t>& bytes) {
    std::size_t pos = 0;
    std::uint64_t n = 0;
    if (!format_detail::read_varint(bytes, pos, n) || n != N)
        return std::nullopt;
    std::int64_t k = 0;
    if (!format_detail::read_zigzag(bytes, pos, k))
        return std::nullopt;
    std::array<Int, N> coeffs{};
    for (std::size_t j = 0; j < N; ++j) {
        std::uint64_t len = 0;
        if (!format_detail::read_varint(bytes, pos, len))
            return std::nullopt;
        if (pos + len > bytes.size())
            return std::nullopt;
        std::string text(bytes.begin() + static_cast<std::ptrdiff_t>(pos),
                          bytes.begin() + static_cast<std::ptrdiff_t>(pos + len));
        pos += len;
        auto v = Int::parse(text);
        if (!v)
            return std::nullopt;
        coeffs[j] = *v;
    }
    return Acn<N, Int>::from_coeffs(std::span<const Int>(coeffs), static_cast<ScaleExp>(k));
}

/// Binary encoding for AcnDyn; see serialize()/deserialize_dyn().
template <class Int>
std::vector<std::uint8_t> serialize_dyn(const AcnDyn<Int>& value) {
    std::vector<std::uint8_t> out;
    format_detail::write_varint(out, static_cast<std::uint64_t>(value.precision()));
    format_detail::write_zigzag(out, static_cast<std::int64_t>(value.scale_exp()));
    auto c = value.coeffs();
    for (std::size_t j = 0; j < c.size(); ++j) {
        std::string text = c[j].to_string();
        format_detail::write_varint(out, static_cast<std::uint64_t>(text.size()));
        out.insert(out.end(), text.begin(), text.end());
    }
    return out;
}

/// Inverse of serialize_dyn(); nullopt on truncated/malformed input.
template <class Int>
std::optional<AcnDyn<Int>> deserialize_dyn(const std::vector<std::uint8_t>& bytes) {
    std::size_t pos = 0;
    std::uint64_t n = 0;
    if (!format_detail::read_varint(bytes, pos, n))
        return std::nullopt;
    std::int64_t k = 0;
    if (!format_detail::read_zigzag(bytes, pos, k))
        return std::nullopt;
    std::vector<Int> coeffs(static_cast<std::size_t>(n));
    for (std::size_t j = 0; j < coeffs.size(); ++j) {
        std::uint64_t len = 0;
        if (!format_detail::read_varint(bytes, pos, len))
            return std::nullopt;
        if (pos + len > bytes.size())
            return std::nullopt;
        std::string text(bytes.begin() + static_cast<std::ptrdiff_t>(pos),
                          bytes.begin() + static_cast<std::ptrdiff_t>(pos + len));
        pos += len;
        auto v = Int::parse(text);
        if (!v)
            return std::nullopt;
        coeffs[j] = *v;
    }
    return AcnDyn<Int>::from_coeffs(std::span<const Int>(coeffs), static_cast<ScaleExp>(k));
}

}  // namespace acn
