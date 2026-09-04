#pragma once

// Concepts that let generic code (matrices, DD packages, tests) be written
// once against any backend/precision-binding pair. See DESIGN.md §3.3/§3.4.

#include <concepts>
#include <cstddef>

#include "acn/types.hpp"

namespace acn {

template <class T>
concept IntBackend = requires(T a, T b, int amount) {
    { T(std::declval<typename T::word_type>()) };
    { a + b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
    { a * b } -> std::same_as<T>;
    { -a } -> std::same_as<T>;
    { a == b } -> std::same_as<bool>;
    { a.is_zero() } -> std::same_as<bool>;
    { a.is_even() } -> std::same_as<bool>;
    { a.halve_exact() } -> std::same_as<T>;
    { a.shift_left(amount) } -> std::same_as<T>;
    { a.ctz() } -> std::same_as<int>;
    { T::overflow_seen() } -> std::same_as<bool>;
    { T::clear_overflow() };
};

template <class T>
concept AlgebraicComplex = requires(T a, T b, T& d, long long j, ScaleExp s) {
    { T::zero() } -> std::same_as<T>;
    { T::one() } -> std::same_as<T>;
    { a + b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
    { a * b } -> std::same_as<T>;
    { -a } -> std::same_as<T>;
    { a.is_zero() } -> std::same_as<bool>;
    { a == b } -> std::same_as<bool>;
    { a.conj() } -> std::same_as<T>;
    { a.mul_omega(j) } -> std::same_as<T>;
    { a.scale(s) } -> std::same_as<T>;
    { hash_value(a) } -> std::same_as<u64>;
    { a.to_complex() };
    { add_into(d, a, b) };
    { mul_into(d, a, b) };
    { fma_into(d, a, b) };
};

}  // namespace acn
