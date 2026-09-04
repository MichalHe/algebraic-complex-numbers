#pragma once

// PLAN.md §9.3: fixed gate matrices over Acn<N,Int>, as Matrix<Acn<N,Int>>
// values.
//
// Not literally `constexpr` per PLAN.md's sketch. Two independent
// blockers: (1) Acn<N,Int>'s own factories (zero/one/imag/omega/...) are
// not constexpr yet -- a gap documented since the Phase 0-3 session,
// caused by Checked<Word>'s overflow path touching a thread_local, which
// C++20 flatly disallows inside a constexpr function regardless of
// whether that branch is ever taken at compile time; (2) even with that
// fixed, Matrix<T> is std::vector-backed, and C++20's "no exposed
// pointers into transient constexpr allocation" rule means a
// *persistent* `constexpr Matrix<...> H = ...;` object still couldn't
// exist -- only a vector used and discarded entirely within one constant
// expression can. Reaching real constexpr gate matrices would need
// Acn<N,Int>'s factories fixed *and* a fixed-capacity (std::array-backed)
// small-matrix type instead of/alongside Matrix<T>. Neither is attempted
// here; these are ordinary functions, computed fresh (and cheaply -- at
// most 8x8) on every call.
//
// Qubit-ordering convention (no prototype exists in this repo to inherit
// one from, so this is fixed here, once): for a k-qubit gate, basis state
// index j corresponds to qubits q_0 q_1 ... q_{k-1} with q_0 the most
// significant bit of j (so CNOT's control is qubit 0, target is qubit 1;
// Toffoli's controls are qubits 0,1, target is qubit 2).

#include <cstddef>

#include "acn/acn_static.hpp"
#include "acn/int_backend.hpp"
#include "acn/matrix.hpp"
#include "acn/types.hpp"

namespace acn::gates {

/// Hadamard: (1/sqrt2)[[1,1],[1,-1]].
template <std::size_t N, class Int = CheckedI64>
Matrix<Acn<N, Int>> H() {
    static_assert(N >= 4, "H needs sqrt(2), which requires N >= 4");
    using A = Acn<N, Int>;
    Matrix<A> m(2, 2);
    A s = A::inv_sqrt2();
    m.at(0, 0) = s;
    m.at(0, 1) = s;
    m.at(1, 0) = s;
    m.at(1, 1) = -s;
    return m;
}

/// Pauli-X (bit flip).
template <std::size_t N, class Int = CheckedI64>
Matrix<Acn<N, Int>> X() {
    using A = Acn<N, Int>;
    Matrix<A> m(2, 2);
    m.at(0, 1) = A::one();
    m.at(1, 0) = A::one();
    return m;
}

/// Pauli-Y.
template <std::size_t N, class Int = CheckedI64>
Matrix<Acn<N, Int>> Y() {
    static_assert(N >= 2, "Y needs i, which requires N >= 2");
    using A = Acn<N, Int>;
    Matrix<A> m(2, 2);
    m.at(0, 1) = -A::imag();
    m.at(1, 0) = A::imag();
    return m;
}

/// Pauli-Z (phase flip).
template <std::size_t N, class Int = CheckedI64>
Matrix<Acn<N, Int>> Z() {
    using A = Acn<N, Int>;
    Matrix<A> m(2, 2);
    m.at(0, 0) = A::one();
    m.at(1, 1) = -A::one();
    return m;
}

/// Phase gate: diag(1, i).
template <std::size_t N, class Int = CheckedI64>
Matrix<Acn<N, Int>> S() {
    static_assert(N >= 2, "S needs i, which requires N >= 2");
    using A = Acn<N, Int>;
    Matrix<A> m(2, 2);
    m.at(0, 0) = A::one();
    m.at(1, 1) = A::imag();
    return m;
}

/// S^-1: diag(1, -i).
template <std::size_t N, class Int = CheckedI64>
Matrix<Acn<N, Int>> S_dag() {
    static_assert(N >= 2, "S_dag needs i, which requires N >= 2");
    using A = Acn<N, Int>;
    Matrix<A> m(2, 2);
    m.at(0, 0) = A::one();
    m.at(1, 1) = -A::imag();
    return m;
}

/// T gate: diag(1, e^{i*pi/4}).
// diag(1, omega(N/4)) = diag(1, e^{i*pi/4}). N/4 must be an integer for
// omega(N/4) to actually equal e^{i*pi/4} (rather than some other,
// unintended rotation), hence N % 4 == 0.
template <std::size_t N, class Int = CheckedI64>
Matrix<Acn<N, Int>> T() {
    static_assert(N >= 4 && N % 4 == 0, "T needs omega^(N/4) = e^{i*pi/4}, requires N a multiple of 4");
    using A = Acn<N, Int>;
    Matrix<A> m(2, 2);
    m.at(0, 0) = A::one();
    m.at(1, 1) = A::omega(static_cast<long long>(N / 4));
    return m;
}

/// T^-1: diag(1, e^{-i*pi/4}).
template <std::size_t N, class Int = CheckedI64>
Matrix<Acn<N, Int>> T_dag() {
    static_assert(N >= 4 && N % 4 == 0, "T_dag needs omega^(-N/4), requires N a multiple of 4");
    using A = Acn<N, Int>;
    Matrix<A> m(2, 2);
    m.at(0, 0) = A::one();
    m.at(1, 1) = A::omega(-static_cast<long long>(N / 4));
    return m;
}

/// Phase gate: diag(1, e^{i*2*pi/2^K}).
// diag(1, e^{i*2*pi/2^K}) = diag(1, omega(N / 2^(K-1))).
template <std::size_t N, unsigned K, class Int = CheckedI64>
Matrix<Acn<N, Int>> phase() {
    static_assert(K >= 1, "phase<K>() requires K >= 1");
    static_assert(N % (std::size_t{1} << (K - 1)) == 0, "N must be divisible by 2^(K-1) for phase<K>()");
    using A = Acn<N, Int>;
    Matrix<A> m(2, 2);
    m.at(0, 0) = A::one();
    m.at(1, 1) = A::omega(static_cast<long long>(N / (std::size_t{1} << (K - 1))));
    return m;
}

/// Z-rotation by 2*pi/2^M: diag(e^{-i*pi/2^M}, e^{i*pi/2^M}).
// Rz(2*pi/2^M) = diag(e^{-i*pi/2^M}, e^{i*pi/2^M}) (PLAN.md's convention:
// the two eigenvalues differ by the full angle 2*pi/2^M).
template <std::size_t N, unsigned M, class Int = CheckedI64>
Matrix<Acn<N, Int>> Rz() {
    static_assert(N % (std::size_t{1} << M) == 0, "N must be divisible by 2^M for Rz<M>()");
    using A = Acn<N, Int>;
    Matrix<A> m(2, 2);
    long long j = static_cast<long long>(N / (std::size_t{1} << M));
    m.at(0, 0) = A::omega(-j);
    m.at(1, 1) = A::omega(j);
    return m;
}

/// Controlled-NOT (control qubit 0, target qubit 1).
// Control is qubit 0, target is qubit 1: flips |10> (index 2) <-> |11>
// (index 3).
template <std::size_t N, class Int = CheckedI64>
Matrix<Acn<N, Int>> CNOT() {
    using A = Acn<N, Int>;
    Matrix<A> m = Matrix<A>::identity(4);
    m.at(2, 2) = A::zero();
    m.at(3, 3) = A::zero();
    m.at(2, 3) = A::one();
    m.at(3, 2) = A::one();
    return m;
}

/// Controlled-Z.
template <std::size_t N, class Int = CheckedI64>
Matrix<Acn<N, Int>> CZ() {
    using A = Acn<N, Int>;
    Matrix<A> m = Matrix<A>::identity(4);
    m.at(3, 3) = -A::one();
    return m;
}

/// Swaps two qubits.
// Swaps |01> (index 1) <-> |10> (index 2).
template <std::size_t N, class Int = CheckedI64>
Matrix<Acn<N, Int>> SWAP() {
    using A = Acn<N, Int>;
    Matrix<A> m = Matrix<A>::identity(4);
    m.at(1, 1) = A::zero();
    m.at(2, 2) = A::zero();
    m.at(1, 2) = A::one();
    m.at(2, 1) = A::one();
    return m;
}

/// Toffoli (CCNOT; controls qubits 0,1, target qubit 2).
// Controls are qubits 0,1, target is qubit 2: flips |110> (index 6) <->
// |111> (index 7).
template <std::size_t N, class Int = CheckedI64>
Matrix<Acn<N, Int>> Toffoli() {
    using A = Acn<N, Int>;
    Matrix<A> m = Matrix<A>::identity(8);
    m.at(6, 6) = A::zero();
    m.at(7, 7) = A::zero();
    m.at(6, 7) = A::one();
    m.at(7, 6) = A::one();
    return m;
}

}  // namespace acn::gates
