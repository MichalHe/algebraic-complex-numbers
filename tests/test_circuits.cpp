// PLAN.md §9.4: end-to-end circuit-level tests built on top of gates.hpp
// and matrix.hpp. Three of PLAN.md's four listed checks are implemented
// here (H^{(x)q} amplitudes, a QFT-2 circuit against its closed form, a
// known circuit-equivalence pair, a known inequivalence pair correctly
// rejected via exact operator==). The fourth -- "checked by confirming
// the double path cannot [distinguish them]" -- is deliberately not
// attempted quantitatively: doing that honestly would need an amplitude
// difference small enough to plausibly fool a double-precision epsilon
// comparison (think 1e-9 or smaller), and the only way to shrink a
// T-gate-induced phase difference that far via H^{(x)q}-style superposition
// is roughly q ~ 60 qubits -- a 2^60-dimensional dense Matrix<T>, wildly
// infeasible to materialize. The inequivalence test below still shows
// exact arithmetic correctly rejecting a genuinely different circuit; it
// just doesn't also stage a numeric near-miss for double to fail.

#include <catch2/catch.hpp>

#include "acn/acn.hpp"

using acn::Acn;
using acn::CheckedI64;
using acn::Matrix;

namespace gates = acn::gates;

using Acn4 = Acn<4, CheckedI64>;
using Mat = Matrix<Acn4>;

namespace {

Mat basis_state(std::size_t dim, std::size_t index) {
    Mat v(dim, 1);
    v.at(index, 0) = Acn4::one();
    return v;
}

Acn4 pow_inv_sqrt2(int q) {
    Acn4 result = Acn4::one();
    Acn4 s = Acn4::inv_sqrt2();
    for (int i = 0; i < q; ++i)
        result = result * s;
    return result;
}

}  // namespace

TEST_CASE("H^{(x)q} |0...0> has every amplitude exactly (1/sqrt2)^q", "[circuits]") {
    for (int q = 1; q <= 3; ++q) {
        std::size_t dim = std::size_t{1} << q;
        Mat h_full = gates::H<4>();
        for (int i = 1; i < q; ++i)
            h_full = h_full.kron(gates::H<4>());

        auto state = h_full * basis_state(dim, 0);
        auto expected = pow_inv_sqrt2(q);
        for (std::size_t j = 0; j < dim; ++j)
            REQUIRE(state.at(j, 0) == expected);
    }
}

TEST_CASE("2-qubit QFT circuit matches the closed-form DFT matrix", "[circuits]") {
    // Standard n=2 QFT circuit (Nielsen & Chuang, adapted to this file's
    // qubit-0-is-MSB convention): H on qubit 0, a controlled-phase(pi/2)
    // between the two qubits, H on qubit 1, then swap qubits 0 and 1.
    auto id2 = Mat::identity(2);
    auto h_qubit0 = gates::H<4>().kron(id2);  // H (x) I
    auto h_qubit1 = id2.kron(gates::H<4>());  // I (x) H

    // Controlled-phase(pi/2) = diag(1,1,1,i): only |11> gets the extra
    // phase omega(N/2) = i. Not in gates.hpp (that only has the
    // single-qubit phase<K>()), so built directly here.
    Mat cphase = Mat::identity(4);
    cphase.at(3, 3) = Acn4::imag();

    auto swap = gates::SWAP<4>();

    Mat qft_circuit = swap * h_qubit1 * cphase * h_qubit0;

    // Closed form: QFT[j][k] = (1/sqrt2)^2 * omega(2*j*k), which for
    // ring size N=4 gives exactly the standard 4-point DFT matrix (values
    // 1, i, -1, -i), derived in PROGRESS.md's session notes and checked
    // here entry-by-entry against the circuit above.
    Mat qft_closed_form(4, 4);
    Acn4 scale = pow_inv_sqrt2(2);
    for (std::size_t j = 0; j < 4; ++j)
        for (std::size_t k = 0; k < 4; ++k)
            qft_closed_form.at(j, k) = scale * Acn4::omega(static_cast<long long>(2 * j * k));

    REQUIRE(qft_circuit == qft_closed_form);
    REQUIRE(qft_circuit * qft_circuit.adjoint() == Mat::identity(4));
}

TEST_CASE("a known circuit equivalence: CNOT == (I(x)H) CZ (I(x)H)", "[circuits]") {
    auto id2 = Mat::identity(2);
    auto h_on_target = id2.kron(gates::H<4>());
    auto conjugated_cz = h_on_target * gates::CZ<4>() * h_on_target;
    REQUIRE(conjugated_cz == gates::CNOT<4>());
}

TEST_CASE("a known inequivalence: an uncancelled T gate is correctly rejected by exact ==", "[circuits]") {
    // Two circuits that would be equal if a stray T were actually
    // cancelled by a T_dag, but aren't: T*T_dag == I (already covered in
    // test_gates.cpp), whereas T*T == S != I. Exact equality correctly
    // distinguishes S from I with no tolerance/epsilon involved.
    auto t = gates::T<4>();
    auto s = gates::S<4>();
    auto id = Mat::identity(2);

    REQUIRE(t * t == s);
    REQUIRE_FALSE(t * t == id);

    // Embedded in a slightly larger circuit (qubit 0 of a 2-qubit system)
    // to show the distinction survives being embedded via kron, not just
    // in the bare 2x2 case.
    auto lhs = (t * t).kron(id);
    auto rhs = id.kron(id);
    REQUIRE_FALSE(lhs == rhs);
}
