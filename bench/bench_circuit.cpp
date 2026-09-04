// PLAN.md §8.1 bullet 4: a realistic driver -- H^q then a Clifford+T
// layer then a QFT-2 circuit (embedded on qubits 0,1) -- reporting time,
// peak coefficient bit-width, max |k|, canonicalization iterations per
// op, and bignum promotions. "This is the number that actually predicts
// verifier performance" (PLAN.md).
//
// Uses Acn<4, BigInt> throughout rather than Checked<i64>, so coefficient
// growth is measured directly (bit_length()) instead of overflowing --
// bignum_promotions is therefore always 0 here (nothing ever needs
// promoting, since BigInt is used from the start); it's still reported,
// as a real (if unexciting) data point, not omitted.
//
// General q-qubit QFT (needing a gate embedded between arbitrary qubit
// pairs i,j, not just a fixed position) would need a generic
// "embed a k-qubit gate at specific qubit positions" helper that doesn't
// exist yet (matrix.hpp only has kron() and apply_qubit_permutation()).
// Scoped down here to the QFT-2 circuit test_circuits.cpp already
// derived and verified, embedded on qubits 0,1 via kron with the
// identity on the rest -- still a genuine multi-gate-type composite
// circuit, just not a full recursive QFT-q.
//
// ACN_STATS is force-enabled for this one translation unit (regardless
// of the project's global ACN_STATS build option) so canonicalization
// iteration counts are actually available to report.
#define ACN_STATS 1

#include <array>
#include <cstdio>
#include <random>

#include "acn/acn.hpp"
#include "bench/support/micro_bench.hpp"

using namespace acn::bench;
using acn::Acn;
using acn::BigInt;
using acn::Matrix;
using acn::ScaleExp;

namespace gates = acn::gates;

namespace {

using Acn4 = Acn<4, BigInt>;
using Mat = Matrix<Acn4>;

Mat basis_state(std::size_t dim) {
    Mat v(dim, 1);
    v.at(0, 0) = Acn4::one();
    return v;
}

// QFT-2 circuit, exactly test_circuits.cpp's derivation (H, controlled-
// phase(pi/2), H, swap), embedded on qubits (0,1) of a q-qubit system via
// kron with the identity on the remaining q-2 qubits.
Mat qft2_embedded(std::size_t q) {
    auto id2 = Mat::identity(2);
    auto h_qubit0 = gates::H<4, BigInt>().kron(id2);
    auto h_qubit1 = id2.kron(gates::H<4, BigInt>());
    Mat cphase = Mat::identity(4);
    cphase.at(3, 3) = Acn4::imag();
    auto swap = gates::SWAP<4, BigInt>();
    Mat qft2 = swap * h_qubit1 * cphase * h_qubit0;

    if (q == 2)
        return qft2;
    return qft2.kron(Mat::identity(std::size_t{1} << (q - 2)));
}

Mat h_power(std::size_t q) {
    auto h = gates::H<4, BigInt>();
    Mat result = h;
    for (std::size_t i = 1; i < q; ++i)
        result = result.kron(h);
    return result;
}

// T on qubit 0, S on qubit 1 (or just T on qubit 0 for q==1), embedded via
// kron with identity on the remaining qubits.
Mat clifford_t_layer(std::size_t q) {
    auto t = gates::T<4, BigInt>();
    if (q == 1)
        return t;
    auto s = gates::S<4, BigInt>();
    Mat layer = t.kron(s);
    if (q > 2)
        layer = layer.kron(Mat::identity(std::size_t{1} << (q - 2)));
    return layer;
}

void bench_circuit_size(std::size_t q) {
    std::size_t dim = std::size_t{1} << q;
    acn::stats::counters().reset();

    auto start = basis_state(dim);
    auto h_layer = h_power(q);
    auto ct_layer = clifford_t_layer(q);
    Mat qft = (q >= 2) ? qft2_embedded(q) : Mat::identity(dim);

    auto result = run(
        "circuit q=" + std::to_string(q),
        [&](std::uint64_t iters) {
            for (std::uint64_t i = 0; i < iters; ++i) {
                auto state = h_layer * start;
                state = ct_layer * state;
                state = qft * state;
                do_not_optimize(state);
            }
        },
        50.0);
    print_result(result);

    // Run once more, uninstrumented by the timing loop, purely to
    // capture the reported structural metrics from a single application.
    acn::stats::counters().reset();
    auto state = h_layer * start;
    state = ct_layer * state;
    state = qft * state;

    ScaleExp max_abs_k = 0;
    std::size_t max_bits = 0;
    for (std::size_t r = 0; r < dim; ++r) {
        auto k = state.at(r, 0).scale_exp();
        max_abs_k = std::max(max_abs_k, static_cast<ScaleExp>(k < 0 ? -k : k));
        for (const auto& coeff : state.at(r, 0).coeffs())
            max_bits = std::max(max_bits, coeff.bit_length());
    }

    std::printf(
        "  q=%zu: max|k|=%d  max_coeff_bits=%zu  canonicalize_calls=%llu  "
        "canonicalize_iterations=%llu  mul_by_sqrt2_calls=%llu  bignum_promotions=%llu\n",
        q, static_cast<int>(max_abs_k), max_bits,
        static_cast<unsigned long long>(acn::stats::counters().canonicalize_calls),
        static_cast<unsigned long long>(acn::stats::counters().canonicalize_iterations),
        static_cast<unsigned long long>(acn::stats::counters().mul_by_sqrt2_calls),
        static_cast<unsigned long long>(acn::stats::counters().bignum_promotions));
}

}  // namespace

int main() {
    for (std::size_t q = 1; q <= 4; ++q)
        bench_circuit_size(q);
    return 0;
}
