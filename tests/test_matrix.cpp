#include <random>

#include <catch2/catch.hpp>

#include "acn/acn.hpp"

using acn::Acn;
using acn::CheckedI64;
using acn::i64;
using acn::Matrix;

using Acn4 = Acn<4, CheckedI64>;
using Mat = Matrix<Acn4>;

namespace {

Acn4 random_acn(std::mt19937_64& rng, i64 bound = 4) {
    std::uniform_int_distribution<i64> dist(-bound, bound);
    std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
    return Acn4::from_ints(coeffs, 0);
}

Mat random_matrix(std::mt19937_64& rng, std::size_t rows, std::size_t cols, i64 bound = 3) {
    Mat m(rows, cols);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            m.at(r, c) = random_acn(rng, bound);
    return m;
}

// Naive matrix multiply via plain +/*, independent of Matrix::operator*'s
// fma_into-based implementation.
Mat naive_multiply(const Mat& a, const Mat& b) {
    REQUIRE(a.cols() == b.rows());
    Mat result(a.rows(), b.cols());
    for (std::size_t r = 0; r < a.rows(); ++r) {
        for (std::size_t c = 0; c < b.cols(); ++c) {
            Acn4 acc = Acn4::zero();
            for (std::size_t k = 0; k < a.cols(); ++k)
                acc = acc + a.at(r, k) * b.at(k, c);
            result.at(r, c) = acc;
        }
    }
    return result;
}

}  // namespace

TEST_CASE("Matrix construction, at(), identity", "[matrix]") {
    Mat m(2, 3);
    REQUIRE(m.rows() == 2);
    REQUIRE(m.cols() == 3);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            REQUIRE(m.at(r, c).is_zero());

    auto id = Mat::identity(3);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            REQUIRE(id.at(r, c) == (r == c ? Acn4::one() : Acn4::zero()));
}

TEST_CASE("Matrix copy and move semantics are correct by construction", "[matrix]") {
    std::mt19937_64 rng(1);
    Mat original = random_matrix(rng, 3, 3);

    Mat copy = original;
    copy.at(0, 0) = copy.at(0, 0) + Acn4::one();
    REQUIRE_FALSE(copy.at(0, 0) == original.at(0, 0));  // deep copy, not aliased

    Mat moved = std::move(copy);
    REQUIRE(moved.rows() == 3);
    REQUIRE(moved.cols() == 3);

    Mat assigned;
    assigned = original;
    REQUIRE(assigned == original);
    assigned.at(1, 1) = assigned.at(1, 1) + Acn4::one();
    REQUIRE_FALSE(assigned == original);
}

TEST_CASE("operator* matches a naive +/* implementation", "[matrix]") {
    std::mt19937_64 rng(2);
    for (int trial = 0; trial < 30; ++trial) {
        auto a = random_matrix(rng, 3, 4);
        auto b = random_matrix(rng, 4, 2);
        REQUIRE(a * b == naive_multiply(a, b));
    }
}

TEST_CASE("identity is a multiplicative identity", "[matrix]") {
    std::mt19937_64 rng(3);
    auto a = random_matrix(rng, 4, 4);
    auto id = Mat::identity(4);
    REQUIRE(a * id == a);
    REQUIRE(id * a == a);
}

TEST_CASE("matrix multiplication is associative", "[matrix]") {
    std::mt19937_64 rng(4);
    auto a = random_matrix(rng, 2, 3);
    auto b = random_matrix(rng, 3, 3);
    auto c = random_matrix(rng, 3, 2);
    REQUIRE((a * b) * c == a * (b * c));
}

TEST_CASE("transpose is an involution and matches the definition", "[matrix]") {
    std::mt19937_64 rng(5);
    auto a = random_matrix(rng, 2, 3);
    auto t = a.transpose();
    REQUIRE(t.rows() == 3);
    REQUIRE(t.cols() == 2);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            REQUIRE(t.at(c, r) == a.at(r, c));
    REQUIRE(t.transpose() == a);
}

TEST_CASE("adjoint is conjugate-transpose and (A B)^dagger == B^dagger A^dagger", "[matrix]") {
    std::mt19937_64 rng(6);
    auto a = random_matrix(rng, 2, 3);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            REQUIRE(a.adjoint().at(c, r) == a.at(r, c).conj());

    auto b = random_matrix(rng, 3, 2);
    REQUIRE((a * b).adjoint() == b.adjoint() * a.adjoint());
}

TEST_CASE("a simple unitary built by hand satisfies U U^dagger == I", "[matrix]") {
    // The Hadamard-like matrix (1/sqrt2)[[1,1],[1,-1]] over Acn<4>, built
    // directly here (gates.hpp doesn't exist yet -- Phase 8/9.3).
    auto s = Acn4::inv_sqrt2();
    Mat h(2, 2);
    h.at(0, 0) = s;
    h.at(0, 1) = s;
    h.at(1, 0) = s;
    h.at(1, 1) = -s;

    REQUIRE(h * h.adjoint() == Mat::identity(2));
    REQUIRE(h.adjoint() * h == Mat::identity(2));
}

TEST_CASE("kron has the right shape and matches the definition", "[matrix]") {
    std::mt19937_64 rng(7);
    auto a = random_matrix(rng, 2, 2);
    auto b = random_matrix(rng, 2, 3);
    auto k = a.kron(b);
    REQUIRE(k.rows() == 4);
    REQUIRE(k.cols() == 6);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            for (std::size_t kk = 0; kk < 2; ++kk)
                for (std::size_t l = 0; l < 3; ++l)
                    REQUIRE(k.at(i * 2 + kk, j * 3 + l) == a.at(i, j) * b.at(kk, l));
}

TEST_CASE("kron satisfies the mixed-product property (A kron B)(C kron D) == (AC) kron (BD)", "[matrix]") {
    std::mt19937_64 rng(70);
    auto a = random_matrix(rng, 2, 2);
    auto b = random_matrix(rng, 2, 2);
    auto c = random_matrix(rng, 2, 2);
    auto d = random_matrix(rng, 2, 2);
    REQUIRE(a.kron(b) * c.kron(d) == (a * c).kron(b * d));
}

TEST_CASE("identity kron identity is the larger identity", "[matrix]") {
    auto id2 = Mat::identity(2);
    REQUIRE(id2.kron(id2) == Mat::identity(4));
}

TEST_CASE("apply_qubit_permutation on a square 2-qubit operator: identity permutation is a no-op",
          "[matrix]") {
    std::mt19937_64 rng(8);
    auto a = random_matrix(rng, 4, 4);
    std::array<std::size_t, 2> perm{0, 1};
    REQUIRE(a.apply_qubit_permutation(perm) == a);
}

TEST_CASE("apply_qubit_permutation never swaps width/height (state-vector case)", "[matrix]") {
    // A 2-qubit state vector is 4x1, not square -- exactly the shape where
    // PLAN.md's flagged prototype bug (constructing the result with
    // width/height swapped) would be observable rather than silently
    // invisible.
    Mat state(4, 1);
    state.at(0, 0) = Acn4::from_ints({1, 0, 0, 0}, 0);
    state.at(1, 0) = Acn4::from_ints({2, 0, 0, 0}, 0);
    state.at(2, 0) = Acn4::from_ints({3, 0, 0, 0}, 0);
    state.at(3, 0) = Acn4::from_ints({4, 0, 0, 0}, 0);

    std::array<std::size_t, 2> swap_perm{1, 0};  // swap qubit 0 and qubit 1
    auto permuted = state.apply_qubit_permutation(swap_perm);

    REQUIRE(permuted.rows() == 4);  // NOT 1 -- the exact bug this guards against
    REQUIRE(permuted.cols() == 1);

    // Swapping two qubits on a state vector swaps basis states |01> and
    // |10> (indices 1 and 2) and fixes |00>, |11> (indices 0, 3) -- the
    // standard 2-qubit SWAP permutation.
    REQUIRE(permuted.at(0, 0) == state.at(0, 0));
    REQUIRE(permuted.at(1, 0) == state.at(2, 0));
    REQUIRE(permuted.at(2, 0) == state.at(1, 0));
    REQUIRE(permuted.at(3, 0) == state.at(3, 0));
}

TEST_CASE("apply_qubit_permutation on a square operator matches conjugation by the same permutation",
          "[matrix]") {
    std::mt19937_64 rng(9);
    auto a = random_matrix(rng, 4, 4);
    std::array<std::size_t, 2> swap_perm{1, 0};
    auto permuted = a.apply_qubit_permutation(swap_perm);

    REQUIRE(permuted.rows() == 4);
    REQUIRE(permuted.cols() == 4);
    // Cross-check against the same index map used for the state-vector
    // test: (0,1,2,3) -> (0,2,1,3) on both row and column.
    std::array<std::size_t, 4> mapped{0, 2, 1, 3};
    for (std::size_t r = 0; r < 4; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            REQUIRE(permuted.at(mapped[r], mapped[c]) == a.at(r, c));
}
