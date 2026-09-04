#pragma once

// Matrix<T>: dense, row-major, generic over AlgebraicComplex T (DESIGN.md
// §9). Owns its storage in a plain std::vector<T> -- there is no external
// "zero" pointer of the kind the prototype used (DESIGN.md §2.2 flags
// ComplexMatrix::zero as fragile and not copy-safe), so copy/move/
// assignment are all the compiler-generated defaults and are correct by
// construction, not by careful hand-written bookkeeping.
//
// operator* accumulates each output cell with repeated fma_into calls (one
// canonicalize per term), not the single-canonicalize-per-cell
// "fma_defer + one flush" PLAN.md §4.1/§9.1 describes -- that API doesn't
// exist yet (see PROGRESS.md). Documented here as the concrete performance
// gap versus the design; correctness is unaffected.

#include <cstddef>
#include <span>
#include <vector>

#include "acn/concepts.hpp"
#include "acn/config.hpp"

namespace acn {

/// Dense row-major matrix over an AlgebraicComplex T; owns its storage.
template <AlgebraicComplex T>
class Matrix {
    std::vector<T> data_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;

public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols) : data_(rows * cols, T::zero()), rows_(rows), cols_(cols) {}

    /// n x n identity.
    static Matrix identity(std::size_t n) {
        Matrix result(n, n);
        for (std::size_t i = 0; i < n; ++i)
            result.at(i, i) = T::one();
        return result;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    T& at(std::size_t r, std::size_t c) {
        ACN_ASSERT(r < rows_ && c < cols_);
        return data_[r * cols_ + c];
    }
    const T& at(std::size_t r, std::size_t c) const {
        ACN_ASSERT(r < rows_ && c < cols_);
        return data_[r * cols_ + c];
    }

    std::span<T> row(std::size_t r) {
        ACN_ASSERT(r < rows_);
        return std::span<T>(data_.data() + r * cols_, cols_);
    }
    std::span<const T> row(std::size_t r) const {
        ACN_ASSERT(r < rows_);
        return std::span<const T>(data_.data() + r * cols_, cols_);
    }

    friend bool operator==(const Matrix& a, const Matrix& b) {
        if (a.rows_ != b.rows_ || a.cols_ != b.cols_)
            return false;
        for (std::size_t i = 0; i < a.data_.size(); ++i)
            if (!(a.data_[i] == b.data_[i]))
                return false;
        return true;
    }

    Matrix operator+(const Matrix& other) const {
        ACN_ASSERT(rows_ == other.rows_ && cols_ == other.cols_);
        Matrix result(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] + other.data_[i];
        return result;
    }

    /// Elementwise scalar multiply.
    Matrix operator*(const T& scalar) const {
        Matrix result(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] * scalar;
        return result;
    }

    /// Matrix product; cols() must equal other.rows().
    Matrix operator*(const Matrix& other) const {
        ACN_ASSERT(cols_ == other.rows_);
        Matrix result(rows_, other.cols_);
        for (std::size_t r = 0; r < rows_; ++r) {
            for (std::size_t c = 0; c < other.cols_; ++c) {
                T acc = T::zero();
                for (std::size_t k = 0; k < cols_; ++k)
                    fma_into(acc, at(r, k), other.at(k, c));
                result.at(r, c) = acc;
            }
        }
        return result;
    }

    /// Transpose.
    Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (std::size_t r = 0; r < rows_; ++r)
            for (std::size_t c = 0; c < cols_; ++c)
                result.at(c, r) = at(r, c);
        return result;
    }

    /// Conjugate transpose.
    Matrix adjoint() const {
        Matrix result(cols_, rows_);
        for (std::size_t r = 0; r < rows_; ++r)
            for (std::size_t c = 0; c < cols_; ++c)
                result.at(c, r) = at(r, c).conj();
        return result;
    }

    /// Kronecker (tensor) product.
    Matrix kron(const Matrix& other) const {
        Matrix result(rows_ * other.rows_, cols_ * other.cols_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j)
                for (std::size_t k = 0; k < other.rows_; ++k)
                    for (std::size_t l = 0; l < other.cols_; ++l)
                        result.at(i * other.rows_ + k, j * other.cols_ + l) = at(i, j) * other.at(k, l);
        return result;
    }

    /// Relabels qubit indices (see perm's meaning below); result shape is
    /// always (rows(), cols()), never swapped.
    // Relabels qubit indices: perm.size() == q, rows_ == 2^q, and cols_ is
    // either 1 (a state vector -- only the row/qubit index is permuted) or
    // rows_ (a square operator -- both row and column indices are, since
    // an operator's bra and ket both live in the same permuted qubit
    // space). perm[i] names which *original* qubit becomes qubit i in the
    // result: bit i of a permuted index is bit perm[i] of the original.
    //
    // result is always constructed as (rows_, cols_) -- the exact same
    // shape as `this` -- never (cols_, rows_). PLAN.md §9.1 flags a
    // prototype bug where the result was constructed with width/height
    // swapped; for a square operator that bug is invisible (both are
    // equal), so test_matrix.cpp specifically exercises the cols_==1
    // (state vector, generally non-square as a 2^q x 1 matrix) case,
    // which is exactly where a width/height swap would corrupt the shape
    // rather than just silently transposing a square result.
    Matrix apply_qubit_permutation(std::span<const std::size_t> perm) const {
        std::size_t q = perm.size();
        ACN_ASSERT((std::size_t{1} << q) == rows_);
        ACN_ASSERT(cols_ == 1 || cols_ == rows_);
        Matrix result(rows_, cols_);
        for (std::size_t a = 0; a < rows_; ++a) {
            std::size_t pa = permute_bits(a, perm);
            if (cols_ == 1) {
                result.at(pa, 0) = at(a, 0);
            } else {
                for (std::size_t b = 0; b < cols_; ++b) {
                    std::size_t pb = permute_bits(b, perm);
                    result.at(pa, pb) = at(a, b);
                }
            }
        }
        return result;
    }

private:
    static std::size_t permute_bits(std::size_t x, std::span<const std::size_t> perm) {
        std::size_t result = 0;
        for (std::size_t i = 0; i < perm.size(); ++i)
            result |= ((x >> perm[i]) & std::size_t{1}) << i;
        return result;
    }
};

}  // namespace acn
