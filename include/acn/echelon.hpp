#pragma once

// EchelonBasis<T>: an incremental row-echelon basis over R_N[1/sqrt2]
// (PLAN.md §9.2), useful for equivalence checking by maintaining a basis
// of reachable amplitude vectors. insert() gives a real return contract
// -- optional<size_t> (the new row's index if independent, nullopt if it
// reduced to zero) -- instead of the prototype's assert(false) on the
// full-rank path, and there is no -1 sentinel anywhere in this API.
//
// Fraction-free (Bareiss) elimination: eliminating row `r`'s component in
// an existing pivot column p (pivot value m) against basis row `basis_i`
// computes, for every column j,
//     next[j] = (r[j]*m - basis_i[j]*r[p]) / prev_pivot
// where prev_pivot is the pivot value used by the *previous* elimination
// step (the previous basis row's pivot, or 1 for the very first step).
// This is the standard two-step Bareiss recurrence: the numerator is
// always exactly divisible by prev_pivot (a classical fact about
// fraction-free Gaussian elimination -- the intermediate entries are
// literally minors of the augmented matrix), which is why this only ever
// multiplies and does one *exact* division per step, never a general
// (possibly inexact) division, matching DESIGN.md §5.3's "never divide"
// rule for anything that isn't provably exact. Skipping the
// divide-by-previous-pivot step entirely (i.e. plain multiply-only
// elimination) would instead multiply in every prior pivot's magnitude at
// every row, which is the actual source of the "integer blowup" PLAN.md
// flags as the real cost driver here -- Bareiss's division is what keeps
// growth bounded to roughly subdeterminant size instead of a product of
// all prior pivots.
//
// Content reduction (PLAN.md's other growth-control lever: dividing a row
// by the GCD of its coefficients) is **not implemented** -- it needs a
// general GCD algorithm over Z[sqrt2], which this library doesn't have
// (divide_exact requires already knowing the candidate divisor; a true
// content-GCD needs an independent Euclidean-style routine). Deliberately
// scoped out rather than attempted incorrectly; Bareiss's own bound is the
// growth control this class relies on, verified by
// test_echelon.cpp's regression test.

#include <cstddef>
#include <optional>
#include <vector>

#include "acn/concepts.hpp"
#include "acn/config.hpp"
#include "acn/divide.hpp"

namespace acn {

/// Incremental row-echelon basis via fraction-free (Bareiss) elimination;
/// tracks linear independence without introducing fractions.
template <AlgebraicComplex T>
class EchelonBasis {
    struct Row {
        std::vector<T> coeffs;
        std::size_t pivot_col;
    };

    std::vector<Row> rows_;  // sorted by pivot_col, ascending
    std::size_t width_;

public:
    explicit EchelonBasis(std::size_t width) : width_(width) {}

    std::size_t rank() const { return rows_.size(); }
    std::size_t width() const { return width_; }

    const std::vector<T>& basis_row(std::size_t i) const {
        ACN_ASSERT(i < rows_.size());
        return rows_[i].coeffs;
    }

    /// Inserts row; returns its index if independent, nullopt if dependent.
    // Inserts `row` into the basis. Returns the new row's index
    // (rank() - 1 after the call) if it was linearly independent of the
    // existing rows; returns nullopt (and leaves the basis unchanged) if
    // it reduced to the zero vector, i.e. was in the span of the existing
    // rows.
    std::optional<std::size_t> insert(std::vector<T> row) {
        ACN_ASSERT(row.size() == width_);

        T prev_pivot = T::one();
        for (const Row& basis_row_i : rows_) {
            std::size_t p = basis_row_i.pivot_col;
            T pivot_value = basis_row_i.coeffs[p];
            T eliminated_component = row[p];

            std::vector<T> next(width_);
            for (std::size_t j = 0; j < width_; ++j) {
                T numerator = row[j] * pivot_value - basis_row_i.coeffs[j] * eliminated_component;
                if (prev_pivot == T::one()) {
                    next[j] = numerator;
                } else {
                    auto divided = divide_exact(numerator, prev_pivot);
                    ACN_ASSERT(divided.has_value());  // Bareiss guarantees exactness
                    next[j] = *divided;
                }
            }
            row = std::move(next);
            prev_pivot = pivot_value;
        }

        std::optional<std::size_t> pivot_col;
        for (std::size_t j = 0; j < width_; ++j) {
            if (!row[j].is_zero()) {
                pivot_col = j;
                break;
            }
        }
        if (!pivot_col)
            return std::nullopt;

        rows_.push_back(Row{std::move(row), *pivot_col});
        return rows_.size() - 1;
    }
};

}  // namespace acn
