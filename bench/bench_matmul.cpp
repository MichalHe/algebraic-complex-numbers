// PLAN.md §8.1 bullet 2: dense 2^q x 2^q products for q = 1..6,
// operator-based vs fma_into-based, showing the temporaries/allocation
// cost.
//
// Matrix<T>::operator* is already fma_into-based (see matrix.hpp), so the
// "operator-based" comparison here is a second, naive implementation
// written directly in this file using +/* (which allocates/canonicalizes
// per term rather than once per output cell), not a second code path in
// the library itself.

#include <array>
#include <random>

#include "acn/acn.hpp"
#include "bench/support/micro_bench.hpp"

using namespace acn::bench;
using acn::Acn;
using acn::CheckedI64;
using acn::Matrix;
using acn::i64;

namespace {

using Acn4 = Acn<4, CheckedI64>;
using Mat = Matrix<Acn4>;

Mat random_matrix(std::mt19937_64& rng, std::size_t dim) {
    std::uniform_int_distribution<i64> dist(-3, 3);
    Mat m(dim, dim);
    for (std::size_t r = 0; r < dim; ++r)
        for (std::size_t c = 0; c < dim; ++c) {
            std::array<i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
            m.at(r, c) = Acn4::from_ints(coeffs, 0);
        }
    return m;
}

// Same product, computed with the naive +/* API (no fma_into): each term
// a.at(r,k)*b.at(k,c) is a full mul_into-then-canonicalize, and each
// accumulation acc + term is a full add_into-then-canonicalize -- one
// canonicalize per term instead of Matrix::operator*'s one per cell.
Mat naive_multiply(const Mat& a, const Mat& b) {
    std::size_t dim = a.rows();
    Mat result(dim, dim);
    for (std::size_t r = 0; r < dim; ++r) {
        for (std::size_t c = 0; c < dim; ++c) {
            Acn4 acc = Acn4::zero();
            for (std::size_t k = 0; k < dim; ++k)
                acc = acc + a.at(r, k) * b.at(k, c);
            result.at(r, c) = acc;
        }
    }
    return result;
}

template <std::size_t Q>
void bench_size() {
    constexpr std::size_t dim = std::size_t{1} << Q;
    std::mt19937_64 rng(3000 + Q);
    auto a = random_matrix(rng, dim);
    auto b = random_matrix(rng, dim);

    char name[96];
    std::snprintf(name, sizeof(name), "matmul_fma_into<dim=%zu>", dim);
    print_result(run(name, [&](std::uint64_t iters) {
        for (std::uint64_t i = 0; i < iters; ++i) {
            auto r = a * b;
            do_not_optimize(r);
        }
    }));

    std::snprintf(name, sizeof(name), "matmul_naive_ops<dim=%zu>", dim);
    print_result(run(name, [&](std::uint64_t iters) {
        for (std::uint64_t i = 0; i < iters; ++i) {
            auto r = naive_multiply(a, b);
            do_not_optimize(r);
        }
    }));
}

}  // namespace

int main() {
    bench_size<1>();
    bench_size<2>();
    bench_size<3>();
    bench_size<4>();
    bench_size<5>();
    bench_size<6>();
    return 0;
}
