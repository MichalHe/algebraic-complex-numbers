// PLAN.md §8.1 bullet 3: insertion of 10^6 amplitudes into an
// unordered_set, measuring both throughput and collision behaviour of
// hash_value(). Uses std::unordered_set<Acn<N,Int>> directly, which picks
// up hash.hpp's std::hash<Acn<N,Int>> specialization.

#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <unordered_set>
#include <vector>

#include "acn/acn.hpp"

int main() {
    using Acn4 = acn::Acn<4, acn::CheckedI64>;
    constexpr std::size_t kCount = 1'000'000;

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<acn::i64> dist(-1000, 1000);

    std::vector<Acn4> values;
    values.reserve(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        std::array<acn::i64, 4> coeffs{dist(rng), dist(rng), dist(rng), dist(rng)};
        values.push_back(Acn4::from_ints(coeffs, 0));
    }

    std::unordered_set<Acn4> set;
    set.reserve(kCount);

    auto start = std::chrono::steady_clock::now();
    for (const auto& v : values)
        set.insert(v);
    double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    std::size_t max_bucket_size = 0;
    for (std::size_t b = 0; b < set.bucket_count(); ++b)
        max_bucket_size = std::max(max_bucket_size, set.bucket_size(b));

    std::printf("inserted %zu values (%zu unique) in %.2f ms  =>  %.2f ns/insert\n", kCount, set.size(), elapsed_ms,
                elapsed_ms * 1.0e6 / static_cast<double>(kCount));
    std::printf("bucket_count=%zu  load_factor=%.3f  max_bucket_size=%zu\n", set.bucket_count(), set.load_factor(),
                max_bucket_size);

    return 0;
}
