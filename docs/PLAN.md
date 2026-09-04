# Implementation Plan

Companion to [DESIGN.md](DESIGN.md). That document says *what* the library is and
*why*; this one says *in what order it gets built, by whom-ever picks up the next
task, and how each step is known to be done*.

The rewrite happens **alongside** the prototype, not on top of it: new code lands in
`include/acn/` + `src/`, the prototype stays in `src/legacy/` until Phase 8 deletes it.
Nothing in the new tree may `#include` a legacy header.

---

## 0. Conventions fixed up front

These are decided now so that no phase has to relitigate them.

| Item | Decision |
| --- | --- |
| Language | C++20 (`CMAKE_CXX_STANDARD 20`, no compiler extensions). Concepts and `if constexpr` are used; nothing needs C++23. |
| Toolchain | GCC 15 / Clang 18+, both must build warning-clean with `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`. |
| Namespace | `acn`. No `using namespace` in any header (the prototype does this in `matrix.hpp` and `matrix_constructs.hpp`). |
| Headers | `include/acn/*.hpp`, `#pragma once`, self-contained (each compiles alone — enforced by a generated TU per header). |
| Integer typedefs | `acn::i32/i64/u32/u64/i128` in `acn/types.hpp`, defined from `<cstdint>`. The prototype's `basics.hpp` maps `s16`/`s32` to `int64_t`, which is a trap; that file is not carried over. |
| Index convention | Coefficient `j` is the multiplier of `omega^j`, `omega = e^(i*pi/n)`, `omega^n = -1`. Written down once, in `acn/ring.hpp`. |
| `sqrt(2)` | `sqrt2 = omega^(n/4) - omega^(3n/4)` for `n >= 4`. Single definition, `mul_by_sqrt2()`, used by every scale change. Requires `n >= 4`; `n in {1,2}` get specialized scale code (§2.4). |
| Scale semantics | `(c, k)` denotes `u / sqrt(2)^k`. Increasing `k` by 1 divides the value by `sqrt(2)`. Rescaling to a **larger** `k` multiplies the numerator by `sqrt(2)`. |
| Errors | No exceptions in the hot path. Overflow and partial operations return `std::optional` / a `Status` enum. `assert` for programmer errors (mismatched `n`, aliasing violations), active only in debug builds. |
| Tests | Catch2 v2 (`<catch2/catch.hpp>`, as installed on the target Fedora). One test binary per phase-scoped test file, all registered with CTest. |
| Style | `snake_case` functions, `PascalCase` types, members without prefixes, `m_`-free. Braces on the same line. Match DESIGN.md's naming exactly. |

**Definition of done for every phase:** code + tests + doc comments land together, CI is
green under debug, release, ASan/UBSan, the phase's acceptance criteria (listed per
phase) hold, and `docs/DESIGN.md` is amended if reality diverged from the design.

---

## 1. Target file layout

```
include/acn/
  types.hpp          # i32/i64/u32/u64/i128, ScaleExp, Status
  config.hpp         # ACN_DEBUG, ACN_STATS, assert macros, force-inline
  concepts.hpp       # AlgebraicComplex, IntBackend concepts
  int_backend.hpp    # Checked<i64>, Checked<i128> wrappers
  bigint.hpp         # BigInt: RAII mpz wrapper (opt-in, needs GMP)
  ring.hpp           # index/sign arithmetic on Z[x]/(x^n+1): rotation, negacyclic
                     # convolution, mul_by_sqrt2, conjugation. n as parameter, no
                     # scale, no canonicalization. The mathematical core.
  canonical.hpp      # canonicalize(), sqrt2-divisibility test, ctz bulk reduction
  acn_static.hpp     # Acn<N, Int>
  acn_dynamic.hpp    # AcnDyn<Int>, NumberContext, arena
  convert.hpp        # widen/narrow, to_complex, to_interval, sqrt2-view, predicates
  interval.hpp       # Interval<double> with directed rounding
  hash.hpp           # hash_value, std::hash specializations
  format.hpp         # operator<<, parse, binary (de)serialization
  divide.hpp         # divide_exact, divide_by_unit, norm-based reduction
  matrix.hpp         # Matrix<T>
  echelon.hpp        # incremental row-echelon basis
  gates.hpp          # constexpr gate matrices
  stats.hpp          # instrumentation counters
  acn.hpp            # umbrella header

src/
  bigint.cpp  format.cpp  stats.cpp  arena.cpp   # only what cannot be header-only

tests/
  test_ring.cpp  test_canonical.cpp  test_static.cpp  test_dynamic.cpp
  test_convert.cpp  test_divide.cpp  test_matrix.cpp  test_echelon.cpp
  test_circuits.cpp  test_overflow.cpp  test_props.cpp
  reference/reference.hpp reference.cpp    # slow, obviously-correct oracle
  support/generators.hpp                   # Catch2 generators for random values

bench/
  bench_arith.cpp  bench_hash.cpp  bench_matmul.cpp  bench_circuit.cpp

docs/
  DESIGN.md  PLAN.md
```

Rationale for header-heavy layout: `Acn<N, Int>` is a template and must be inlinable
into callers' hot loops; only the genuinely non-generic pieces (bignum glue, printing,
arena, counters) get `.cpp` files.

---

## 2. Phase 1 — Foundations and the ring core

**Goal:** exact, canonicalization-free arithmetic on `Z[x]/(x^n + 1)` coefficient
arrays, with the index/sign conventions nailed down and tested. Nothing here knows
about scale exponents. This is where the prototype's sign bug lived, so it is built and
tested in isolation, first.

### 2.1 `types.hpp`, `config.hpp`

- `using ScaleExp = i32;` plus `ACN_SCALE_EXP_64` opt-in.
- `enum class Status { ok, overflow, not_divisible, precision_mismatch };`
- `ACN_ASSERT(cond)` — active iff `ACN_DEBUG`; `ACN_ASSUME`/`ACN_INLINE` shims.
- `ACN_STATS` compile switch (off by default; Phase 7 fills in the counters).

### 2.2 `concepts.hpp`

`IntBackend<T>`: `T{0}`, `T{1}`, add/sub/mul, `is_zero`, `is_even`, `halve_exact`,
`ctz`, `shift_left`, negation, comparison, and a `checked_*` family returning
`Status`. `AlgebraicComplex<T>` exactly as listed in DESIGN.md §3.3.

### 2.3 `int_backend.hpp`

```cpp
template <class W> struct Checked;          // W in {i64, i128}
// checked_add(a,b,&out) -> bool overflowed, via __builtin_add_overflow
// ctz, is_even, halve_exact (asserts evenness), shl_checked
```
`BigInt` in `bigint.hpp` is deferred to Phase 5 but its *concept conformance* is
designed for now so nothing in `ring.hpp` assumes a machine word.

### 2.4 `ring.hpp` — the mathematical core

Free functions over `span<Int>`-like coefficient views, `n` passed as a value (static
callers pass `std::integral_constant<size_t, N>` so it constant-folds):

```cpp
// c_out = c_a + c_b, c_a - c_b, -c_a          (componentwise)
// c_out = conj(c_a):  out[0] = a[0]; out[j] = -a[n-j] for 0<j<n
// rotate(c_out, c_a, j):  multiply by omega^j, j in [0, 2n);
//     out[(j+t) mod n] = +-a[t], sign flips once per wrap past n
// mul(c_out, c_a, c_b):   negacyclic convolution
//     out[l] = sum_{t+u == l} a[t]b[u] - sum_{t+u == l+n} a[t]b[u]
// mul_by_sqrt2(c_out, c_a):  rotate(+n/4) - rotate(+3n/4)   [n >= 4]
// mul_small(c_out, c_a, Int z)
// norm2(c_out, c_a):      a * conj(a), specialized; result is real
```

Requirements:
- `mul` must not alias its inputs; `ACN_ASSERT` on overlap.
- Sign bookkeeping in `rotate` is a single `bool`, not a modulus per element.
- `n in {1, 2}`: `mul_by_sqrt2` is unavailable (there is no `omega^(n/4)`). These
  precisions instead express `sqrt(2)` scale changes only in even steps, and
  `canonicalize` (Phase 2) restricts to the `ctz` path. Encoded as
  `constexpr bool has_sqrt2 = (n >= 4)` and `static_assert`ed at use sites.

### 2.5 Tests — `test_ring.cpp`

1. Hand-computed tables for `n = 4`: `omega^j * omega^t` for all `(j,t)` against the
   negacyclic rule; `mul_by_sqrt2(1) == omega - omega^3`; `mul_by_sqrt2(mul_by_sqrt2(x))
   == 2x` **for every** random `x` (this is the single test that would have caught the
   prototype's `i*sqrt(2)` bug).
2. `conj(conj(x)) == x`, and `norm2(x)` equals `mul(x, conj(x))` computed by the
   general path, with an imaginary part of exactly zero.
3. `rotate(x, 2n) == x`, `rotate(x, n) == -x`, `rotate(x, n/2)` equals multiplication by
   `i` verified numerically.
4. Ring axioms over random inputs: commutativity, associativity, distributivity.
5. Numeric cross-check: evaluate coefficient vectors at `omega = exp(i*pi/n)` in
   `long double` and compare `mul`'s exact result to the numeric product within a bound
   derived from the coefficient magnitudes. Run for `n in {1,2,4,8,16,32}`.

**Acceptance:** all of the above green for `n` up to 32, ASan/UBSan clean, and
`mul_by_sqrt2` twice equals doubling for 10^5 random vectors per `n`.

---

## 3. Phase 2 — Canonical form

**Goal:** the invariant that DESIGN.md §4 makes the centerpiece.

### 3.1 `canonical.hpp`

```cpp
template <class Int> void canonicalize(std::span<Int> c, ScaleExp& k);
template <class Int> bool is_canonical(std::span<const Int> c, ScaleExp k);
```

Algorithm, in this order (each step cheap-rejects before doing work):

1. If all coefficients are zero: `k = 0`, return.
2. **Bulk even reduction.** `t = min_j ctz(c_j)` over nonzero `c_j`. If `t > 0`, shift
   all coefficients right by `t` and `k -= 2*t`. (Dividing the numerator by `2` is the
   same as reducing `k` by 2.)
3. **Odd `sqrt(2)` step** (only for `n >= 4`): compute `s = mul_by_sqrt2(c)` into
   scratch; if every coefficient of `s` is even, set `c = s/2`, `k -= 1`, and go back to
   step 2. Otherwise stop.

Parity pre-check for step 3: `mul_by_sqrt2` is a signed permutation-difference, so the
parity of `s_l` is `parity(c_{l-n/4}) ^ parity(c_{l-3n/4})`. Computing the parity mask
of `c` once (an `n`-bit word for `n <= 64`) lets step 3 be *predicted* in a couple of
instructions and skipped without touching scratch. Implement the mask fast path plus
the straightforward version, and cross-test them against each other.

Termination: each iteration strictly decreases `k` while keeping the value fixed and
the coefficients integral; the 2-adic valuation of the coefficient vector bounds the
number of iterations.

### 3.2 Tests — `test_canonical.cpp`

- Idempotence: `canonicalize(canonicalize(x)) == canonicalize(x)`.
- Value preservation: numeric evaluation before/after agrees exactly (compare via the
  Phase 1 `long double` evaluator with a bound).
- Uniqueness (the property everything else rests on): for random `x` and random
  `d >= 0`, the representation `(x * sqrt(2)^d, k + d)` canonicalizes to the *identical*
  coefficient array and `k` as `x` itself.
- `1/sqrt(2)` has `k = 1`; `sqrt(2)` has `k = -1`; `2` has `k = -2` with `c = (1,0,..)`;
  zero always has `k = 0`. Negative `k` is reachable and stable.
- Parity fast path and the naive path agree on 10^6 random inputs.
- `is_canonical` returns false for every non-canonical input the tests construct.

**Acceptance:** the uniqueness property holds for all tested `n` and `d in [0, 64]`.
This is the gate that permits hashing, so it is treated as blocking.

---

## 4. Phase 3 — `Acn<N, Int>`, the compile-time type

**Goal:** the type tools will actually use, with `N = 4` as the flagship.

### 4.1 The type

```cpp
template <std::size_t N, class Int = Checked<i64>>
class Acn {
    static_assert(std::has_single_bit(N));
    std::array<Int, N> c{};
    ScaleExp k = 0;
public:
    static constexpr std::size_t precision = N;
    // constexpr factories
    static constexpr Acn zero(), one(), imag(), inv_sqrt2(), sqrt2();
    template <std::size_t J> static constexpr Acn omega();
    static constexpr Acn from_ints(std::array<i64, N>, ScaleExp);
    // arithmetic (canonical in, canonical out)
    Acn operator+(const Acn&) const;  Acn operator-(const Acn&) const;
    Acn operator*(const Acn&) const;  Acn operator-() const;
    Acn& operator+=(const Acn&);      Acn& operator*=(const Acn&);
    Acn conj() const; Acn mul_omega(int j) const; Acn scale(ScaleExp d) const;
    Acn norm2() const;
    // queries
    bool is_zero() const; bool is_one() const; bool is_real() const;
    bool is_integer() const; bool is_unit() const;
    bool operator==(const Acn&) const; std::strong_ordering operator<=>(const Acn&) const;
    // destination forms — the performance API
    friend void add_into(Acn& d, const Acn& a, const Acn& b);
    friend void sub_into(Acn& d, const Acn& a, const Acn& b);
    friend void mul_into(Acn& d, const Acn& a, const Acn& b);   // d must not alias
    friend void fma_into(Acn& d, const Acn& a, const Acn& b);   // d += a*b
    // raw access for tests/serialization only
    std::span<const Int> coeffs() const; ScaleExp scale_exp() const;
};
```

Notes on the mechanics:

- `align_scales(a, b) -> (ca, cb, k)`: a **single** helper, used by add/sub/compare, that
  raises the smaller `k` to `max(k_a, k_b)`. Even part = `shl` by `d/2`; odd remainder =
  one `mul_by_sqrt2`. Never aligns downward. Everything about scale direction lives here
  — the prototype's bug class is structurally excluded.
- `operator*`: `ring::mul` on coefficients, `k = k_a + k_b`, then `canonicalize`.
- `mul_omega`, `conj`, `-`: no coefficient multiplies, and no canonicalization needed
  (they preserve the `sqrt(2)`-valuation). Documented and asserted.
- `scale(d)`: `k -= d` then canonicalize; `d` may be negative.
- `fma_into`: accumulate the raw convolution into `d`'s coefficients after aligning `d`
  to `k_a + k_b`, and canonicalize **once**. Provides a `fma_defer` variant plus an
  explicit `flush()` for accumulation chains (matrix dot products), so a loop of `w`
  terms canonicalizes once, not `w` times.
- Overflow: every coefficient operation goes through `Int`'s checked API. On overflow the
  operation leaves `d` unspecified and sets a per-object-free, thread-local
  `overflow_flag` (or returns `Status` from the `_into` forms — decide in review; the
  flag keeps `operator*` usable and is what a restart-based driver wants). `Acn` exposes
  `static bool overflow_seen()` / `clear_overflow()`.
- `operator==` is a componentwise compare and is `constexpr`; for trivially-copyable
  `Int` it compiles to a `memcmp`-shaped comparison.
- `<=>` gives the lexicographic total order on `(k, c)` DESIGN.md §5 lists, for use as a
  `std::map` key and for deterministic tie-breaking in DD normalization.

### 4.2 `hash.hpp`

`hash_value(const Acn&) -> u64`: for trivially-copyable `Int`, hash the `N` words plus
`k` with a fixed 64-bit mixer (xxh3-style or splitmix64 rounds — no `std::hash<size_t>`
combining, which is weak and libstdc++-specific). `std::hash` specialization delegates.
Documented precondition: canonical input, which the invariant guarantees.

### 4.3 Specializations

`if constexpr (N == 4)` selects a hand-unrolled 16-product convolution with
compile-time signs; `N == 2` and `N == 1` get their own trivial paths. No separate
`AlgebraicComplexNumber4` class — the prototype's duplicated `n = 4` implementation is
exactly what let the two code paths disagree.

### 4.4 Tests — `test_static.cpp`

- Everything in `test_ring.cpp` re-run through the public type for `N in {1,2,4,8,16}`.
- The prototype's existing test corpus, ported: the rescaling cases, addition/
  subtraction with and without rescaling, multiplication commutativity, negation.
  **The expected values for the rescaling tests are recomputed from
  `sqrt(2) = omega^(n/4) - omega^(3n/4)`** — the current fixtures encode the `i*sqrt(2)`
  behaviour and must not be copied over.
- Canonicity: `assert_canonical` (debug hook) on the result of every operation, over
  random inputs.
- `a == b  <=>  bytes(a) == bytes(b)` and `a == b => hash(a) == hash(b)`, over a corpus
  built to contain many equal-but-differently-derived values (e.g. `H*H` products).
- Constants: `inv_sqrt2() * sqrt2() == one()`, `imag()^2 == -one()`,
  `omega<1>()^(2N) == one()`, `omega<N/4>()` numerically `e^(i*pi/4)`.
- `fma_into` / `fma_defer` + `flush` equal the naive `sum a_i*b_i`.
- `sizeof(Acn<4, Checked<i64>>) <= 40` and `std::is_trivially_copyable_v` — asserted in
  the test, since both are load-bearing performance claims.

**Acceptance:** `Acn<4>` passes the full property suite; the differential test against
the Phase 1 ring layer and against the numeric evaluator is green; no allocation occurs
during any `Acn<N, Checked<i64>>` operation (checked with an operator-new interposer in
the test binary).

---

## 5. Phase 4 — `AcnDyn`, the runtime-precision type

**Goal:** same semantics, `n` chosen at construction.

### 5.1 `NumberContext`

Per DESIGN.md §11 open question 1, the plan **commits to context-owned precision** as
the primary model, with per-value `n` retained only as a field for validation:

```cpp
class NumberContext {
    u32 n_;                     // power of two, fixed at construction
    Arena arena_;               // bump allocator for coefficient blocks
    ScratchBuffer scratch_;     // thread_local per context, for mul/canonicalize
public:
    explicit NumberContext(u32 n, ArenaOptions = {});
    u32 precision() const;
    AcnDyn zero(), one(), imag(), inv_sqrt2();
    AcnDyn from_ints(std::span<const i64>, ScaleExp);
    void reset();               // free everything allocated from this context
};
```

Values are created only via a context (or by copying another value). This removes the
prototype's `ComplexMatrix::zero` pointer-to-external-prototype pattern, which does not
survive copy construction (`ComplexMatrix`'s copy ctor never copies `zero`, leaving a
dangling/garbage pointer — a real bug in `matrix.hpp` today).

Rationale for the commitment: it shrinks the value, removes a per-operation `n`
comparison from the hot path, and matches how verifiers work (one gate set per run).
Mixed-precision arithmetic is then an explicit `widen` between contexts (Phase 6), which
is also where the statistics hook can see it.

### 5.2 The type

```cpp
template <class Int = Checked<i64>>
class AcnDyn {
    Int*     c;       // n coefficients; inline buffer when n <= kInlineN (8)
    u32      n;
    ScaleExp k;
    // small-buffer union: alignas storage for kInlineN Ints
};
```

- SBO threshold `kInlineN = 8` (covers Clifford+T and `n = 8`), tunable.
- Same operator set and same `_into` forms as `Acn`, dispatching to the same
  `ring.hpp` / `canonical.hpp` code with a runtime `n`.
- Copy/move/assign are correct and tested (the prototype's `AlgebraicComplexNumber`
  move constructor copies the storage pointer *and* leaves the source owning it, and
  `DenseNumberStore::operator=` does not guard self-assignment — both must be covered
  by explicit tests here).
- `operator==` compares `(n, k, c)`; mismatched `n` in a binary op is an assert in debug
  and a `precision_mismatch` status in release, never a silent wrong answer.

### 5.3 `Arena`

Bump allocator with chunked growth, `reset()`, no per-value free. Coefficient blocks are
`n * sizeof(Int)` bytes, so a free list keyed by nothing at all is enough. Thread-safety:
one arena per context, contexts are not shared across threads (documented); a
`SharedArena` variant is out of scope for now.

### 5.4 Tests — `test_dynamic.cpp`

- **Differential against `Acn<N>`:** for `N in {1,2,4,8,16,32}`, run identical random
  operation sequences through `Acn<N, Checked<i64>>` and `AcnDyn` at `n = N`, and require
  bit-identical coefficient arrays and `k` at every step. This is the main correctness
  argument for the dynamic backend and it directly prevents the two-implementations-
  disagree failure the prototype exhibits.
- Copy/move/self-assignment/swap semantics, under ASan.
- SBO boundary: `n = 8` (inline) and `n = 16` (heap) behave identically.
- Allocation counting: number of arena allocations for a fixed workload is bounded, and
  the `_into` forms allocate zero.
- `NumberContext::reset()` invalidation is documented and checked in a death test.

**Acceptance:** bit-identical differential results against Phase 3 for all tested `n`;
zero leaks/UB under ASan/UBSan; `_into` forms allocation-free.

---

## 6. Phase 5 — Bignum backend and the two-tier promotion path

**Goal:** soundness when `i64` runs out, without paying for it in the common case.

- `bigint.hpp`: `BigInt` RAII wrapper over `mpz_t` — correct copy/move/assign,
  `noexcept` move, no raw `mpz_t` in the public API, `ctz` via `mpz_scan1`,
  `halve_exact` via `mpz_tdiv_q_2exp`, `to_double` via a correctly-rounded conversion
  (never `mpz_get_si`, which the prototype uses for printing and for `into_approx` and
  which silently truncates).
- GMP becomes an **optional** dependency: `ACN_WITH_GMP` CMake option, default ON;
  everything except `BigInt` builds without it. (The prototype's `arith.hpp` includes
  `<gmp-x86_64.h>` directly, which is a Fedora-specific internal header — replaced by
  `<gmp.h>` plus a `find_package`/`find_library` check.)
- `Acn<N, BigInt>` and `AcnDyn<BigInt>` instantiate and pass the full Phase 3/4 suites.
- **Promotion driver** in `convert.hpp`: `promote(const Acn<N, Checked<i64>>&) ->
  Acn<N, BigInt>` and a `try_narrow` back. Documented restart recipe: run with `i64`,
  check `overflow_seen()` at a natural checkpoint, re-run the affected computation with
  `BigInt`. The library provides the primitives and one worked example; it does not
  impose a policy.

### 6.1 Tests — `test_overflow.cpp`

- Constructed inputs that overflow `i64` at a known step (repeated squaring of a value
  with large coefficients); assert the flag fires and that no wrong-but-flagless result
  is produced.
- The same computation under `BigInt` produces the mathematically correct answer,
  verified against the reference oracle.
- Fuzz: random operation sequences with deliberately large coefficients; every run
  either flags overflow or matches the `BigInt` result exactly. **No silent divergence
  is permitted** — this is the acceptance criterion.

---

## 7. Phase 6 — Conversion, approximation, division, formatting

Independent work items; can be parallelized.

### 7.1 `convert.hpp` — precision embedding

- `Acn<N> -> Acn<M>` with `static_assert(M % N == 0)`: coefficient `j` moves to
  `j * (M/N)`, `k` unchanged. Tested as a ring homomorphism (preserves `+`, `*`, `1`).
- `narrow<N>(Acn<M>) -> std::optional<Acn<N>>`: succeeds iff off-sublattice
  coefficients are zero. `narrow(widen(x)) == x`.
- `AcnDyn` widening across contexts, counted in stats.

### 7.2 Approximation

- `to_complex() -> std::complex<double>`: evaluate `sum c_j * exp(i*pi*j/n)` in
  `double` from a precomputed `constexpr` table of the `n` roots, then apply the scale
  as `ldexp(x, -(k/2))` for the even part and one multiply by `M_SQRT1_2` for an odd
  remainder. Explicit underflow-to-zero / overflow-to-inf reporting via an
  out-parameter or a `Status`. Documented as **heuristic only**.
- `interval.hpp` + `to_interval() -> std::complex<Interval<double>>` with directed
  rounding (`std::fesetround` or explicit `nextafter` widening — prefer the latter, it
  does not fight the optimizer). Root table stored as pre-rounded lower/upper bounds.
  Property test: the exact value, evaluated at high precision by the oracle, always lies
  inside the returned interval; and `to_interval(x).contains(0)` is false whenever
  `x.is_zero()` is false and the interval is tight enough — the soundness contract a
  verifier's pruning relies on.
- Exact views/predicates: `into_sqrt2_repr` (the prototype's `(e,f,g,h,k)` form, kept
  because it is useful for interop), `is_real`, `is_integer`, `is_rational`, `is_unit`.
  `is_integer` is rewritten from the ring definition rather than the prototype's
  ad-hoc `k`-parity reasoning, which only handles the real case and mishandles negative
  `k`.

### 7.3 `divide.hpp`

- `divide_by_unit(a, u)`: total for `u = +-omega^j * sqrt(2)^m`; implemented as
  `mul_omega(-j)` + `scale(-m)` + sign. `is_unit()` classifies and decomposes.
- `divide_exact(a, b) -> optional<Acn>`: multiply `a` by the product of `b`'s nontrivial
  conjugates (equivalently, by `norm(b)/b`), giving `a * norm(b) / norm(b)`; divide the
  resulting coefficients by the rational integer `norm(b)` and succeed iff the division
  is exact in every coefficient and the scale works out. `norm(b)` is computed as a
  plain integer. Complexity is `O(n^2 log n)`-ish for the conjugate product; acceptable
  because this is not the innermost loop.
- Tests: `divide_exact(a*b, b) == a` for random `a, b != 0`; `divide_exact` returns
  `nullopt` for genuinely non-divisible pairs (verified against the oracle);
  `divide_by_unit` agrees with `divide_exact` whenever the divisor is a unit.

### 7.4 `format.hpp`

- `operator<<`: a readable canonical form, `(c_0, ..., c_{n-1}) / sqrt2^k`, plus an
  approximate `a+bi` in a comment-like suffix. Must not use `mpz_get_si`.
- `parse(std::string_view) -> optional<...>` round-tripping the above.
- Binary serialization: `n`, `k`, coefficient count and varint-encoded coefficients, for
  caching verification results. Round-trip property test.

---

## 8. Phase 7 — Algorithm scaling, instrumentation, benchmarks

**Goal:** make the performance claims in DESIGN.md §8 measurable rather than asserted.

### 8.1 `bench/`

Google Benchmark if available, otherwise a small built-in harness (must not become a
hard dependency). Benchmarks, all reported per-operation:

1. `add`, `mul`, `fma`, `canonicalize`, `==`, `hash` for
   `Acn<4|8|16|32, Checked<i64>>`, `Acn<4, BigInt>`, `AcnDyn` at matching `n`.
2. `bench_matmul`: dense `2^q x 2^q` products for `q = 1..6`, operator-based vs
   `fma_into`-based, showing the temporaries/allocation cost.
3. `bench_hash`: insertion of `10^6` amplitudes into an `unordered_set`, measuring both
   throughput and *collision behaviour* of `hash_value`.
4. `bench_circuit`: a realistic driver — `H^q` then a Clifford+T layer sequence then
   QFT-`q` — reporting time, peak coefficient bit-width, max `|k|`, canonicalization
   iterations per op, and bignum promotions. This is the number that actually predicts
   verifier performance.

### 8.2 Convolution ladder

With benchmark 1 in place, implement and measure:
- Karatsuba over the coefficient polynomial with negacyclic reduction.
- Optional negacyclic NTT / Nussbaumer for large `n` (multi-modular with machine-word
  primes + CRT to stay exact).
Select via `if constexpr` for `Acn<N>` and a runtime threshold for `AcnDyn`. **Only
merge a variant that the benchmark shows to win**, and record the measured crossovers in
DESIGN.md §5.2 (replacing the current guesses of `n >= 32` and `n >= 256`).

### 8.3 `stats.hpp`

Counters behind `ACN_STATS` (off by default, zero cost when off): operation counts,
canonicalization iterations, `mul_by_sqrt2` invocations, scale-alignment distance
histogram, widenings, overflow flags, bignum promotions, arena bytes. Dumped as a small
report a host tool can log.

### 8.4 SIMD (stretch)

Coefficient-array add/sub/rotate/parity for `n >= 8` with `i32`/`i64` coefficients, via
GCC vector extensions behind a runtime-dispatch-free compile option. Gated on benchmark
evidence; explicitly optional.

**Acceptance:** a checked-in baseline results file, a documented way to reproduce it,
and DESIGN.md §5.2/§8 updated with real numbers. Target to beat: `Acn<4, i64>` multiply
plus canonicalize at least an order of magnitude faster than the prototype's
`AlgebraicComplexNumber4::operator*` (which allocates five `mpz`s per product).

---

## 9. Phase 8 — Layers above, and retiring the prototype

### 9.1 `matrix.hpp`

- `template <AlgebraicComplex T> class Matrix` — dense row-major, **owns** its data,
  no external `zero` pointer, correct copy/move (the prototype's move ctor forgets
  `zero`; its copy ctor forgets it too).
- `operator*` implemented with `fma_defer` + one `flush` per output cell and a
  preallocated result; `adjoint`, `kron`, `transpose`, `identity`, `operator==`,
  scalar multiply, and the qubit-permutation operation carried over from
  `apply_qubit_permutation` (which currently swaps `width`/`height` when constructing
  the result — fixed and tested).
- `at(i,j)` bounds-checked in debug; `span`-based row access.

### 9.2 `echelon.hpp`

Incremental row-echelon basis, rewritten from `add_row_to_row_echelon_matrix`:
- `class EchelonBasis<T>` with `insert(row) -> optional<size_t>` (index if the row was
  independent, `nullopt` if it reduced to zero) — a real return contract instead of the
  current `assert(false)` on the full-rank path, and no `-1` sentinel.
- Fraction-free elimination: scale by pivots, never divide (§5.3), and *then*
  canonicalize + optionally reduce each row by its content (`gcd` of coefficients over
  `Z`, divided out along with the `sqrt(2)` power) to control the coefficient blowup
  that is the real cost driver here. Content reduction changes the row by a rational
  factor, so it is only valid for a *basis*, not for a general matrix — documented as
  such and confined to this class.
- Tests: rank of known matrices, independence detection, permutation invariance of the
  resulting rank, and a coefficient-growth regression test (bit-width of the largest
  coefficient after `m` insertions stays under a recorded bound).

### 9.3 `gates.hpp`

`constexpr` `H`, `X`, `Y`, `Z`, `S`, `S_dag`, `T`, `T_dag`, `CNOT`, `CZ`, `SWAP`,
`Toffoli`, `phase<k>`, `Rz<2*pi/2^m>` over `Acn<N>`, each with a `static_assert` that
`N` suffices for its angle. Tests: unitarity (`U * U.adjoint() == I`) and the standard
identities (`T^8 == I`, `H*Z*H == X`, `H*X*H == Z`, `S^2 == Z`, `CNOT^2 == I`,
Toffoli from Clifford+T decomposition equals the direct Toffoli).

### 9.4 `test_circuits.cpp`

End-to-end: `H^{(x)q}` amplitudes all equal `2^{-q/2}`; QFT-`q` matrix against the
closed form; a known circuit-equivalence pair decided correctly; a known
*in*equivalence pair (differing by a `T` gate) rejected — with the *exact* arithmetic
being what makes the distinction, checked by confirming the `double` path cannot.

### 9.5 Reference oracle — `tests/reference/`

Built early (needed from Phase 1) but listed here for completeness: dense polynomial
arithmetic mod `x^n + 1` over `BigInt` (or rationals), with the `sqrt(2)` power kept
symbolically and no canonicalization — deliberately naive, obviously correct, slow.
Every phase's differential tests compare against it.

### 9.6 Retirement

1. Port the remaining prototype tests, correcting the fixtures identified in DESIGN.md §2.2.
2. Delete `src/arith.{hpp,cpp}`, `src/basics.hpp`, `src/matrix*.hpp`, `src/matrix.cpp`,
   the old `src/tests.cpp`, and `src/legacy/`.
3. Rewrite `README.md`: correct the root-of-unity convention (the README's
   `e^(j*2*pi i/N)` disagrees with the code's `e^(i*pi/n)`), document both types, and
   link DESIGN.md.

---

## 10. Build, CI, tooling

Landing with Phase 1:

```
CMakeLists.txt (top level)
  option(ACN_WITH_GMP     "BigInt backend"       ON)
  option(ACN_BUILD_TESTS  "Build tests"          ON)
  option(ACN_BUILD_BENCH  "Build benchmarks"     OFF)
  option(ACN_STATS        "Instrumentation"      OFF)
  add_library(acn ...)                  # small; mostly INTERFACE
  add_library(acn::acn ALIAS acn)
  target_compile_features(acn PUBLIC cxx_std_20)
  install(...) + acnConfig.cmake        # so tools can find_package(acn)
tests/CMakeLists.txt   # one target per test file, all via catch_discover_tests / add_test
bench/CMakeLists.txt
```

Changes from the current build: the project moves out of `src/` to the repo root (a
`CMakeLists.txt` inside `src/` forces the awkward `cmake ../src` in the README);
GMP is linked through an imported target rather than a bare `find_library` that is
never actually required; and `tests` no longer recompiles the library sources
alongside the library.

- **CI matrix:** {GCC 15, Clang} x {Debug, Release, RelWithDebInfo} x {ASan+UBSan, off},
  plus one `ACN_WITH_GMP=OFF` build and one `-D_GLIBCXX_ASSERTIONS` build.
- `Dockerfile` updated: it currently omits `catch2-devel`, so the tests cannot be built
  in the container it describes. Add `catch2-devel`, `ninja-build`, `clang`.
- `clang-format` config + a format check job; `clang-tidy` with a curated check list.
- `compile_commands.json` already exported; keep it.
- A per-header "does it compile alone" target.

---

## 11. Sequencing, parallelism, and effort

```
Phase 1 (ring core)  ──> Phase 2 (canonical) ──> Phase 3 (Acn<N>) ──┬─> Phase 4 (AcnDyn)
                                                                     ├─> Phase 5 (BigInt)
                                                                     ├─> Phase 6 (convert/divide/format)
                                                                     └─> Phase 7 (bench/scaling)
Phases 4-7 ──> Phase 8 (matrix/echelon/gates, retire prototype)
Phase 0 (build+CI+oracle skeleton) runs alongside Phase 1.
```

| Phase | Content | Rough effort | Blocking? |
| --- | --- | --- | --- |
| 0 | Build, CI, oracle skeleton | 1 unit | yes |
| 1 | `ring.hpp` + tests | 2 units | yes |
| 2 | Canonical form | 1.5 units | yes — everything downstream depends on the invariant |
| 3 | `Acn<N, Int>` + hashing | 3 units | yes |
| 4 | `AcnDyn` + context/arena | 2.5 units | no |
| 5 | `BigInt` + promotion | 1.5 units | no |
| 6 | Convert / approx / divide / format | 3 units | no |
| 7 | Benchmarks, Karatsuba/NTT, stats | 3 units | no |
| 8 | Matrix, echelon, gates, retirement | 3 units | no |

"Unit" is deliberately unitless — relative sizing, not a schedule.

**Minimum useful deliverable (MUD):** Phases 0-3 give a complete, correct, fast,
hashable `Acn<4, i64>` with matrix-free arithmetic. A DD-based verifier can be built on
that alone; everything after is breadth and robustness.

---

## 12. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Sign/index convention errors (the prototype's actual failure mode) | One definition in `ring.hpp`; `mul_by_sqrt2` twice == doubling as a property test; numeric cross-check on every operation from Phase 1 onward; differential tests between all backends. |
| Canonicalization cost eats the gains | Parity-mask fast path; `ctz` bulk step; `fma_defer` so accumulation chains canonicalize once; canonicalization-iteration counter in stats to catch regressions. |
| Coefficient blowup in echelon elimination | Content reduction in `EchelonBasis`; coefficient-bit-width regression test; two-tier promotion so blowup degrades performance rather than correctness. |
| Two implementations drifting apart | `AcnDyn` shares `ring.hpp`/`canonical.hpp` with `Acn<N>` (no duplicated `n=4` code path), plus the mandatory bit-identical differential test in Phase 4. |
| `_into` API is unpleasant enough that callers use operators in hot loops | Operators stay available and correct; benchmarks quantify the gap; the library's own internals use only `_into`, so the API gets dogfooded. |
| Over-engineering (NTT, SIMD, sparse storage) before it pays | Phase 7 gates all of it on measurements; each is explicitly optional and independently revertible. |
| GMP / Fedora-specific build assumptions | GMP optional, `<gmp.h>` only, a no-GMP CI build. |
| Scope creep into general algebraic numbers | DESIGN.md §1.1 non-goals; `Frac<Acn>` explicitly a client concern. |

---

## 13. Decisions this plan makes on DESIGN.md's open questions

| DESIGN.md §11 | Decision here |
| --- | --- |
| 1. Context-owned `n` vs per-value `n` | **Context-owned** (§5.1), with `n` kept in the value only for validation. Revisit only if a concrete use case needs per-value precision. |
| 2. Sparse coefficient storage | **Deferred**, not designed in. Small-`n` plus explicit widening is the answer until a benchmark says otherwise (Phase 7). |
| 3. Delayed canonicalization | **Yes, but narrowly**: `fma_defer` + `flush` only, with a debug assert that a deferred value never reaches `hash_value` or `operator==`. |
| 4. Exact division for DD normalization | **Both**: `divide_by_unit` is the fast path DD callers should use; `divide_exact` exists and is measured in Phase 7. Recommendation recorded once numbers exist. |
| 5. Interning / hash-consing | **Client concern.** The library provides `hash_value` and a total order; it does not own a unique table. |
