# Algebraic Complex Numbers — Design Document

Status: **draft, targets a from-scratch rewrite** (the current `src/` tree is treated
as a prototype and a source of lessons, not as a codebase to evolve).

---

## 1. Purpose

The library provides **exact** (error-free) arithmetic over the complex numbers that
occur as amplitudes in quantum circuits built from a finite, dyadic gate set. It is a
low-level building block for **formal-methods tools for quantum computing**:
equivalence checkers, symbolic simulators, decision-diagram (DD) packages,
model checkers and SMT-adjacent decision procedures.

Floating point is unusable in that setting for two reasons:

1. **Soundness.** A verifier that answers "these two circuits are equivalent" must not
   do so because two amplitudes agreed to 15 digits. Conversely, near-cancellation
   must not be reported as inequivalence.
2. **Canonicity.** DD-based tools identify nodes by the *bit pattern* of their edge
   weights. With floats, `1/sqrt(2) * sqrt(2) != 1`, so structurally identical states
   get distinct nodes, the unique table blows up, and the whole point of the data
   structure is lost. Exact arithmetic with a **canonical representation** restores
   the property that equal amplitudes have equal representations.

The library is therefore not a general-purpose computer-algebra number type. It is a
narrow, fast, allocation-conscious value type for one specific ring, designed to be
embedded in hot loops (matrix products, DD `apply` recursions, unique-table lookups).

### 1.1 Non-goals

- No general algebraic numbers, no field extensions outside the 2-power cyclotomic
  tower, no minimal-polynomial machinery, no Groebner/resultant computation.
- No transcendental functions, no rotation by angles that are not dyadic multiples of
  `2*pi`. Arbitrary-angle gates must be handled by the caller (approximate synthesis,
  or a different number type).
- No arbitrary-precision *floating* arithmetic (MPFR-style). Approximation to
  `double`/`complex<double>` exists only as an output/heuristic facility (§7).
- Not a linear-algebra library. A thin matrix layer is provided (§9) because the
  prototype has one and tools need it, but BLAS-grade performance is out of scope.

---

## 2. Mathematical model

Fix `n`, a power of two, and let

```
omega = e^(i*pi/n),        so omega^n = -1  and  omega is a primitive 2n-th root of unity.
```

The ring of interest is

```
R_n = Z[omega] = Z[x] / (x^n + 1)
```

which is the ring of integers of the cyclotomic field `Q(zeta_{2n})` (for `2n` a power
of two, the `2n`-th cyclotomic polynomial *is* `x^n + 1`, so `R_n` is free of rank `n`
over `Z` with basis `1, omega, ..., omega^(n-1)`).

A library value is a pair `(u, k)` with `u in R_n` and `k in Z`, denoting

```
     u                                    n-1
z = ------- ,        where   u  =   sum   sum  c_j * omega^j ,   c_j in Z.
    sqrt(2)^k                             j=0
```

Equivalently `z = (1/sqrt(2))^k * (c_0 + c_1*omega + ... + c_{n-1}*omega^(n-1))`.
`k` is called the **scale exponent**; it may be **negative** (§4.3 explains why this
matters). The stored data is thus the coefficient vector `c` plus `k`.

### 2.1 Why this ring

- `n = 1`: the Gaussian-like integers `Z` — trivial.
- `n = 2`: `Z[i]` plus `sqrt(2)` powers — exactly the amplitudes of **Clifford**
  circuits.
- `n = 4`: `Z[e^(i*pi/4)][1/sqrt(2)]` — exactly the amplitudes reachable by
  **Clifford+T** circuits. Every amplitude of every Clifford+T circuit is
  representable, and this is the case the prototype hard-codes as
  `AlgebraicComplexNumber4`. This is the single most important configuration.
- `n = 2^(m-1)`: amplitudes of circuits containing `Rz(2*pi/2^m)` / controlled-phase
  gates of order `2^m`, i.e. QFT-style circuits.

So `n` **is** the "precision" knob referred to throughout: it selects which dyadic
angles are representable exactly. Larger `n` costs memory (`n` coefficients) and
multiplication time; the coefficients themselves are exact at every `n`.

### 2.2 Key identities the implementation relies on

Let `w = omega`. All of these are exact and cheap.

| Identity | Consequence for the implementation |
| --- | --- |
| `w^n = -1` | Multiplication is a **negacyclic convolution**: index overflow flips sign. |
| `w^j * u` for `0 <= j < 2n` | Multiplying by a root of unity is a **rotation of the coefficient array with sign flips** — no arithmetic on coefficients at all. |
| `sqrt(2) = w^(n/4) - w^(3n/4)` (valid for `n >= 4`) | Scale changes by an odd amount are two shifted adds/subs, not a general multiply. |
| `i = w^(n/2)` | Multiplication by `i` is a rotation. |
| `conj(w^j) = w^(-j) = -w^(n-j)` (for `0 < j < n`) | Conjugation is an index reversal with sign flips. |
| `2 = -i * (sqrt(2))^2`, and `(1 - w)^n ~ 2` up to a unit | `sqrt(2)` is invertible once `k` is allowed to be negative; the prime above 2 is `lambda = 1 - w`, and `sqrt(2)` is a unit times `lambda^(n/2)`. Underlies the canonicalization test in §4.2. |
| `R_n` embeds in `R_m` whenever `n \| m`, via `w_n -> w_m^(m/n)` | Values of different precision can be combined by **index stretching** (§6), no re-derivation. |

> **Prototype discrepancy to fix.** The two prototype implementations disagree on what
> "increase `k` by one" means. `AlgebraicComplexNumber4::rescale` multiplies the
> numerator by `sqrt(2) = w - w^3`, while the generic `AlgebraicComplexNumber::rescale`
> multiplies by `w^(n/4) + w^(3n/4) = i*sqrt(2)`, and the current tests encode the
> latter. The generic `operator*` also *multiplies* the two scale exponents
> (`mpz_mul(result.scaling_factor, ...)`) where it must *add* them. The rewrite must
> fix `sqrt(2) = w^(n/4) - w^(3n/4)` as the one definition, in one place, and test the
> two backends against each other (§10).

---

## 3. The two implementations

Both implement the same concept (§3.3) and the same semantics; they differ only in
where `n` comes from and what that permits.

### 3.1 `Acn<N, Int>` — compile-time precision

```cpp
template <std::size_t N, class Int = i64>
class Acn {
    std::array<Int, N> c;   // coefficients, index j == omega^j
    ScaleExp           k;    // i32 or i64; see 4.4
};
```

Properties that follow from `N` being a template parameter, all of which matter for
verification workloads:

- **No heap allocation, no indirection.** The value is `sizeof(Int)*N + sizeof(k)`
  bytes, stored inline. `Acn<4, i64>` is 40 bytes and fits in a cache line — it can
  live directly inside a DD edge, a matrix cell, or a hash-table slot.
- **Trivially copyable** when `Int` is (`i64`, `i32`, `__int128`), so containers can
  `memcpy`, and `operator==` can be a `memcmp` on canonical values.
- **Fully unrolled kernels.** The `N*N` negacyclic convolution for `N = 4` unrolls to
  16 multiply-accumulates with compile-time-known signs. The wrap-around test
  `j1 + j2 >= N` and the sign flip are resolved at compile time, and `N/4`, `3N/4`,
  `N/2` are constants. This is the difference between a handful of nanoseconds and a
  function call with loop overhead and bignum bookkeeping.
- **Static checks.** `static_assert(std::has_single_bit(N))`; conversions
  `Acn<N> -> Acn<M>` only compile when `N` divides `M`.
- **Constexpr constants.** `zero()`, `one()`, `i()`, `omega<j>()`, `inv_sqrt2()` are
  `constexpr`, so gate matrices can be built at compile time.

This is the type tools should use by default, with `N` fixed by the gate set
(`N = 4` for Clifford+T).

### 3.2 `AcnDyn<Int>` — runtime precision

```cpp
class AcnDyn {
    Int*      c;      // n coefficients, from an allocator/arena
    u32       n;      // power of two, chosen at construction
    ScaleExp  k;
};
```

Needed when the gate set is only known after parsing the input circuit, when a tool
wants to *raise* precision adaptively (a QFT over `q` qubits needs `n = 2^(q-1)`), or
when different subproblems use different `n`.

Design consequences:

- `n` is carried per value; every binary operation asserts (debug) / promotes (release,
  §6) on mismatch.
- Allocation is the dominant cost, so: (a) a **small-buffer optimization** keeping
  `n <= 8` coefficients inline covers the Clifford+T case with no allocation at all;
  (b) all allocation goes through an **injected allocator/arena**, so a tool can bump-
  allocate a whole DD level and free it in one shot; (c) every operation has an
  `_into(dst, a, b)` form that writes into an existing value and never allocates.
- A `NumberContext` (or "factory") object owns `n`, the allocator, thread-local scratch
  and the interned constants; values are created through it. This replaces the
  prototype's habit of passing a pointer to an external `zero` element around
  (`ComplexMatrix::zero`), which is fragile and does not survive copies.

### 3.3 Shared concept

Generic code (matrices, DD packages, tests) is written against a concept, so the two
implementations — and any future backend — are interchangeable:

```cpp
template <class T>
concept AlgebraicComplex = requires (T a, T b, T& d, int j) {
    { T::zero() } -> std::same_as<T>;   { T::one() } -> std::same_as<T>;
    { a + b } -> std::same_as<T>;       { a * b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;       { -a }    -> std::same_as<T>;
    { a.is_zero() } -> std::same_as<bool>;
    { a == b }      -> std::same_as<bool>;
    { a.conj() }    -> std::same_as<T>;
    { a.mul_omega(j) } -> std::same_as<T>;      // rotation
    { a.scale(j) }     -> std::same_as<T>;      // multiply by sqrt(2)^j
    { canonicalize(d) };
    { hash_value(a) } -> std::same_as<u64>;
    { a.to_complex() } -> std::same_as<std::complex<double>>;
    add_into(d, a, b); mul_into(d, a, b); fma_into(d, a, b);
};
```

The prototype's approach — templating on a *storage* policy
(`AlgebraicComplexNumber<DenseNumberStore>`) — solved the wrong axis. The interesting
variation is `(precision binding) x (integer backend)`, not the coefficient container.

### 3.4 Integer backend `Int`

Coefficient growth, not `n`, is what actually decides performance.

| Backend | Use |
| --- | --- |
| `i64` (default) | Covers the overwhelming majority of verification runs. Every operation is **checked** for overflow (`__builtin_*_overflow`), and overflow is a *reported condition*, never silent UB. |
| `__int128` | Middle tier; still register-resident and trivially copyable. |
| `mpz` wrapper (`BigInt`) | Fallback for pathological growth. RAII, move-aware, and **never** exposed in the public API as a raw `mpz_t` (the prototype leaks `mpz_t&` out of `at()`, which makes const-correctness and canonicity impossible to enforce). |

**Two-tier execution is an explicit design goal:** a tool runs the whole computation
with `Acn<N, i64>`; if any operation reports overflow, the driver restarts (or promotes
the affected subtree) with `Acn<N, BigInt>`. Paying bignum costs on every amplitude
because a few might overflow is the single biggest performance mistake available here,
and it is the one the prototype makes — it uses `mpz_t` for *everything*, including the
scale exponent `k`, which never needs more than 32 bits in practice.

---

## 4. Canonical form — the central design decision

### 4.1 Why it is central

Verification tools spend their time on **equality tests and hashing**, not on
arithmetic:

- DD unique tables hash edge weights on every node creation.
- Symbolic/stabilizer-style methods keep sets of amplitude vectors and dedupe them.
- Gaussian elimination over amplitudes (as in the prototype's
  `add_row_to_row_echelon_matrix`) tests "is this exactly zero?" in its inner loop.

If a value has multiple representations, then `==` must normalize before comparing
(slow, allocating) and hashing is *impossible* to do correctly. So:

> **Invariant.** Every value returned by a public operation is in canonical form.
> Consequently `operator==` is a componentwise integer comparison (a `memcmp` for
> fixed-width `Int`), and `hash_value` may hash the raw representation.

### 4.2 The canonical form

A value `(c, k)` is canonical iff:

1. **Zero is unique:** if all `c_j == 0` then `k == 0`.
2. **`k` is minimal:** `sqrt(2)` does not divide the numerator `u`, i.e. one cannot
   write `z = u' / sqrt(2)^(k-1)` with `u' in R_n`.

Condition 2 needs an effective divisibility test. Using
`sqrt(2) * u = (w^(n/4) - w^(3n/4)) * u` (two sign-flipped rotations and an add):

```
canonicalize(c, k):
    if c == 0: k = 0; return
    loop:
        t = mul_by_sqrt2(c)                # exact, in R_n, no division
        if any coefficient of t is odd: break
        c = t / 2                          # exact integer halving
        k = k - 1
```

Because `z = u/sqrt(2)^k = (u*sqrt(2)/2)/sqrt(2)^(k-1)`, each iteration preserves the
value and strictly decreases `k`; the loop terminates because the 2-adic valuation of
the coefficient vector is finite. In the fixed-width backend the test is
`(t_0 | t_1 | ... | t_{n-1}) & 1`, and the halving is an arithmetic shift — a few
instructions. Two refinements worth having:

- Fast pre-pass: if all `c_j` are even, divide by 2 and `k -= 2` (this is the
  prototype's `normalize()`, generalized) — it removes many factors at once via
  `min_j ctz(c_j)`.
- Cheap rejection: track the parity mask; skip `mul_by_sqrt2` entirely when it cannot
  succeed.

Note that canonicalization **must not** clamp `k` at zero. `z = 1/sqrt(2)` has `k = 1`;
`z = sqrt(2)` has `k = -1`. The prototype's `normalize()` refuses to reduce when it
would drive `k` negative (`if (available_k < 0) return;`), which breaks the invariant
and hence breaks hashing. Allowing negative `k` also makes `sqrt(2)` and every
`+-w^j * sqrt(2)^m` a **unit** of the represented ring `Z[w][1/sqrt(2)]`, which is what
makes DD edge-weight normalization by division feasible (§5.3).

### 4.3 What canonical form does *not* include

We deliberately do **not** canonicalize by dividing out a "leading unit" (e.g. forcing
the first nonzero coefficient positive). It would buy nothing for equality — values are
already unique — and would cost a rotation on every operation.

### 4.4 `k` representation

`k` is `i32` (with `i64` as a compile option). Each `H` gate adds one to `k`, each
multiply adds the operands' `k`; a circuit deep enough to overflow `i32` (2 billion) is
far beyond anything tractable. Keeping `k` a machine word — rather than the prototype's
`mpz_t` — removes an allocation and a function call from *every* arithmetic operation
and from `operator==`. Overflow of `k` is checked in debug builds and detectable in
release.

---

## 5. Operations and their cost

Let `n` be the precision and `M` the coefficient multiply cost.

| Operation | Algorithm | Cost |
| --- | --- | --- |
| `a + b`, `a - b` | align scales (§5.1), then `n` integer adds | `O(n)` |
| `a * b` | negacyclic convolution, `c_l = sum_{j+j'=l} a_j b_j' - sum_{j+j'=l+n} a_j b_j'`; `k` adds | `O(n^2 * M)`, unrolled for small `n` |
| `-a`, `conj(a)` | sign flips / index reversal | `O(n)`, no multiplies |
| `a * w^j` | rotate by `j mod 2n` with sign flip on wrap | `O(n)`, no multiplies |
| `a * sqrt(2)^(+-m)` | `k -= m` then canonicalize | `O(n)` amortized |
| `a * z` for integer `z` | `n` integer multiplies | `O(n*M)` |
| `norm2(a) = a * conj(a)` | specialized (result is real: `n/2 + 1` distinct products) | ~half of a multiply |
| `a == b` | componentwise compare on canonical values | `O(n)`, branch-predictable |
| `hash_value(a)` | hash of raw bytes (fixed-width) | `O(n)` |
| `a.to_complex()` | §7 | `O(n)` |
| total order `a < b` | lexicographic on `(k, c)` | `O(n)` |

### 5.1 Scale alignment

`a + b` requires equal `k`. Align **upward** to `max(k_a, k_b)` — that only ever
multiplies a numerator by `sqrt(2)^d`, staying inside `R_n`; aligning downward would
require division. `d/2` of the factors are a plain `<< (d/2)` on all coefficients, and
an odd `d` costs one `mul_by_sqrt2`. The result is canonicalized, so repeated
addition does not let `k` drift upward.

The prototype gets this wrong twice: `AlgebraicComplexNumber4::operator+` picks the
*smaller* `k` and calls `rescale`, whose own assertion demands the larger; and the
generic `operator+` returns a value that was never canonicalized. Both are the kind of
bug that a single well-specified `align_scales` helper plus the canonicity invariant
make unrepresentable.

### 5.2 Multiplication for larger `n`

Naive `O(n^2)` is the right choice for `n <= 16` (the Clifford+T and small-QFT cases)
because it unrolls and has no setup cost. For larger `n` the library should provide,
behind the same interface:

- **Karatsuba** on the coefficient polynomial with negacyclic reduction — a clear win
  from roughly `n >= 32`.
- **Negacyclic FFT / Nussbaumer transform** for large `n`. Because `x^n + 1` is exactly
  the negacyclic modulus, this is the standard NTT setting; an exact integer NTT (or
  multi-modular CRT with machine-word primes) keeps the result exact. Only worth it for
  `n >= 256`, which is unusual for circuit verification but arises in QFT-heavy inputs.

The crossovers must be measured, not guessed, and selected at compile time for
`Acn<N>` (a `if constexpr` ladder) and at construction for `AcnDyn`.

### 5.3 Division

Division is **not** total in `R_n[1/sqrt(2)]`, and pretending otherwise is a common
source of unsoundness. The API is explicit about it:

- `divide_exact(a, b) -> optional<Acn>`: succeeds iff `b` divides `a` in the ring.
  Implemented as a multiply by `conj`-products / a norm-based reduction, then a check
  that the result has integer coefficients.
- `divide_by_unit(a, unit)`: total, cheap. Units of the form `+-w^j * sqrt(2)^m` cover
  the normalizations DD packages actually need (dividing a node's edge weights by the
  first one, when that first one happens to be a unit — which for Clifford+T circuits
  it very often is).
- No general field arithmetic. A tool needing `Q(zeta_{2n})` should build a rational
  wrapper `Frac<Acn>` on top; that is a separate, non-hot type and is out of scope for
  the core.

### 5.4 In-place and fused forms

Every operation has a destination form:

```cpp
void add_into(Acn& d, const Acn& a, const Acn& b);
void mul_into(Acn& d, const Acn& a, const Acn& b);   // d must not alias a or b
void fma_into(Acn& d, const Acn& a, const Acn& b);   // d += a*b, one canonicalize
```

`fma_into` matters: a matrix dot product or a DD `apply` accumulates `sum a_i*b_i`, and
doing it with `operator*` + `operator+=` costs a temporary and a canonicalization per
term. The prototype's `ComplexMatrix::operator*` allocates a fresh `mpz`-backed number
per multiply-add — for a 2^q x 2^q product that is the entire runtime. With
`fma_into`, the inner loop performs one canonicalization per output cell, and (for
`Acn<N, i64>`) zero allocations for the whole product.

The rule: **operators are convenience; the `_into` forms are the performance API**, and
the library's own internals use only the latter.

---

## 6. Mixing precisions

`R_n` embeds into `R_m` when `n | m` via `w_n -> w_m^(m/n)`, i.e. coefficient `c_j`
moves to index `j*(m/n)` and the rest are zero. Scale exponents are unaffected.

- `Acn<N>`: `template <size_t M> Acn<M> widen() const` with
  `static_assert(M % N == 0)` — an explicit, checked, compile-time-safe conversion.
- `AcnDyn`: binary operations on mismatched `n` **automatically widen** the
  narrower operand to `lcm` (= the larger, since both are powers of two). This is the
  main ergonomic reason the dynamic variant exists. It is also a silent performance
  cliff, so widening is counted in a statistics hook and can be made an error via a
  strict mode.
- Narrowing is `optional<Acn<N>>`: it succeeds only when all coefficients outside the
  sublattice are zero.

---

## 7. Approximation and interoperation

`to_complex()` returns `complex<double>`. Two hazards, both present in the prototype:

- Computing `pow(1/sqrt(2), k)` overflows/underflows for realistic `k`. Instead
  evaluate the coefficient sum in `double`, then apply the scale with `ldexp` for the
  even part and one multiply by `M_SQRT1_2` for the odd part; report underflow to zero
  and overflow to infinity explicitly.
- Converting each coefficient with `mpz_get_si` (a lossy `long` cast) silently
  corrupts large values. Use a correctly-rounded bignum-to-double conversion.

Because verifiers use approximations for *pruning*, an approximate value is only sound
if it comes with an error bound. The library therefore also offers
`to_interval() -> complex<Interval<double>>` with directed rounding, so a caller can
conclude "definitely nonzero" or "cannot decide, fall back to exact". Plain
`to_complex()` is documented as heuristic/diagnostic only.

Also provided: exact `Z[i, sqrt(2)]`-style views (the prototype's `into_sqrt2_repr`),
`is_real`, `is_integer`, `is_rational`, `is_unit`, and a round-trippable text format
plus a compact binary serialization (for caching verification results across runs).

---

## 8. Performance summary — what the design buys

Ranked by expected impact on a verification workload:

1. **Machine-word coefficients and `k` by default**, bignum only on demand. Removes
   nearly all allocation and pointer chasing from the hot path.
2. **Canonical form maintained as an invariant**, making `==` a comparison and hashing
   sound — this is what keeps DD unique tables small and lookups `O(1)`.
3. **Inline, fixed-size, trivially copyable values** (`Acn<N, i64>`), so amplitudes can
   be stored *by value* inside DD edges and matrices with good locality.
4. **Compile-time `n`**, giving unrolled convolutions and constant-folded index/sign
   arithmetic; the `N = 4` Clifford+T kernel is straight-line code.
5. **`_into` / `fma_into` forms** with no temporaries, plus arena allocation for
   `AcnDyn`, so composite operations do not allocate per term.
6. **Rotation, conjugation and negation as pure index/sign work** — the common gates
   (`S`, `T`, `Z`, phase) cost no coefficient multiplies at all.
7. **Cheap-path canonicalization** (parity mask, `ctz`-based bulk reduction) so the
   invariant is affordable.
8. **Value semantics with no shared mutable state**; only a thread-local scratch buffer
   for the dynamic backend, so tools can parallelize over subcircuits.
9. **Algorithm selection by `n`** (naive / Karatsuba / NTT) rather than one fixed
   convolution.
10. **A statistics/instrumentation hook** (counts of canonicalization loops, widenings,
    bignum promotions, overflow retries) — performance regressions in this library are
    otherwise invisible from inside a verifier.

Micro-optimizations explicitly considered and deferred: SIMD over the coefficient array
(worthwhile for `n >= 8` with `i32` coefficients), Montgomery-style delayed
canonicalization, and interning of frequently occurring amplitudes.

---

## 9. Layers above the number type

- **`Matrix<T>`** — dense, row-major, generic over `AlgebraicComplex T`, owning its
  storage (no external `zero` pointer). Multiplication uses `fma_into` into
  preallocated cells. Kronecker product, adjoint, and the qubit-permutation operation
  from the prototype belong here.
- **Row-echelon / rank machinery** — the prototype's incremental
  `add_row_to_row_echelon_matrix` is genuinely useful for equivalence checking
  (maintaining a basis of reachable amplitude vectors). Rewritten it needs: a
  correct return contract instead of `assert(false)` on the full-rank path, pivot
  handling that only ever multiplies (never divides, per §5.3), and coefficient-growth
  control, since fraction-free elimination is where integer blowup actually happens.
- **Gate library** — `H`, `S`, `T`, `X/Y/Z`, `CNOT`, `CZ`, controlled phases,
  `Rz(2*pi/2^m)`, as `constexpr` matrices over `Acn<N>` with a `static_assert` that `N`
  is large enough for the requested angle.

These are separate headers/targets; the number type must be usable without them.

---

## 10. Correctness strategy

Exact arithmetic is worth nothing if it is exactly wrong, and §2.2/§5.1 show the
prototype has silent sign and scale errors. The rewrite treats testing as part of the
design:

- **Invariant checks.** A debug-only `assert_canonical()` runs at the exit of every
  public operation.
- **Differential testing.** `Acn<4, i64>` vs `Acn<4, BigInt>` vs `AcnDyn` vs a
  deliberately slow, obviously-correct reference (dense polynomial arithmetic modulo
  `x^n + 1` over rationals, with the `sqrt(2)` power kept symbolically).
- **Numeric cross-check.** Every exact result is compared against
  `complex<long double>` evaluation within a derived bound. This catches sign and index
  errors — the `sqrt(2)` vs `i*sqrt(2)` bug in the prototype would have failed on the
  first random test.
- **Property/randomized tests.** Ring axioms; `canonicalize` idempotence;
  `a == b <=> repr(a) == repr(b)`; `widen` is a ring homomorphism; `x * conj(x)` real
  and nonnegative; `divide_exact(a*b, b) == a`.
- **Circuit-level tests.** Known amplitudes for `H^n`, QFT, Toffoli decompositions,
  and equivalence of gate identities (`T^8 = I`, `H*Z*H = X`, ...).
- **Overflow tests** driving the `i64` backend past its range and checking that the
  promotion path reproduces the bignum result.
- Sanitizer and valgrind-clean CI; a benchmark suite tracked over time.

---

## 11. Open questions

1. Should `AcnDyn` also carry `n` as *runtime-selected but per-context* (all values in
   a `NumberContext` share `n`), which would shrink the value and remove per-operation
   mismatch checks? This is likely the better trade for verifier use, with fully
   independent per-value `n` as the fallback.
2. Sparse coefficient storage: circuit amplitudes at large `n` are often supported on
   few indices. Is a sparse backend worth the branchy inner loop, or does small-`n` plus
   widening-on-demand cover it?
3. Whether to expose a delayed-canonicalization mode ("dirty" values) for long
   accumulation chains, and how to keep it from leaking into hashed contexts.
4. Exact division: is a norm-based `divide_exact` fast enough for DD normalization, or
   should DD callers be restricted to `divide_by_unit`?
5. Interning/hash-consing of amplitudes — belongs here, or in the client DD package?
