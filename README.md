# acn — Algebraic Complex Numbers

Header-only C++20 library for **exact, canonical** arithmetic over the
complex numbers that occur as amplitudes in quantum circuits built from a
finite, dyadic gate set (Clifford+T, QFT-style circuits, ...). It is meant
as a low-level building block for formal-methods tools for quantum
computing: equivalence checkers, symbolic simulators, decision-diagram
packages, and model checkers — anywhere floating point's rounding error and
lack of canonical form (`1/sqrt(2)*sqrt(2) != 1`) are disqualifying.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design rationale and
[`docs/PLAN.md`](docs/PLAN.md) / [`PROGRESS.md`](PROGRESS.md) for
implementation status and history.

## What it provides

A value is a pair `(u, k)` with `u` in the ring `Z[omega] = Z[x]/(x^N + 1)`
(`omega = e^{i*pi/N}`) and `k` a (possibly negative) integer scale exponent,
denoting

```
z = u / sqrt(2)^k
```

`N` fixes which dyadic angles are representable exactly (`N = 4` covers
every amplitude of every Clifford+T circuit). Every public operation
returns a value in **canonical form**, so `operator==` is a componentwise
integer comparison and hashing is sound — the property decision-diagram
unique tables and echelon-basis dedup depend on.

Two implementations share one concept (`AlgebraicComplex`):

- `Acn<N, Int>` — compile-time precision `N`, no heap allocation, trivially
  copyable, fits in a cache line for small `N` (`Acn<4, i64>` is 40 bytes).
  The default choice when the gate set is known ahead of time.
- `AcnDyn<Int>` — runtime precision, for when `N` is only known after
  parsing the input circuit or must vary across subproblems.

`Int` is itself a choice: `Checked<i64>` / `CheckedI64` (default, overflow
is a checked, reported condition rather than silent UB), `__int128`, or
`BigInt` (a GMP `mpz_t` wrapper, opt-in via `ACN_WITH_GMP`).

On top of the number type: `Matrix<T>` (dense, `fma_into`-based products,
Kronecker product, qubit permutation), `include/acn/gates.hpp` (H, X, Y, Z,
S, T, CNOT, CZ, SWAP, Toffoli, parametrized phase/`Rz` gates), and
`include/acn/echelon.hpp` (an incremental row-echelon basis using
fraction-free/Bareiss elimination, for equivalence checking).

## Requirements

- A C++20 compiler (developed against GCC 15 / Clang 18+).
- [CMake](https://cmake.org/) >= 3.20.
- [GMP](https://gmplib.org/) (`libgmp-dev` / `gmp-devel`), for the `BigInt`
  backend — on by default, can be turned off (see below).
- [Catch2 v2](https://github.com/catchorg/Catch2) (`libcatch2-dev`), only to
  build the test suite. Note: this must be **Catch2 v2**, not v3 — the
  single-header `<catch2/catch.hpp>` include and
  `find_package(Catch2 2 REQUIRED)` are v2-specific. If your distro only
  packages v3 (e.g. Ubuntu 24.04's `libcatch2-dev`), build v2 from source:

  ```sh
  git clone --branch v2.13.10 --depth 1 https://github.com/catchorg/Catch2.git
  cmake -S Catch2 -B Catch2/build -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/some/prefix
  cmake --build Catch2/build --target install
  # then point this project's configure step at it:
  #   -DCMAKE_PREFIX_PATH=/some/prefix
  ```

## Using the library

`acn` is header-only. The simplest way to consume it is to add
`include/` to your include path and `#include "acn/acn.hpp"` (or a
specific header, e.g. `"acn/acn_static.hpp"`, to reduce compile-time
dependencies). Via CMake, add this repo as a subdirectory and link against
the `acn::acn` interface target:

```cmake
add_subdirectory(path/to/acn)
target_link_libraries(your_target PRIVATE acn::acn)
```

## Building the tests / benchmarks

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

Useful CMake options (pass as `-D<OPTION>=<ON|OFF>`):

| Option | Default | Effect |
| --- | --- | --- |
| `ACN_WITH_GMP` | `ON` | Build the `BigInt` backend against GMP; gates `test_bigint`, `test_overflow`, and the `BigInt`-based reference oracle/benchmarks. |
| `ACN_BUILD_TESTS` | `ON` | Build the Catch2 test suite (`tests/`). |
| `ACN_BUILD_BENCH` | `OFF` | Build the benchmark programs (`bench/`). |
| `ACN_STATS` | `OFF` | Enable instrumentation counters (canonicalize calls/iterations, overflow/promotion counts, ...); see `include/acn/stats.hpp`. Zero-cost when off. |

Example: a GMP-free debug build with benchmarks enabled:

```sh
cmake -S . -B build -DACN_WITH_GMP=OFF -DACN_BUILD_BENCH=ON
cmake --build build -j
ctest --test-dir build
./build/bench/bench_arith
```

## API documentation

A [Doxygen](https://www.doxygen.nl/) config is provided at the repo root
(`Doxyfile`; input is `include/acn` plus `docs/DESIGN.md` as the main
page). Generate HTML docs into `docs/api/` (gitignored) with:

```sh
doxygen Doxyfile
```

## Examples

### Basic arithmetic (compile-time precision)

```cpp
#include "acn/acn.hpp"

using acn::Acn;
using acn::CheckedI64;

using A = Acn<4, CheckedI64>;  // N = 4: exact for Clifford+T amplitudes

int main() {
    A half_root2 = A::inv_sqrt2();          // 1/sqrt(2)
    A one        = half_root2 * A::sqrt2(); // == A::one(), exactly

    A a = A::from_ints({1, 2, 0, -1}, /*k=*/0);
    A b = A::imag();                        // i = omega^(N/2)

    A sum  = a + b;
    A prod = a * b;

    bool equal = (one == A::one());         // exact, componentwise compare
    std::complex<double> approx = prod.to_complex(); // heuristic only

    return equal ? 0 : 1;
}
```

### Runtime precision with `AcnDyn`

```cpp
#include "acn/acn.hpp"

using acn::AcnDyn;
using acn::CheckedI64;

void example(std::size_t n /* power of two, chosen at runtime */) {
    auto a = AcnDyn<CheckedI64>::one(n);
    auto b = AcnDyn<CheckedI64>::imag(n);
    auto c = a + b * b;   // mismatched-n operands automatically widen
}
```

### Building and applying a gate

```cpp
#include "acn/acn.hpp"

using acn::Acn;
using acn::CheckedI64;
using acn::Matrix;

int main() {
    using A = Acn<4, CheckedI64>;

    Matrix<A> h = acn::gates::H<4, CheckedI64>();      // Hadamard
    Matrix<A> state(2, 1);
    state.at(0, 0) = A::one();                          // |0>

    Matrix<A> after_h = h * state;                       // (H|0> = |+>)
    bool is_uniform = (after_h.at(0, 0) == A::inv_sqrt2()) &&
                       (after_h.at(1, 0) == A::inv_sqrt2());
    return is_uniform ? 0 : 1;
}
```

### Checking a circuit identity exactly

```cpp
#include "acn/acn.hpp"

using acn::Acn;
using acn::CheckedI64;
using acn::Matrix;
namespace gates = acn::gates;

int main() {
    using A = Acn<4, CheckedI64>;

    // H * Z * H == X, checked as exact matrix equality (no epsilon).
    Matrix<A> lhs = gates::H<4, CheckedI64>() * gates::Z<4, CheckedI64>() *
                    gates::H<4, CheckedI64>();
    Matrix<A> rhs = gates::X<4, CheckedI64>();
    return (lhs == rhs) ? 0 : 1;
}
```

More worked examples — `H^{(x)q}` state preparation, a hand-derived 2-qubit
QFT circuit, `CNOT == (I⊗H) CZ (I⊗H)`, echelon-basis rank tracking — live in
`tests/test_circuits.cpp`, `tests/test_echelon.cpp`, and `tests/test_gates.cpp`.

## Project layout

```
include/acn/     Public headers (header-only library)
src/              (reserved; currently empty — no legacy prototype in this repo)
tests/            Catch2 test suite, plus tests/reference/ (independent symbolic oracle)
bench/            Benchmark programs (bench_arith, bench_hash, bench_matmul, bench_circuit)
docs/DESIGN.md    Design document (math model, canonical form, performance rationale)
docs/PLAN.md      Implementation plan / phases
PROGRESS.md       Session-by-session progress log
```

## Status

Under active development; see `PROGRESS.md` for what is implemented,
verified, and still outstanding. There is no stability guarantee on the
API yet.

## License

MIT — see [LICENSE](LICENSE).
