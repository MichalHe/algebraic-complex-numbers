#include <catch2/catch.hpp>

#include "acn/acn.hpp"

using acn::Acn;
using acn::CheckedI64;
using acn::Matrix;

namespace gates = acn::gates;

using Acn4 = Acn<4, CheckedI64>;
using Mat = Matrix<Acn4>;

namespace {

template <class M>
void require_unitary(const M& u) {
    auto dim = u.rows();
    REQUIRE(u.cols() == dim);
    REQUIRE(u * u.adjoint() == M::identity(dim));
    REQUIRE(u.adjoint() * u == M::identity(dim));
}

Mat power(const Mat& u, int exponent) {
    Mat result = Mat::identity(u.rows());
    for (int i = 0; i < exponent; ++i)
        result = result * u;
    return result;
}

}  // namespace

TEST_CASE("single-qubit Clifford+T gates are unitary", "[gates]") {
    require_unitary(gates::H<4>());
    require_unitary(gates::X<4>());
    require_unitary(gates::Y<4>());
    require_unitary(gates::Z<4>());
    require_unitary(gates::S<4>());
    require_unitary(gates::S_dag<4>());
    require_unitary(gates::T<4>());
    require_unitary(gates::T_dag<4>());
}

TEST_CASE("two/three-qubit gates are unitary", "[gates]") {
    require_unitary(gates::CNOT<4>());
    require_unitary(gates::CZ<4>());
    require_unitary(gates::SWAP<4>());
    require_unitary(gates::Toffoli<4>());
}

TEST_CASE("Pauli matrices square to identity and are self-adjoint", "[gates]") {
    auto x = gates::X<4>();
    auto y = gates::Y<4>();
    auto z = gates::Z<4>();
    REQUIRE(x == x.adjoint());
    REQUIRE(y == y.adjoint());
    REQUIRE(z == z.adjoint());
    REQUIRE(x * x == Mat::identity(2));
    REQUIRE(y * y == Mat::identity(2));
    REQUIRE(z * z == Mat::identity(2));
}

TEST_CASE("T^8 == I", "[gates]") {
    REQUIRE(power(gates::T<4>(), 8) == Mat::identity(2));
}

TEST_CASE("S^2 == Z", "[gates]") {
    auto s = gates::S<4>();
    REQUIRE(s * s == gates::Z<4>());
}

TEST_CASE("S and S_dag, T and T_dag are mutual inverses", "[gates]") {
    REQUIRE(gates::S<4>() * gates::S_dag<4>() == Mat::identity(2));
    REQUIRE(gates::T<4>() * gates::T_dag<4>() == Mat::identity(2));
}

TEST_CASE("H*Z*H == X and H*X*H == Z", "[gates]") {
    auto h = gates::H<4>();
    REQUIRE(h * gates::Z<4>() * h == gates::X<4>());
    REQUIRE(h * gates::X<4>() * h == gates::Z<4>());
}

TEST_CASE("H*H == I", "[gates]") {
    auto h = gates::H<4>();
    REQUIRE(h * h == Mat::identity(2));
}

TEST_CASE("CNOT^2 == I, CZ^2 == I, SWAP^2 == I", "[gates]") {
    REQUIRE(gates::CNOT<4>() * gates::CNOT<4>() == Mat::identity(4));
    REQUIRE(gates::CZ<4>() * gates::CZ<4>() == Mat::identity(4));
    REQUIRE(gates::SWAP<4>() * gates::SWAP<4>() == Mat::identity(4));
}

TEST_CASE("Toffoli^2 == I", "[gates]") {
    REQUIRE(gates::Toffoli<4>() * gates::Toffoli<4>() == Mat::identity(8));
}

TEST_CASE("phase<K> and Rz<M> are unitary and match their diagonal definitions", "[gates]") {
    auto p1 = gates::phase<4, 1>();  // e^{i*2*pi/2} = -1
    require_unitary(p1);
    REQUIRE(p1.at(1, 1) == -Acn4::one());

    auto p2 = gates::phase<4, 2>();  // e^{i*2*pi/4} = i
    require_unitary(p2);
    REQUIRE(p2.at(1, 1) == Acn4::imag());

    auto rz1 = gates::Rz<4, 1>();  // diag(e^{-i pi/2}, e^{i pi/2}) = diag(-i, i)
    require_unitary(rz1);
    REQUIRE(rz1.at(0, 0) == -Acn4::imag());
    REQUIRE(rz1.at(1, 1) == Acn4::imag());
}

TEST_CASE("gates are consistent across N (embedding via widen<M>)", "[gates]") {
    // T at N=8 (also a multiple of 4) should widen-match T at N=4 in the
    // sense that both represent the same abstract phase e^{i*pi/4} --
    // checked here via to_complex() rather than widen<M>(), since T<4>()
    // and T<8>() live in genuinely different rings (N=4 vs N=8), not one
    // embedded in the other through Acn::widen (which maps R_4 -> R_8 via
    // a fixed stride, not via "same gate at higher precision").
    auto t4 = gates::T<4>();
    auto t8 = gates::T<8, CheckedI64>();
    auto c4 = t4.at(1, 1).to_complex();
    auto c8 = t8.at(1, 1).to_complex();
    REQUIRE(std::abs(c4.real() - c8.real()) < 1e-9);
    REQUIRE(std::abs(c4.imag() - c8.imag()) < 1e-9);
}
