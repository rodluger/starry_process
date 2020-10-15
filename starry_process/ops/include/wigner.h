/**
\file wigner.h
\brief Rotation matrices.

*/

#ifndef _SP_WIGNER_H
#define _SP_WIGNER_H

#include "constants.h"
#include "utils.h"

namespace sp {
namespace wigner {

using namespace utils;

/**
Number of terms in the Wigner rotation matrix up to and including degree `l`.

*/
constexpr int nwig(const int l) {
  return (((l + 1) * (2 * l + 1) * (2 * l + 3)) / 3);
}

/**
Number of terms in the Wigner rotation matrix of degree `l`.

*/
constexpr int nwigl(const int l) { return (2 * l + 1) * (2 * l + 1); }

/**
Compute the Wigner d matrices.

*/
template <class Scalar, class T2, class T1, class T>
inline void dlmn(int l, const Scalar &c2, const Scalar &s2, const T2 &Dlm2,
                 const T2 &Dlm2p, const T1 &Dlm1, const T1 &Dlm1p, T &Dl,
                 T &Dlp) {
  int iinf = 1 - l;
  int isup = -iinf;
  int m, mp;
  int al, al1, tal1, amp, laux, lbux, am, lauz, lbuz;
  int sign;
  Scalar ali, auz, aux, cux, fact, cuz;
  Scalar term, cosaux, termp;

  Scalar tgbet2;
  if (abs(s2) < Scalar(SP_WIGNER_TOL))
    tgbet2 = s2; // = 0
  else
    tgbet2 = (Scalar(1.0) - c2) / s2;

  // Compute the D[l;m',m) matrix.
  // First row by recurrence (Eq. 19 and 20 in Alvarez Collado et al.)
  Dl(2 * l, 2 * l) =
      0.5 * Dlm1(isup + l - 1, isup + l - 1) * (Scalar(1.0) + c2);
  Dlp(2 * l, 2 * l) =
      0.5 * (Dlm1p(isup + l - 1, isup + l - 1) * (Scalar(1.0) + c2) -
             Dlm1(isup + l - 1, isup + l - 1) * s2);
  Dl(2 * l, 0) = 0.5 * Dlm1(isup + l - 1, -isup + l - 1) * (Scalar(1.0) - c2);
  Dlp(2 * l, 0) =
      0.5 * (Dlm1p(isup + l - 1, -isup + l - 1) * (Scalar(1.0) - c2) +
             Dlm1(isup + l - 1, -isup + l - 1) * s2);
  for (m = isup; m > iinf - 1; --m) {
    Dl(2 * l, m + l) =
        -tgbet2 * sqrt(Scalar(l + m + 1) / (l - m)) * Dl(2 * l, m + 1 + l);
    Dlp(2 * l, m + l) = -sqrt(Scalar(l + m + 1) / (l - m)) *
                        (Dl(2 * l, m + 1 + l) / (Scalar(1.0) + c2) +
                         tgbet2 * Dlp(2 * l, m + 1 + l));
  }

  // The rows of the upper quarter triangle of the D[l;m',m) matrix
  // (Eq. 21 in Alvarez Collado et al.)
  al = l;
  al1 = al - 1;
  tal1 = al + al1;
  ali = Scalar(1.0) / al1;
  cosaux = c2 * al * al1;
  for (mp = l - 1; mp > -1; --mp) {
    amp = mp;
    laux = l + mp;
    lbux = l - mp;
    aux = ali / sqrt(Scalar(laux * lbux));
    cux = sqrt(Scalar((laux - 1) * (lbux - 1))) * al;
    for (m = isup; m > iinf - 1; --m) {
      am = m;
      lauz = l + m;
      lbuz = l - m;
      auz = Scalar(1.0) / sqrt(Scalar(lauz * lbuz));
      fact = aux * auz;
      term = tal1 * (cosaux - Scalar(am * amp)) * Dlm1(mp + l - 1, m + l - 1);
      termp =
          tal1 * (-s2 * al * al1 * Dlm1(mp + l - 1, m + l - 1) +
                  (cosaux - Scalar(am * amp)) * Dlm1p(mp + l - 1, m + l - 1));
      if ((lbuz != 1) && (lbux != 1)) {
        cuz = sqrt(Scalar((lauz - 1) * (lbuz - 1)));
        term = term - Dlm2(mp + l - 2, m + l - 2) * cux * cuz;
        termp = termp - Dlm2p(mp + l - 2, m + l - 2) * cux * cuz;
      }
      Dl(mp + l, m + l) = fact * term;
      Dlp(mp + l, m + l) = fact * termp;
    }
    ++iinf;
    --isup;
  }

  // The remaining elements of the D[l;m',m) matrix are calculated
  // using the corresponding symmetry relations:
  // reflection ---> ((-1)**(m-m')) D[l;m,m') = D[l;m',m), m'<=m
  // inversion ---> ((-1)**(m-m')) D[l;-m',-m) = D[l;m',m)

  // Reflection
  sign = 1;
  iinf = -l;
  isup = l - 1;
  for (m = l; m > 0; --m) {
    for (mp = iinf; mp < isup + 1; ++mp) {
      Dl(mp + l, m + l) = sign * Dl(m + l, mp + l);
      Dlp(mp + l, m + l) = sign * Dlp(m + l, mp + l);
      sign *= -1;
    }
    ++iinf;
    --isup;
  }

  // Inversion
  iinf = -l;
  isup = iinf;
  for (m = l - 1; m > -(l + 1); --m) {
    sign = -1;
    for (mp = isup; mp > iinf - 1; --mp) {
      Dl(mp + l, m + l) = sign * Dl(-mp + l, -m + l);
      Dlp(mp + l, m + l) = sign * Dlp(-mp + l, -m + l);
      sign *= -1;
    }
    ++isup;
  }
}

/**
Compute the Wigner D matrices.

*/
template <class Scalar, class T>
inline void rotar(const Scalar &theta, T &R, T &Rp) {
  // Temporaries
  Scalar root_two = sqrt(Scalar(2.0));
  Scalar d1, d2, d1p, d2p;
  int aux, cosag, sinag, cosmal, sinmal, cosmga, sinmga, cosagm, sinagm;
  int sign;

  Scalar c2 = cos(theta);
  Scalar s2 = sin(theta);
  Scalar c2p = -s2;
  Scalar s2p = c2;

  // The complex Wigner matrix
  Vector<Scalar, SP__NWIG> D;
  Vector<Scalar, SP__NWIG> Dp;

  // Compute the initial complex matrices D[0], D[1]
  D(0) = 1.0;
  Dp(0) = 0.0;
  D(9) = 0.5 * (Scalar(1.0) + c2);
  Dp(9) = 0.5 * c2p;
  D(8) = -s2 / root_two;
  Dp(8) = -s2p / root_two;
  D(7) = 0.5 * (Scalar(1.0) - c2);
  Dp(7) = -0.5 * c2p;
  D(6) = -D(8);
  Dp(6) = -Dp(8);
  D(5) = D(9) - D(7);
  Dp(5) = Dp(9) - Dp(7);
  D(4) = D(8);
  Dp(4) = Dp(8);
  D(3) = D(7);
  Dp(3) = Dp(7);
  D(2) = D(6);
  Dp(2) = Dp(6);
  D(1) = D(9);
  Dp(1) = Dp(9);

  // Compute the initial real matrices R[0], R[1]
  R(0) = 1.0;
  Rp(0) = 0.0;
  R(1) = D(9) - D(7);
  Rp(1) = Dp(9) - Dp(7);
  R(2) = -root_two * D(6);
  Rp(2) = -root_two * Dp(6);
  R(3) = 0;
  Rp(3) = 0;
  R(4) = -root_two * D(8);
  Rp(4) = -root_two * Dp(8);
  R(5) = D(5);
  Rp(5) = Dp(5);
  R(6) = 0;
  Rp(6) = 0;
  R(7) = 0;
  Rp(7) = 0;
  R(8) = 0;
  Rp(8) = 0;
  R(9) = D(9) + D(7);
  Rp(9) = Dp(9) + Dp(7);

  // The remaining matrices are calculated using
  // symmetry and and recurrence relations
  for_constexpr<for_bounds<2, SP__LMAX + 1>>([&](auto l) {

    // Compute D[l] from D[l - 1] and D[l - 2]
    Map<RowMatrix<Scalar, 2 * (l - 2) + 1, 2 * (l - 2) + 1>> Dlm2(
        D.segment(nwig(l - 3), nwigl(l - 2)).data());
    Map<RowMatrix<Scalar, 2 * (l - 2) + 1, 2 * (l - 2) + 1>> Dlm2p(
        Dp.segment(nwig(l - 3), nwigl(l - 2)).data());
    Map<RowMatrix<Scalar, 2 * (l - 1) + 1, 2 * (l - 1) + 1>> Dlm1(
        D.segment(nwig(l - 2), nwigl(l - 1)).data());
    Map<RowMatrix<Scalar, 2 * (l - 1) + 1, 2 * (l - 1) + 1>> Dlm1p(
        Dp.segment(nwig(l - 2), nwigl(l - 1)).data());
    Map<RowMatrix<Scalar, 2 * l + 1, 2 * l + 1>> Dl(
        D.segment(nwig(l - 1), nwigl(l)).data());
    Map<RowMatrix<Scalar, 2 * l + 1, 2 * l + 1>> Dlp(
        Dp.segment(nwig(l - 1), nwigl(l)).data());
    dlmn(l, c2, s2, Dlm2, Dlm2p, Dlm1, Dlm1p, Dl, Dlp);

    // Compute the real rotation matrix R[l] from the complex one D[l]
    Map<RowMatrix<Scalar, 2 * l + 1, 2 * l + 1>> Rl(
        R.segment(nwig(l - 1), nwigl(l)).data());
    Map<RowMatrix<Scalar, 2 * l + 1, 2 * l + 1>> Rlp(
        Rp.segment(nwig(l - 1), nwigl(l)).data());
    Rl(0 + l, 0 + l) = Dl(0 + l, 0 + l);
    Rlp(0 + l, 0 + l) = Dlp(0 + l, 0 + l);
    cosmal = 0;
    sinmal = -1;
    sign = -1;
    for (int mp = 1; mp < l + 1; ++mp) {
      cosmga = 0;
      sinmga = 1;
      Rl(mp + l, 0 + l) = root_two * Dl(0 + l, mp + l) * cosmal;
      Rlp(mp + l, 0 + l) = root_two * Dlp(0 + l, mp + l) * cosmal;
      Rl(-mp + l, 0 + l) = root_two * Dl(0 + l, mp + l) * sinmal;
      Rlp(-mp + l, 0 + l) = root_two * Dlp(0 + l, mp + l) * sinmal;
      for (int m = 1; m < l + 1; ++m) {
        d1 = Dl(-mp + l, -m + l);
        d1p = Dlp(-mp + l, -m + l);
        d2 = sign * Dl(mp + l, -m + l);
        d2p = sign * Dlp(mp + l, -m + l);
        cosag = cosmal * cosmga - sinmal * sinmga;
        cosagm = cosmal * cosmga + sinmal * sinmga;
        sinag = sinmal * cosmga + cosmal * sinmga;
        sinagm = sinmal * cosmga - cosmal * sinmga;
        Rl(l, m + l) = root_two * Dl(m + l, 0 + l) * cosmga;
        Rlp(l, m + l) = root_two * Dlp(m + l, 0 + l) * cosmga;
        Rl(l, -m + l) = -root_two * Dl(m + l, 0 + l) * sinmga;
        Rlp(l, -m + l) = -root_two * Dlp(m + l, 0 + l) * sinmga;
        Rl(mp + l, m + l) = d1 * cosag + d2 * cosagm;
        Rlp(mp + l, m + l) = d1p * cosag + d2p * cosagm;
        Rl(mp + l, -m + l) = -d1 * sinag + d2 * sinagm;
        Rlp(mp + l, -m + l) = -d1p * sinag + d2p * sinagm;
        Rl(-mp + l, m + l) = d1 * sinag + d2 * sinagm;
        Rlp(-mp + l, m + l) = d1p * sinag + d2p * sinagm;
        Rl(-mp + l, -m + l) = d1 * cosag - d2 * cosagm;
        Rlp(-mp + l, -m + l) = d1p * cosag - d2p * cosagm;
        aux = -sinmga;
        sinmga = cosmga;
        cosmga = aux;
      }
      sign *= -1;
      aux = sinmal;
      sinmal = -cosmal;
      cosmal = aux;
    }

  });

  return;
}

/**
 * Compute the Wigner rotation matrix Rx(theta).
*/
template <typename SCALAR, typename VECTOR>
inline void computeRx(const SCALAR &theta, VECTOR &Rx, VECTOR &dRxdtheta) {
  rotar(theta, Rx, dRxdtheta);
}

/**
 * Compute the tensor dot product M . Rz(theta)
*/
template <typename VECTOR, typename MATRIX>
inline void computeTensordotRz(const MATRIX &M, const VECTOR &theta,
                               MATRIX &f) {

  using Scalar = typename VECTOR::Scalar;
  int K = theta.size();

  // Compute sin & cos
  auto costheta = theta.array().cos();
  auto sintheta = theta.array().sin();

  // Initialize our z rotation vectors
  RowMatrix<Scalar, Dynamic, SP__LMAX + 1> cosnt(K, SP__LMAX + 1);
  RowMatrix<Scalar, Dynamic, SP__LMAX + 1> sinnt(K, SP__LMAX + 1);
  RowMatrix<Scalar, Dynamic, SP__N> cosmt(K, SP__N);
  RowMatrix<Scalar, Dynamic, SP__N> sinmt(K, SP__N);
  cosnt.col(0).setOnes();
  sinnt.col(0).setZero();

  // Compute the cos and sin vectors for the zhat rotation
  cosnt.col(1) = costheta;
  sinnt.col(1) = sintheta;
  for (int n = 2; n < SP__LMAX + 1; ++n) {
    cosnt.col(n) =
        2.0 * cosnt.col(n - 1).cwiseProduct(cosnt.col(1)) - cosnt.col(n - 2);
    sinnt.col(n) =
        2.0 * sinnt.col(n - 1).cwiseProduct(cosnt.col(1)) - sinnt.col(n - 2);
  }
  int n = 0;
  for (int l = 0; l < SP__LMAX + 1; ++l) {
    for (int m = -l; m < 0; ++m) {
      cosmt.col(n) = cosnt.col(-m);
      sinmt.col(n) = -sinnt.col(-m);
      ++n;
    }
    for (int m = 0; m < l + 1; ++m) {
      cosmt.col(n) = cosnt.col(m);
      sinmt.col(n) = sinnt.col(m);
      ++n;
    }
  }

  // Dot them in
  for (int l = 0; l < SP__LMAX + 1; ++l) {
    for (int j = 0; j < 2 * l + 1; ++j) {
      f.col(l * l + j) =
          M.col(l * l + j).cwiseProduct(cosmt.col(l * l + j)) +
          M.col(l * l + 2 * l - j).cwiseProduct(sinmt.col(l * l + j));
    }
  }
}

/**
 * Computes the gradient of the tensor dot product M . Rz(theta).
*/
template <typename VECTOR, typename MATRIX>
inline void computeTensordotRzGradient(const MATRIX &M, const VECTOR &theta,
                                       const MATRIX &bf, MATRIX &bM,
                                       VECTOR &btheta) {

  using Scalar = typename VECTOR::Scalar;
  int K = theta.size();

  // Init grads
  btheta.setZero();
  bM.setZero();

  // Compute sin & cos
  auto costheta = theta.array().cos();
  auto sintheta = theta.array().sin();

  // Initialize our z rotation vectors
  RowMatrix<Scalar, Dynamic, SP__LMAX + 1> cosnt(K, SP__LMAX + 1);
  RowMatrix<Scalar, Dynamic, SP__LMAX + 1> sinnt(K, SP__LMAX + 1);
  RowMatrix<Scalar, Dynamic, SP__N> cosmt(K, SP__N);
  RowMatrix<Scalar, Dynamic, SP__N> sinmt(K, SP__N);
  cosnt.col(0).setOnes();
  sinnt.col(0).setZero();

  // Compute the cos and sin vectors for the zhat rotation
  cosnt.col(1) = costheta;
  sinnt.col(1) = sintheta;
  for (int n = 2; n < SP__LMAX + 1; ++n) {
    cosnt.col(n) =
        2.0 * cosnt.col(n - 1).cwiseProduct(cosnt.col(1)) - cosnt.col(n - 2);
    sinnt.col(n) =
        2.0 * sinnt.col(n - 1).cwiseProduct(cosnt.col(1)) - sinnt.col(n - 2);
  }
  int n = 0;
  for (int l = 0; l < SP__LMAX + 1; ++l) {
    for (int m = -l; m < 0; ++m) {
      cosmt.col(n) = cosnt.col(-m);
      sinmt.col(n) = -sinnt.col(-m);
      ++n;
    }
    for (int m = 0; m < l + 1; ++m) {
      cosmt.col(n) = cosnt.col(m);
      sinmt.col(n) = sinnt.col(m);
      ++n;
    }
  }

  // Dot them in
  for (int l = 0; l < SP__LMAX + 1; ++l) {
    for (int j = 0; j < 2 * l + 1; ++j) {
      Vector<Scalar, Dynamic> tmp_c =
          bf.col(l * l + j).cwiseProduct(cosmt.col(l * l + j));
      Vector<Scalar, Dynamic> tmp_s =
          bf.col(l * l + j).cwiseProduct(sinmt.col(l * l + j));
      btheta += (j - l) * (M.col(l * l + 2 * l - j).cwiseProduct(tmp_c) -
                           M.col(l * l + j).cwiseProduct(tmp_s));
      bM.col(l * l + 2 * l - j) += tmp_s;
      bM.col(l * l + j) += tmp_c;
    }
  }
}

/**
 * Compute the batched tensor dot product T_ij R_ilk M_lj
*/
template <typename VECTOR, typename MATRIX>
inline void computeSpecialTensordotRz(const MATRIX &T, const MATRIX &M,
                                      const VECTOR &theta, VECTOR &f) {

  using Scalar = typename VECTOR::Scalar;
  int K = theta.size();

  // Compute sin & cos
  auto costheta = theta.array().cos();
  auto sintheta = theta.array().sin();

  // Initialize our z rotation vectors
  RowMatrix<Scalar, Dynamic, SP__LMAX + 1> cosnt(K, SP__LMAX + 1);
  RowMatrix<Scalar, Dynamic, SP__LMAX + 1> sinnt(K, SP__LMAX + 1);
  RowMatrix<Scalar, Dynamic, SP__N> cosmt(K, SP__N);
  RowMatrix<Scalar, Dynamic, SP__N> sinmt(K, SP__N);
  cosnt.col(0).setOnes();
  sinnt.col(0).setZero();
  RowMatrix<Scalar, Dynamic, Dynamic> TM1 = T.cwiseProduct(M);
  RowMatrix<Scalar, Dynamic, Dynamic> TM2(SP__N, SP__N);

  // Compute the cos and sin vectors for the zhat rotation
  cosnt.col(1) = costheta;
  sinnt.col(1) = sintheta;
  for (int n = 2; n < SP__LMAX + 1; ++n) {
    cosnt.col(n) =
        2.0 * cosnt.col(n - 1).cwiseProduct(cosnt.col(1)) - cosnt.col(n - 2);
    sinnt.col(n) =
        2.0 * sinnt.col(n - 1).cwiseProduct(cosnt.col(1)) - sinnt.col(n - 2);
  }
  int n = 0;
  for (int l = 0; l < SP__LMAX + 1; ++l) {
    for (int m = -l; m < 0; ++m) {
      cosmt.col(n) = cosnt.col(-m);
      sinmt.col(n) = -sinnt.col(-m);
      TM2.col(l * l + l + m) =
          T.col(l * l + l + m).cwiseProduct(M.col(l * l + l - m));
      ++n;
    }
    for (int m = 0; m < l + 1; ++m) {
      cosmt.col(n) = cosnt.col(m);
      sinmt.col(n) = sinnt.col(m);
      TM2.col(l * l + l + m) =
          T.col(l * l + l + m).cwiseProduct(M.col(l * l + l - m));
      ++n;
    }
  }

  // Apply the rotation
  f = (cosmt * TM1 + sinmt * TM2).rowwise().sum();
}

/**
 * Computes the gradient of the batched tensor dot product T_ij R_ilk M_lj
*/
template <typename VECTOR, typename MATRIX>
inline void computeSpecialTensordotRzGradient(const MATRIX &T, const MATRIX &M,
                                              const VECTOR &theta,
                                              const VECTOR &bf, MATRIX &bM,
                                              VECTOR &btheta) {

  using Scalar = typename VECTOR::Scalar;
  int K = theta.size();

  // Compute sin & cos
  auto costheta = theta.array().cos();
  auto sintheta = theta.array().sin();

  // Initialize our z rotation vectors
  RowMatrix<Scalar, Dynamic, SP__LMAX + 1> cosnt(K, SP__LMAX + 1);
  RowMatrix<Scalar, Dynamic, SP__LMAX + 1> sinnt(K, SP__LMAX + 1);
  RowMatrix<Scalar, Dynamic, SP__N> cosmt(K, SP__N);
  RowMatrix<Scalar, Dynamic, SP__N> sinmt(K, SP__N);
  cosnt.col(0).setOnes();
  sinnt.col(0).setZero();
  RowMatrix<Scalar, Dynamic, Dynamic> TM1 = T.cwiseProduct(M);
  RowMatrix<Scalar, Dynamic, Dynamic> TM2(SP__N, SP__N);
  RowMatrix<Scalar, Dynamic, Dynamic> T_r(SP__N, SP__N);
  RowMatrix<Scalar, Dynamic, Dynamic> mmat(SP__N, SP__N);

  // Compute the cos and sin vectors for the zhat rotation
  cosnt.col(1) = costheta;
  sinnt.col(1) = sintheta;
  for (int n = 2; n < SP__LMAX + 1; ++n) {
    cosnt.col(n) =
        2.0 * cosnt.col(n - 1).cwiseProduct(cosnt.col(1)) - cosnt.col(n - 2);
    sinnt.col(n) =
        2.0 * sinnt.col(n - 1).cwiseProduct(cosnt.col(1)) - sinnt.col(n - 2);
  }
  int n = 0;
  for (int l = 0; l < SP__LMAX + 1; ++l) {
    for (int m = -l; m < 0; ++m) {
      cosmt.col(n) = cosnt.col(-m);
      sinmt.col(n) = -sinnt.col(-m);
      TM2.col(l * l + l + m) =
          T.col(l * l + l + m).cwiseProduct(M.col(l * l + l - m));
      T_r.col(l * l + l + m) = T.col(l * l + l - m);
      mmat.col(n).setConstant(m);
      ++n;
    }
    for (int m = 0; m < l + 1; ++m) {
      cosmt.col(n) = cosnt.col(m);
      sinmt.col(n) = sinnt.col(m);
      TM2.col(l * l + l + m) =
          T.col(l * l + l + m).cwiseProduct(M.col(l * l + l - m));
      T_r.col(l * l + l + m) = T.col(l * l + l - m);
      mmat.col(n).setConstant(m);
      ++n;
    }
  }

  // d/dM
  for (n = 0; n < SP__N; ++n) {
    bM.row(n) = bf.cwiseProduct(sinmt.col(n)).sum() * T_r.row(n) +
                bf.cwiseProduct(cosmt.col(n)).sum() * T.row(n);
  }

  // d/dtheta
  btheta = bf.cwiseProduct(
      (-mmat.cwiseProduct(sinmt) * TM1 + mmat.cwiseProduct(cosmt) * TM2)
          .rowwise()
          .sum());
}

} // namespace wigner
} // namespace sp

#endif