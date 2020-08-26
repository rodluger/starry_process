/**
 * \file size.h
 * \brief Size integral functions.
*/

#ifndef _SP_SIZE_H
#define _SP_SIZE_H

#include "constants.h"
#include "utils.h"

namespace sp {
namespace size {

using namespace utils;
using special::hyp2f1;
using special::EulerBeta;

// Dimensions & constants
const double W_data[] = {SP_C0,
                         SP_C0,
                         0,
                         0,
                         0,
                         (SP_C0 * SP_C0),
                         (2 * SP_C0 * SP_C0),
                         (SP_C0 * SP_C0),
                         0,
                         0,
                         (SP_C0 * SP_C0 * SP_C0),
                         (3 * SP_C0 * SP_C0 * SP_C0),
                         (3 * SP_C0 * SP_C0 * SP_C0),
                         (SP_C0 * SP_C0 * SP_C0),
                         0,
                         (SP_C0 * SP_C0 * SP_C0 * SP_C0),
                         (4 * SP_C0 * SP_C0 * SP_C0 * SP_C0),
                         (6 * SP_C0 * SP_C0 * SP_C0 * SP_C0),
                         (4 * SP_C0 * SP_C0 * SP_C0 * SP_C0),
                         (SP_C0 * SP_C0 * SP_C0 * SP_C0)};
const int IMAX = 3;
const int KMAX = 2;
const int MMAX = 4;
const int JMAX = 2 * SP_LMAX + 1;

template <typename Scalar>
inline RowMatrix<Scalar, KMAX + 1, MMAX + 1> getGfac(const Scalar &alpha,
                                                     const Scalar &beta) {
  RowMatrix<Scalar, KMAX + 1, MMAX + 1> lam;
  lam.col(0).setOnes();
  for (int m = 1; m < MMAX + 1; ++m) {
    for (int k = 0; k < KMAX + 1; ++k) {
      lam(k, m) = (lam(k, m - 1) * SP_Q * (alpha + m - 1) /
                   (alpha + beta + k * SP_C3 + m - 1));
    }
  }
  Scalar norm = SP_C2 / EulerBeta(alpha, beta);
  for (int k = 1; k < KMAX + 1; ++k) {
    lam.row(k) *= norm * EulerBeta(alpha, beta + k * SP_C3);
    norm *= SP_C2;
  }
  return lam;
}

template <typename Scalar>
inline Scalar G_num(const Scalar &alpha, const Scalar &beta, const int j,
                    const int k, const int m) {
  Scalar G =
      hyp2f1(1.0 + j, beta + k * SP_C3, alpha + beta + k * SP_C3 + m, SP_ZBAR);
  G /= pow((1.0 + SP_C0) * (1.0 - SP_Z), 1 + j);
  return G;
}

template <typename Scalar>
inline Scalar arec(const Scalar &alpha, const Scalar &beta, const int j,
                   const int k, const int m) {
  return -((alpha + beta + k * SP_C3 + m - j) * (1.0 + SP_C0)) /
         (j * SP_P * (1.0 + SP_C0) * (1.0 + SP_C0));
}

template <typename Scalar>
inline Scalar brec(const Scalar &alpha, const Scalar &beta, const int j,
                   const int k, const int m) {
  return -(1.0 -
           ((alpha + beta + k * SP_C3 + m - j) * (1.0 + SP_C0) +
            (alpha + m) * SP_C1) /
               (j * SP_P)) /
         (1.0 + SP_C0);
}

template <typename Scalar>
inline void
computeG(const Scalar &alpha, const Scalar &beta,
         Vector<RowMatrix<Scalar, JMAX + 1, MMAX + 1>, KMAX + 1> &G) {

  // Initialize
  RowMatrix<Scalar, KMAX + 1, MMAX + 1> Gfac = getGfac(alpha, beta);
  Vector<Scalar, JMAX - 1> B;
  Vector<Scalar, JMAX - 2> A;
  RowMatrix<Scalar, JMAX - 1, JMAX - 1> M;
  M.setZero();
  M.diagonal(1).setOnes();
  Vector<Scalar, JMAX - 1> x;
  x.setZero();

  // Solve the problem for each value of `k`
  for (int k = 0; k < KMAX + 1; ++k) {

    // Boundary conditions
    G[k](0, 0) = G_num(alpha, beta, 0, k, 0);
    G[k](0, 1) = G_num(alpha, beta, 0, k, 1);
    G[k](JMAX, 0) = G_num(alpha, beta, JMAX, k, 0);
    G[k](JMAX, 1) = G_num(alpha, beta, JMAX, k, 1);

    // Recurse upward in m
    std::vector<int> zero_and_JMAX = {0, JMAX};
    for (int &j : zero_and_JMAX) {

      for (int m = 2; m < MMAX + 1; ++m) {

        // Be careful about division by zero here
        if (abs(alpha + beta + k * SP_C3 - j + m - 2) < SP_G_DIV_BY_ZERO_TOL) {

          G[k](j, m) = G_num(alpha, beta, j, k, m);

        } else {

          Scalar term = (alpha + m - 1) / (alpha + beta + k * SP_C3 + m - 1);
          Scalar am = ((alpha + beta + k * SP_C3 + m - 2) * (1 + SP_C0)) /
                      ((alpha + beta + k * SP_C3 - j + m - 2) * term * SP_C1);
          Scalar bm =
              1.0 / term -
              ((alpha + beta + k * SP_C3 + m - 2) * (1 + SP_C0) + beta +
               k * SP_C3 * SP_C1) /
                  ((alpha + beta + k * SP_C3 - j + m - 2) * term * SP_C1);
          G[k](j, m) = am * G[k](j, m - 2) + bm * G[k](j, m - 1);
        }
      }
    }

    // Recurse along the j dimension @ each m
    // We're solving the tridiagonal matrix system `M G = x`
    for (int m = 0; m < MMAX + 1; ++m) {

      // Populate the tridiagonal matrix
      for (int j = 2; j < JMAX + 1; ++j) {
        B(j - 2) = brec(alpha, beta, j, k, m);
      }
      for (int j = 3; j < JMAX + 1; ++j) {
        A(j - 3) = arec(alpha, beta, j, k, m);
      }
      M.diagonal(0) = B;
      M.diagonal(-1) = A;

      // Populate the `data` vector
      x(0) = -arec(alpha, beta, 2, k, m) * G[k](0, m);
      x(JMAX - 2) = -G[k](JMAX, m);

      // Solve
      // TODO: We should probably use a sparse solve here!
      // TODO: Fix-sized block not working here!?
      G[k].block(1, m, JMAX - 1, 1) = M.lu().solve(x);

      // Finally, apply the amplitude factor
      G[k].col(m) *= Gfac(k, m);
    }
  }
}

template <typename Scalar>
inline void
computeH(const Scalar &alpha, const Scalar &beta,
         Vector<RowMatrix<Scalar, IMAX + 1, JMAX + 1>, KMAX + 1> &H) {
  Vector<RowMatrix<Scalar, JMAX + 1, MMAX + 1>, KMAX + 1> G;
  RowMatrix<Scalar, IMAX + 1, MMAX + 1> W(W_data);
  computeG(alpha, beta, G);
  for (int k = 0; k < KMAX + 1; ++k) {
    for (int i = 0; i < IMAX + 1; ++i) {
      H[k].row(i) = G[k] * W.row(i).transpose();
    }
  }
}

/**
 * Compute the mean `q` and variance `Q` size integrals.
*/
template <typename S, typename V, typename M>
inline void computeSizeIntegrals(const S &alpha, const S &beta, V &q, V &dqda,
                                 V &dqdb, M &Q, M &dQda, M &dQdb) {

  // Dimensions & constants
  const int N = (SP_LMAX + 1) * (SP_LMAX + 1);

  // Initialize the output
  q.setZero();
  dqda.setZero();
  dqdb.setZero();
  Q.setZero();
  dQda.setZero();
  dQdb.setZero();

  // Hypergeometric sequences
  Vector<RowMatrix<S, IMAX + 1, JMAX + 1>, KMAX + 1> H;
  computeH(alpha, beta, H);
  std::function<S(const int, const int)> J = [&](const int i, const int j) {
    return H[0](i, j) + H[1](i, j);
  };
  std::function<S(const int, const int)> K = [&](const int i, const int j) {
    return H[0](i, j) + 2 * H[1](i, j) + H[2](i, j);
  };

  // Case 1: l = 0
  q(0) = -0.5 * J(0, 0);

  // Case 1: l = l' = 0
  Q(0, 0) = 0.25 * K(1, 1);

  // Outer loop
  for (int l = 1; l < SP_LMAX + 1; ++l) {

    int n = l * (l + 1);
    S sql = 1.0 / sqrt(2 * l + 1.0);

    // Case 2: l > 0
    q(n) = -0.5 * sql * (2 * J(0, l) + J(1, l));

    // Case 2: l > 0, l' = 0
    Q(n, 0) = 0.25 * sql * (2 * K(1, l + 1) + K(2, l + 1));
    Q(0, n) = Q(l, 0);

    // Inner loop
    for (int lp = 1; lp < l + 1; ++lp) {

      int np = lp * (lp + 1);
      S sqlp = 1.0 / sqrt(2 * lp + 1.0);

      // Case 3: l > 0, l' > 0
      int q = l + lp + 1;
      Q(np, np) = (0.25 * sql * sqlp * (4 * K(1, q) + 4 * K(2, q) + K(3, q)));
      Q(np, n) = Q(n, np);
    }
  }

  // TODO: derivatives
}

} // namespace size
} // namespace sp

#endif