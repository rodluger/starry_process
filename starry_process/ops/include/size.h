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
const double W_data[] = {SP__C0,
                         SP__C0,
                         0,
                         0,
                         0,
                         (SP__C0 * SP__C0),
                         (2 * SP__C0 * SP__C0),
                         (SP__C0 * SP__C0),
                         0,
                         0,
                         (SP__C0 * SP__C0 * SP__C0),
                         (3 * SP__C0 * SP__C0 * SP__C0),
                         (3 * SP__C0 * SP__C0 * SP__C0),
                         (SP__C0 * SP__C0 * SP__C0),
                         0,
                         (SP__C0 * SP__C0 * SP__C0 * SP__C0),
                         (4 * SP__C0 * SP__C0 * SP__C0 * SP__C0),
                         (6 * SP__C0 * SP__C0 * SP__C0 * SP__C0),
                         (4 * SP__C0 * SP__C0 * SP__C0 * SP__C0),
                         (SP__C0 * SP__C0 * SP__C0 * SP__C0)};
const int IMAX = 3;
const int KMAX = 2;
const int MMAX = 4;
const int JMAX = 2 * SP__LMAX + 1;

template <typename Scalar>
inline RowMatrix<Scalar, KMAX + 1, MMAX + 1>
getGfac(const Scalar &alpha, const Scalar &beta,
        RowMatrix<Scalar, KMAX + 1, MMAX + 1> &dGfacda,
        RowMatrix<Scalar, KMAX + 1, MMAX + 1> &dGfacdb) {
  RowMatrix<Scalar, KMAX + 1, MMAX + 1> Gfac;
  Gfac.col(0).setOnes();
  dGfacda.setZero();
  dGfacdb.setZero();
  Scalar term1, term2, term3, frac;
  for (int m = 1; m < MMAX + 1; ++m) {
    for (int k = 0; k < KMAX + 1; ++k) {
      // Value
      term1 = SP__Q * (alpha + m - 1) / (alpha + beta + k * SP__C3 + m - 1);
      Gfac(k, m) = Gfac(k, m - 1) * term1;

      // d / dalpha
      frac = 1.0 / (alpha + beta + k * SP__C3 + m - 1);
      frac *= frac;
      term2 = SP__Q * (beta + k * SP__C3) * frac;
      dGfacda(k, m) = dGfacda(k, m - 1) * term1 + Gfac(k, m - 1) * term2;

      // d / dbeta
      term3 = SP__Q * (1 - alpha - m) * frac;
      dGfacdb(k, m) = dGfacdb(k, m - 1) * term1 + Gfac(k, m - 1) * term3;
    }
  }

  Scalar EB, dEBda, dEBdb;
  EB = EulerBeta(alpha, beta, dEBda, dEBdb);
  Scalar norm = SP__C2 / EB;
  Scalar dnormda = -norm / EB * dEBda;
  Scalar dnormdb = -norm / EB * dEBdb;
  for (int k = 1; k < KMAX + 1; ++k) {
    EB = EulerBeta(alpha, beta + k * SP__C3, dEBda, dEBdb);
    dGfacda.row(k) = dGfacda.row(k) * norm * EB +
                     Gfac.row(k) * (dnormda * EB + norm * dEBda);
    dGfacdb.row(k) = dGfacdb.row(k) * norm * EB +
                     Gfac.row(k) * (dnormdb * EB + norm * dEBdb);
    Gfac.row(k) *= norm * EB;
    norm *= SP__C2;
    dnormda *= SP__C2;
    dnormdb *= SP__C2;
  }

  return Gfac;
}

template <typename Scalar>
inline Scalar G_num(const Scalar &alpha, const Scalar &beta, const int j,
                    const int k, const int m, Scalar &dGda, Scalar &dGdb) {
  Scalar dfdb, dfdc;
  Scalar G = hyp2f1(1.0 + j, beta + k * SP__C3, alpha + beta + k * SP__C3 + m,
                    SP__ZBAR, dfdb, dfdc);
  dGda = dfdc;
  dGdb = dfdb + dfdc;
  Scalar fac = pow((1.0 + SP__C0) * (1.0 - SP__Z), -(1 + j));
  G *= fac;
  dGda *= fac;
  dGdb *= fac;
  return G;
}

template <typename Scalar>
inline Scalar arec(const Scalar &alpha, const Scalar &beta, const int j,
                   const int k, const int m) {
  return -((alpha + beta + k * SP__C3 + m - j) * (1.0 + SP__C0)) /
         (j * SP__P * (1.0 + SP__C0) * (1.0 + SP__C0));
}

template <typename Scalar>
inline Scalar brec(const Scalar &alpha, const Scalar &beta, const int j,
                   const int k, const int m) {
  return -(1.0 -
           ((alpha + beta + k * SP__C3 + m - j) * (1.0 + SP__C0) +
            (alpha + m) * SP__C1) /
               (j * SP__P)) /
         (1.0 + SP__C0);
}

template <typename Scalar>
inline void
computeG(const Scalar &alpha, const Scalar &beta,
         Vector<RowMatrix<Scalar, JMAX + 1, MMAX + 1>, KMAX + 1> &G,
         Vector<RowMatrix<Scalar, JMAX + 1, MMAX + 1>, KMAX + 1> &dGda,
         Vector<RowMatrix<Scalar, JMAX + 1, MMAX + 1>, KMAX + 1> &dGdb) {

  // Initialize
  RowMatrix<Scalar, KMAX + 1, MMAX + 1> dGfacda, dGfacdb;
  RowMatrix<Scalar, KMAX + 1, MMAX + 1> Gfac =
      getGfac(alpha, beta, dGfacda, dGfacdb);
  Vector<Scalar, JMAX - 1> B;
  Vector<Scalar, JMAX - 2> A;
  RowMatrix<Scalar, JMAX - 1, JMAX - 1> M;
  M.setZero();
  M.diagonal(1).setOnes();
  Vector<Scalar, JMAX - 1> x;
  x.setZero();

  if (SP_COMPUTE_G_NUMERICALLY) {

    // Compute G numerically for all j, k, m
    Scalar dfdb, dfdc;
    for (int k = 0; k < KMAX + 1; ++k) {
      for (int m = 0; m < MMAX + 1; ++m) {
        for (int j = 0; j < JMAX + 1; ++j) {
          G[k](j, m) =
              G_num(alpha, beta, j, k, m, dGda[k](j, m), dGdb[k](j, m));
        }
        dGda[k].col(m) =
            dGda[k].col(m) * Gfac(k, m) + G[k].col(m) * dGfacda(k, m);
        dGdb[k].col(m) =
            dGdb[k].col(m) * Gfac(k, m) + G[k].col(m) * dGfacdb(k, m);
        G[k].col(m) *= Gfac(k, m);
      }
    }

  } else {

    // TODO: Implement the gradients for this branch. Tedious but
    // straightforward.
    throw StarryProcessException(
        "Gradient of `G` integral not yet implemented.", "size.h", "computeG",
        "n/a");
    Scalar dfda, dfdb;

    // Solve by recursion

    // Solve the problem for each value of `k`
    for (int k = 0; k < KMAX + 1; ++k) {

      // Boundary conditions
      G[k](0, 0) = G_num(alpha, beta, 0, k, 0, dfda, dfdb);
      G[k](0, 1) = G_num(alpha, beta, 0, k, 1, dfda, dfdb);
      G[k](JMAX, 0) = G_num(alpha, beta, JMAX, k, 0, dfda, dfdb);
      G[k](JMAX, 1) = G_num(alpha, beta, JMAX, k, 1, dfda, dfdb);

      std::vector<int> zero_and_JMAX = {0, JMAX};
      for (int &j : zero_and_JMAX) {

        for (int m = 2; m < MMAX + 1; ++m) {

          if (SP_G_RECURSE_UPWARD_IN_M) {

            // Recurse upward in m
            Scalar div1 = alpha + m - 1;
            Scalar div2 = (alpha + beta + k * SP__C3 - j + m - 2) * SP__C1;

            // Be careful about division by zero here
            if ((abs(div1) < SP_G_DIV_BY_ZERO_TOL) ||
                (abs(div2) < SP_G_DIV_BY_ZERO_TOL)) {

              G[k](j, m) = G_num(alpha, beta, j, k, m, dfda, dfdb);

            } else {

              Scalar term = (alpha + beta + k * SP__C3 + m - 1) / div1;
              Scalar am = ((alpha + beta + k * SP__C3 + m - 2) * (1 + SP__C0)) *
                          (term / div2);
              Scalar bm = term -
                          ((alpha + beta + k * SP__C3 + m - 2) * (1 + SP__C0) +
                           (beta + k * SP__C3) * SP__C1) *
                              (term / div2);
              G[k](j, m) = am * G[k](j, m - 2) + bm * G[k](j, m - 1);
            }

          } else {

            // Compute numerically
            G[k](j, m) = G_num(alpha, beta, j, k, m, dfda, dfdb);
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
        // NOTE: Fix-sized block not working here for some reason.
        G[k].block(1, m, JMAX - 1, 1) = M.lu().solve(x);

        // Finally, apply the amplitude factor
        G[k].col(m) *= Gfac(k, m);
      }
    }
  }
}

template <typename Scalar>
inline void
computeH(const Scalar &alpha, const Scalar &beta,
         Vector<RowMatrix<Scalar, IMAX + 1, JMAX + 1>, KMAX + 1> &H,
         Vector<RowMatrix<Scalar, IMAX + 1, JMAX + 1>, KMAX + 1> &dHda,
         Vector<RowMatrix<Scalar, IMAX + 1, JMAX + 1>, KMAX + 1> &dHdb) {
  Vector<RowMatrix<Scalar, JMAX + 1, MMAX + 1>, KMAX + 1> G, dGda, dGdb;
  RowMatrix<Scalar, IMAX + 1, MMAX + 1> W(W_data);
  computeG(alpha, beta, G, dGda, dGdb);
  for (int k = 0; k < KMAX + 1; ++k) {
    for (int i = 0; i < IMAX + 1; ++i) {
      H[k].row(i) = G[k] * W.row(i).transpose();
      dHda[k].row(i) = dGda[k] * W.row(i).transpose();
      dHdb[k].row(i) = dGdb[k] * W.row(i).transpose();
    }
  }
}

/**
 * Compute the mean `q` and variance `Q` size integrals.
*/
template <typename SCALAR, typename VECTOR, typename MATRIX>
inline void computeSizeIntegrals(const SCALAR &alpha, const SCALAR &beta,
                                 VECTOR &q, VECTOR &dqda, VECTOR &dqdb,
                                 MATRIX &Q, MATRIX &dQda, MATRIX &dQdb) {

  // Initialize the output
  q.setZero();
  dqda.setZero();
  dqdb.setZero();
  Q.setZero();
  dQda.setZero();
  dQdb.setZero();

  // Hypergeometric sequences
  Vector<RowMatrix<SCALAR, IMAX + 1, JMAX + 1>, KMAX + 1> H, dHda, dHdb;
  computeH(alpha, beta, H, dHda, dHdb);
  std::function<SCALAR(const int, const int)> J =
      [&](const int i, const int j) { return H[0](i, j) + H[1](i, j); };
  std::function<SCALAR(const int, const int)> dJda =
      [&](const int i, const int j) { return dHda[0](i, j) + dHda[1](i, j); };
  std::function<SCALAR(const int, const int)> dJdb =
      [&](const int i, const int j) { return dHdb[0](i, j) + dHdb[1](i, j); };
  std::function<SCALAR(const int, const int)> K = [&](const int i,
                                                      const int j) {
    return H[0](i, j) + 2 * H[1](i, j) + H[2](i, j);
  };
  std::function<SCALAR(const int, const int)> dKda = [&](const int i,
                                                         const int j) {
    return dHda[0](i, j) + 2 * dHda[1](i, j) + dHda[2](i, j);
  };
  std::function<SCALAR(const int, const int)> dKdb = [&](const int i,
                                                         const int j) {
    return dHdb[0](i, j) + 2 * dHdb[1](i, j) + dHdb[2](i, j);
  };

  // Case 1: l = 0
  q(0) = -0.5 * J(0, 0);
  dqda(0) = -0.5 * dJda(0, 0);
  dqdb(0) = -0.5 * dJdb(0, 0);

  // Case 1: l = l' = 0
  Q(0, 0) = 0.25 * K(1, 1);
  dQda(0, 0) = 0.25 * dKda(1, 1);
  dQdb(0, 0) = 0.25 * dKdb(1, 1);

  // Outer loop
  for (int l = 1; l < SP__LMAX + 1; ++l) {

    int n = l * (l + 1);
    SCALAR sql = 1.0 / sqrt(2 * l + 1.0);

    // Case 2: l > 0
    q(n) = -0.5 * sql * (2 * J(0, l) + J(1, l));
    dqda(n) = -0.5 * sql * (2 * dJda(0, l) + dJda(1, l));
    dqdb(n) = -0.5 * sql * (2 * dJdb(0, l) + dJdb(1, l));

    // Case 2: l > 0, l' = 0
    Q(n, 0) = 0.25 * sql * (2 * K(1, l + 1) + K(2, l + 1));
    dQda(n, 0) = 0.25 * sql * (2 * dKda(1, l + 1) + dKda(2, l + 1));
    dQdb(n, 0) = 0.25 * sql * (2 * dKdb(1, l + 1) + dKdb(2, l + 1));
    Q(0, n) = Q(n, 0);
    dQda(0, n) = dQda(n, 0);
    dQdb(0, n) = dQdb(n, 0);

    // Inner loop
    for (int lp = 1; lp < l + 1; ++lp) {

      int np = lp * (lp + 1);
      SCALAR sqlp = 1.0 / sqrt(2 * lp + 1.0);

      // Case 3: l > 0, l' > 0
      int v = l + lp + 1;
      Q(n, np) = (0.25 * sql * sqlp * (4 * K(1, v) + 4 * K(2, v) + K(3, v)));
      dQda(n, np) =
          (0.25 * sql * sqlp * (4 * dKda(1, v) + 4 * dKda(2, v) + dKda(3, v)));
      dQdb(n, np) =
          (0.25 * sql * sqlp * (4 * dKdb(1, v) + 4 * dKdb(2, v) + dKdb(3, v)));
      Q(np, n) = Q(n, np);
      dQda(np, n) = dQda(n, np);
      dQdb(np, n) = dQdb(n, np);
    }
  }
}

} // namespace size
} // namespace sp

#endif