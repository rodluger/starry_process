/**
 * \file latitude.h
 * \brief Latitude integral functions.
*/

#ifndef _SP_LATITUDE_H
#define _SP_LATITUDE_H

#include "constants.h"
#include "utils.h"

namespace sp {
namespace latitude {

using namespace utils;
using special::hyp2f1;

/**
 * Compute the mean `q` and variance `Q` latitude integrals.
*/
template <typename SCALAR, typename VECTOR, typename MATRIX>
inline void computeLatitudeIntegrals(const SCALAR &alpha, const SCALAR &beta,
                                     VECTOR &q, VECTOR &dqda, VECTOR &dqdb,
                                     MATRIX &Q, MATRIX &dQda, MATRIX &dQdb) {

  // Dimensions
  const int n = 4 * SP__LMAX + 1;
  int n1, n2, j1, i1, j2, i2;

  // Helper matrices
  Vector<SCALAR, n> B;
  Vector<SCALAR, n> dBda, dBdb;
  Vector<SCALAR, n> F;
  Vector<SCALAR, n> dFda, dFdb;
  RowMatrix<SCALAR, n, n> term;
  RowMatrix<SCALAR, n, n> dtermda, dtermdb;
  SCALAR c1, c2, c3, c4, c5, c6, c7, c8, c9;

  // Initialize the output
  q.setZero();
  dqda.setZero();
  dqdb.setZero();
  Q.setZero();
  dQda.setZero();
  dQdb.setZero();

  // B functions
  B(0) = 1.0;
  dBda(0) = 0.0;
  dBdb(0) = 0.0;

  for (int k = 1; k < n; ++k) {
    c1 = 1.0 / (alpha + beta + k - 1.0);
    c2 = (alpha + k - 1.0) * c1;
    c3 = beta * c1 * c1;
    c4 = (1 - k - alpha) * c1 * c1;
    B(k) = c2 * B(k - 1);
    dBda(k) = c3 * B(k - 1) + c2 * dBda(k - 1);
    dBdb(k) = c4 * B(k - 1) + c2 * dBdb(k - 1);
  }

  // F functions
  SCALAR ab = alpha + beta;
  SCALAR a2 = alpha * alpha;
  SCALAR dfdb, dfdc;
  F(0) = sqrt(2.0) * hyp2f1(-0.5, beta, ab, 0.5, dfdb, dfdc);
  dFda(0) = sqrt(2.0) * dfdc;
  dFdb(0) = sqrt(2.0) * (dfdb + dfdc);
  F(1) = sqrt(2.0) * hyp2f1(-0.5, beta, ab + 1.0, 0.5, dfdb, dfdc);
  dFda(1) = sqrt(2.0) * dfdc;
  dFdb(1) = sqrt(2.0) * (dfdb + dfdc);
  for (int k = 2; k < n; ++k) {
    // Value
    c1 = (ab + k - 1.0) / ((alpha + k - 1.0) * (ab + k - 0.5));
    c2 = c1 * (ab + k - 2.0);
    c3 = c1 * (1.5 - beta);
    F(k) = c2 * F(k - 2) + c3 * F(k - 1);

    // d / dalpha
    c6 = 1.0 / (2.0 * ab + 2.0 * k - 1.0);
    c6 *= c6;
    c7 = 1.0 / (alpha + k - 1.0);
    c7 *= c7;
    c4 = 6.0 * c6 + 2.0 * (1.0 - beta) * beta * c7;
    c4 /= (1.0 + 2.0 * beta);
    c5 =
        (2.0 * beta - 3.0) * (c7 * c6) *
        (2.0 - 3.0 * beta - 4.0 * k +
         2.0 * (a2 + 2.0 * alpha * (beta + k - 1.0) + (beta + k) * (beta + k)));
    dFda(k) =
        (c4 * F(k - 2) + c2 * dFda(k - 2)) + (c5 * F(k - 1) + c3 * dFda(k - 1));

    // d / dbeta
    c8 = (4.0 * (a2 + (beta + k) * (beta + k) +
                 alpha * (2.0 * beta + 2.0 * k - 1.0)) -
          2.0 * (1.0 + 2.0 * beta + 2.0 * k)) *
         c6 / (alpha + k - 1.0);
    c9 = (1.0 + 4.0 * beta + 6.0 * k -
          2.0 * (2.0 * a2 + 2.0 * (beta + k) * (beta + k) +
                 alpha * (4.0 * beta + 4.0 * k - 3.0))) *
         c6 / (alpha + k - 1.0);
    dFdb(k) =
        (c8 * F(k - 2) + c2 * dFdb(k - 2)) + (c9 * F(k - 1) + c3 * dFdb(k - 1));
  }
  F.array() = F.array().cwiseProduct(B.array()).eval();
  dFda.array() = dFda.array().cwiseProduct(B.array()).eval() +
                 F.array().cwiseProduct(dBda.array()).eval();
  dFdb.array() = dFdb.array().cwiseProduct(B.array()).eval() +
                 F.array().cwiseProduct(dBdb.array()).eval();

  // Terms
  Map<Vector<SCALAR, n>> func(NULL), dfuncda(NULL), dfuncdb(NULL);
  SCALAR fac1, fac2;
  term.setZero();
  dtermda.setZero();
  dtermdb.setZero();
  for (int i = 0; i < n; ++i) {
    if (is_even(i)) {
      new (&func) Map<Vector<SCALAR, n>>(B.data());
      new (&dfuncda) Map<Vector<SCALAR, n>>(dBda.data());
      new (&dfuncdb) Map<Vector<SCALAR, n>>(dBdb.data());
      i2 = i / 2;
    } else {
      new (&func) Map<Vector<SCALAR, n>>(F.data());
      new (&dfuncda) Map<Vector<SCALAR, n>>(dFda.data());
      new (&dfuncdb) Map<Vector<SCALAR, n>>(dFdb.data());
      i2 = (i - 1) / 2;
    }
    for (int j = 0; j < n; j += 2) {
      j2 = j / 2;
      fac1 = 1.0;
      for (int k1 = 0; k1 < i2 + 1; ++k1) {
        fac2 = fac1;
        for (int k2 = 0; k2 < j2 + 1; ++k2) {
          term(i, j) += fac2 * func(k1 + k2);
          dtermda(i, j) += fac2 * dfuncda(k1 + k2);
          dtermdb(i, j) += fac2 * dfuncdb(k1 + k2);
          fac2 *= (k2 - j2) / (k2 + 1.0);
        }
        fac1 *= (i2 - k1) / (k1 + 1.0);
      }
    }
  }

  // Moment integrals
  n1 = 0;
  SCALAR inv_two_l1 = 1.0;
  SCALAR inv_two_l1l2;
  for (int l1 = 0; l1 < SP__LMAX + 1; ++l1) {
    for (int m1 = -l1; m1 < l1 + 1; ++m1) {
      j1 = m1 + l1;
      i1 = l1 - m1;
      q(n1) = term(j1, i1) * inv_two_l1;
      dqda(n1) = dtermda(j1, i1) * inv_two_l1;
      dqdb(n1) = dtermdb(j1, i1) * inv_two_l1;
      n2 = 0;
      inv_two_l1l2 = inv_two_l1;
      for (int l2 = 0; l2 < SP__LMAX + 1; ++l2) {
        for (int m2 = -l2; m2 < l2 + 1; ++m2) {
          j2 = m2 + l2;
          i2 = l2 - m2;
          Q(n1, n2) = term(j1 + j2, i1 + i2) * inv_two_l1l2;
          dQda(n1, n2) = dtermda(j1 + j2, i1 + i2) * inv_two_l1l2;
          dQdb(n1, n2) = dtermdb(j1 + j2, i1 + i2) * inv_two_l1l2;
          n2 += 1;
        }
        inv_two_l1l2 *= 0.5;
      }
      n1 += 1;
    }
    inv_two_l1 *= 0.5;
  }
}

} // namespace latitude
} // namespace sp

#endif