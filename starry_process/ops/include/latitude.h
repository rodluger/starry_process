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
template <typename S, typename V, typename M>
inline void computeLatitudeIntegrals(const S &alpha, const S &beta, V &q,
                                     V &dqda, V &dqdb, M &Q, M &dQda, M &dQdb) {

  // Dimensions
  const int N = (SP_LMAX + 1) * (SP_LMAX + 1);
  const int n = 4 * SP_LMAX + 1;
  int n1, n2, j1, i1, j2, i2;

  // Helper matrices
  Vector<S, n> B;
  Vector<S, n> F;
  RowMatrix<S, n, n> term;

  // Initialize the output
  q.setZero();
  dqda.setZero();
  dqdb.setZero();
  Q.setZero();
  dQda.setZero();
  dQdb.setZero();

  // B functions
  B(0) = 1.0;
  for (int k = 1; k < n; ++k) {
    B(k) = (alpha - 1.0 + k) / (alpha + beta - 1.0 + k) * B(k - 1);
  }

  // F functions
  S ab = alpha + beta;
  F(0) = sqrt(2.0) * hyp2f1(-0.5, beta, ab, 0.5);
  F(1) = sqrt(2.0) * hyp2f1(-0.5, beta, ab + 1.0, 0.5);
  for (int k = 2; k < n; ++k) {
    F(k) = ((ab + k - 1.0) / ((alpha + k - 1.0) * (ab + k - 0.5)) *
            ((ab + k - 2.0) * F(k - 2) + (1.5 - beta) * F(k - 1)));
  }
  F.array() = F.array().cwiseProduct(B.array()).eval();

  // Terms
  Map<Vector<S, n>> func(NULL);
  S fac1, fac2;
  term.setZero();
  for (int i = 0; i < n; ++i) {
    if (is_even(i)) {
      new (&func) Map<Vector<S, n>>(B.data());
      i2 = i / 2;
    } else {
      new (&func) Map<Vector<S, n>>(F.data());
      i2 = (i - 1) / 2;
    }
    for (int j = 0; j < n; j += 2) {
      j2 = j / 2;
      fac1 = 1.0;
      for (int k1 = 0; k1 < i2 + 1; ++k1) {
        fac2 = fac1;
        for (int k2 = 0; k2 < j2 + 1; ++k2) {
          term(i, j) += fac2 * func(k1 + k2);
          fac2 *= (k2 - j2) / (k2 + 1.0);
        }
        fac1 *= (i2 - k1) / (k1 + 1.0);
      }
    }
  }

  // Beta normalization
  term /= B(0);

  // Moment integrals
  n1 = 0;
  S inv_two_l1 = 1.0;
  S inv_two_l1l2;
  for (int l1 = 0; l1 < SP_LMAX + 1; ++l1) {
    for (int m1 = -l1; m1 < l1 + 1; ++m1) {
      j1 = m1 + l1;
      i1 = l1 - m1;
      q(n1) = term(j1, i1) * inv_two_l1;
      n2 = 0;
      inv_two_l1l2 = inv_two_l1;
      for (int l2 = 0; l2 < SP_LMAX + 1; ++l2) {
        for (int m2 = -l2; m2 < l2 + 1; ++m2) {
          j2 = m2 + l2;
          i2 = l2 - m2;
          Q(n1, n2) = term(j1 + j2, i1 + i2) * inv_two_l1l2;
          n2 += 1;
        }
        inv_two_l1l2 *= 0.5;
      }
      n1 += 1;
    }
    inv_two_l1 *= 0.5;
  }

  // TODO: derivatives!
}

} // namespace latitude
} // namespace sp

#endif