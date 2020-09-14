/**
\file eigh.h
\brief Eigendecomposition routines.

*/

#ifndef _SP_EIGH_H
#define _SP_EIGH_H

#include "constants.h"
#include "utils.h"

namespace sp {
namespace eigh {

using namespace utils;

template <typename VECTOR, typename SQUARE_MATRIX, typename MATRIX>
inline void eigh_grad(const SQUARE_MATRIX &x, const VECTOR &w, const MATRIX &v,
                      const VECTOR &W, const MATRIX &V, SQUARE_MATRIX &X) {
  using Scalar = typename VECTOR::Scalar;
  int M = x.rows();
  int neig = w.rows();

  RowMatrix<Scalar, Dynamic, Dynamic> g(M, M);
  Vector<Scalar, Dynamic> Gn(M);
  Scalar diff, fac;
  g.setZero();
  for (int n = 0; n < neig; ++n) {
    Gn.setZero();
    for (int m = 0; m < neig; ++m) {
      if (m != n) {
        diff = w(n) - w(m);
        // NOTE: If two eigenvalues `w` are the same (or very
        // close to each other), the gradient here is +/- inf.
        // (This is expected, I think.) However, the subsequent sum of
        // infinities can lead to NaNs.
        // The cases in which I've encountered this correspond
        // to eigenvalues that are extremely small, so I've found
        // that I get the correct result for the gradient if I
        // simply zero out their contributions.
        // TODO: Figure out a more rigorous workaround for this!
        if (abs(diff) > SP__EIGH_MINDIFF)
          fac = 1.0 / diff;
        else
          fac = 0.0;
        Gn += v.col(m) * (V.col(n).transpose().dot(v.col(m))) * fac;
      }
    }
    g += v.col(n) * (v.col(n) * W(n) + Gn).transpose();
  }

  /*
  Numpy's eigh(a, 'L') (eigh(a, 'U')) is a function of tril(a)
  (triu(a)) only.  This means that partial derivative of
  eigh(a, 'L') (eigh(a, 'U')) with respect to a[i,j] is zero
  for i < j (i > j).  At the same time, non-zero components of
  the gradient must account for the fact that variation of the
  opposite triangle contributes to variation of two elements
  of Hermitian (symmetric) matrix. The following line
  implements the necessary logic.
  */
  X = g.template triangularView<Eigen::Lower>();
  X += g.template triangularView<Eigen::StrictlyUpper>().transpose();
}

} // namespace eigh
} // namespace sp

#endif