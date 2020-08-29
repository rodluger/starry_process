/**
\file flux.h
\brief Flux matrices.

*/

#ifndef _SP_FLUX_H
#define _SP_FLUX_H

#include "constants.h"
#include "utils.h"

namespace sp {
namespace flux {

using namespace utils;

/**
Compute the `r^T` phase curve solution vector.

*/
template <typename Scalar> void computerT(RowVector<Scalar, SP__N> &rT) {
  Scalar amp0, amp, lfac1, lfac2;
  int mu, nu;
  rT.setZero();
  amp0 = M_PI;
  lfac1 = 1.0;
  lfac2 = 2.0 / 3.0;
  for (int l = 0; l < SP__LMAX + 1; l += 4) {
    amp = amp0;
    for (int m = 0; m < l + 1; m += 4) {
      mu = l - m;
      nu = l + m;
      rT(l * l + l + m) = amp * lfac1;
      rT(l * l + l - m) = amp * lfac1;
      if (l < SP__LMAX) {
        rT((l + 1) * (l + 1) + l + m + 1) = amp * lfac2;
        rT((l + 1) * (l + 1) + l - m + 1) = amp * lfac2;
      }
      amp *= (nu + 2.0) / (mu - 2.0);
    }
    lfac1 /= (l / 2 + 2) * (l / 2 + 3);
    lfac2 /= (l / 2 + 2.5) * (l / 2 + 3.5);
    amp0 *= 0.0625 * (l + 2) * (l + 2);
  }
  amp0 = 0.5 * M_PI;
  lfac1 = 0.5;
  lfac2 = 4.0 / 15.0;
  for (int l = 2; l < SP__LMAX + 1; l += 4) {
    amp = amp0;
    for (int m = 2; m < l + 1; m += 4) {
      mu = l - m;
      nu = l + m;
      rT(l * l + l + m) = amp * lfac1;
      rT(l * l + l - m) = amp * lfac1;
      if (l < SP__LMAX) {
        rT((l + 1) * (l + 1) + l + m + 1) = amp * lfac2;
        rT((l + 1) * (l + 1) + l - m + 1) = amp * lfac2;
      }
      amp *= (nu + 2.0) / (mu - 2.0);
    }
    lfac1 /= (l / 2 + 2) * (l / 2 + 3);
    lfac2 /= (l / 2 + 2.5) * (l / 2 + 3.5);
    amp0 *= 0.0625 * l * (l + 4);
  }
}

/**
Multiply a polynomial vector/matrix by `z`.

*/
template <typename T1, typename T2>
inline void polymulz(int lmax, const Eigen::MatrixBase<T1> &p,
                     Eigen::MatrixBase<T2> &pz) {
  int n = 0;
  int lz, nz;
  bool odd1;
  pz.setZero();
  for (int l = 0; l < lmax + 1; ++l) {
    for (int m = -l; m < l + 1; ++m) {
      odd1 = (l + m) % 2 == 0 ? false : true;
      lz = l + 1;
      nz = lz * lz + lz + m;
      if (odd1) {
        pz.row(nz - 4 * lz + 2) += p.row(n);
        pz.row(nz - 2) -= p.row(n);
        pz.row(nz + 2) -= p.row(n);
      } else {
        pz.row(nz) += p.row(n);
      }
      ++n;
    }
  }
}

/**
Compute the `P(z)` part of the Ylm vectors.

*/
template <typename Scalar>
inline void legendre(int lmax,
                     std::vector<std::vector<Eigen::Triplet<Scalar>>> &M) {
  // Compute densely
  int N = (lmax + 1) * (lmax + 1);
  int ip, im;
  Scalar term = 1.0, fac = 1.0;
  Vector<Scalar, Dynamic> colvec(N);
  ColMatrix<Scalar, Dynamic, Dynamic> dnsM(N, N);
  dnsM.setZero();
  for (int m = 0; m < lmax + 1; ++m) {
    // 1
    ip = m * m + 2 * m;
    im = m * m;
    dnsM(0, ip) = fac;
    dnsM(0, im) = fac;
    if (m < lmax) {
      // z
      ip = m * m + 4 * m + 2;
      im = m * m + 2 * m + 2;
      dnsM(2, ip) = (2 * m + 1) * dnsM(m * m + 2 * m, 0);
      dnsM(2, im) = dnsM(2, ip);
    }
    for (int l = m + 1; l < lmax + 1; ++l) {
      // Recurse
      ip = l * l + l + m;
      im = l * l + l - m;
      polymulz(lmax - 1, dnsM.col((l - 1) * (l - 1) + l - 1 + m), colvec);
      dnsM.col(ip) = (2 * l - 1) * colvec / (l - m);
      if (l > m + 1)
        dnsM.col(ip) -=
            (l + m - 1) * dnsM.col((l - 2) * (l - 2) + l - 2 + m) / (l - m);
      dnsM.col(im) = dnsM.col(ip);
    }
    fac *= -term;
    term += 2;
  }

  // Store as triplets
  M.resize(N);
  for (int col = 0; col < N; ++col) {
    int n2 = 0;
    for (int l = 0; l < lmax + 1; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        if (dnsM(n2, col) != 0)
          M[col].push_back(Eigen::Triplet<Scalar>(l, m, dnsM(n2, col)));
        ++n2;
      }
    }
  }
}

/**
Compute the `theta(x, y)` term of the Ylm vectors.

*/
template <typename Scalar>
inline void theta(int lmax,
                  std::vector<std::vector<Eigen::Triplet<Scalar>>> &M) {
  int N = (lmax + 1) * (lmax + 1);
  M.resize(N);
  Scalar term1, term2;
  int n1, n2;
  for (int m = 0; m < lmax + 1; ++m) {
    term1 = 1.0;
    term2 = m;
    for (int j = 0; j < m + 1; j += 2) {
      if (j > 0) {
        term1 *= -(m - j + 1.0) * (m - j + 2.0) / (j * (j - 1.0));
        term2 *= -(m - j) * (m - j + 1.0) / (j * (j + 1.0));
      }
      for (int l = m; l < lmax + 1; ++l) {
        n1 = l * l + l + m;
        n2 = l * l + l - m;
        M[n1].push_back(Eigen::Triplet<Scalar>(m, 2 * j - m, term1));
        if (j < m) {
          M[n2].push_back(Eigen::Triplet<Scalar>(m, 2 * (j + 1) - m, term2));
        }
      }
    }
  }
}

/**
Compute the amplitudes of the Ylm vectors.

*/
template <typename Derived>
inline void amp(int lmax, Eigen::MatrixBase<Derived> &M) {
  M.setZero();
  typename Derived::Scalar inv_root_two = sqrt(0.5);
  for (int l = 0; l < lmax + 1; ++l) {
    M.col(l * l + l).setConstant(sqrt(2 * (2 * l + 1)));
    for (int m = 1; m < l + 1; ++m) {
      M.col(l * l + l + m) =
          -M.col(l * l + l + m - 1) / sqrt((l + m) * (l - m + 1));
      M.col(l * l + l - m) = M.col(l * l + l + m);
    }
    M.col(l * l + l) *= inv_root_two;
  }
  M /= (2 * sqrt(M_PI));
}

/**
Compute a sparse polynomial product using Eigen Triplets.

*/
template <typename Scalar>
inline void
computeSparsePolynomialProduct(const std::vector<Eigen::Triplet<Scalar>> &p1,
                               const std::vector<Eigen::Triplet<Scalar>> &p2,
                               std::vector<Eigen::Triplet<Scalar>> &p1p2) {
  using Triplet = Eigen::Triplet<Scalar>;
  int l1, m1, l2, m2;
  bool odd1;
  Scalar prod;
  p1p2.clear();
  for (Triplet t1 : p1) {
    l1 = t1.row();
    m1 = t1.col();
    odd1 = (l1 + m1) % 2 == 0 ? false : true;
    for (Triplet t2 : p2) {
      l2 = t2.row();
      m2 = t2.col();
      prod = t1.value() * t2.value();
      if (odd1 && ((l2 + m2) % 2 != 0)) {
        p1p2.push_back(Triplet(l1 + l2 - 2, m1 + m2, prod));
        p1p2.push_back(Triplet(l1 + l2, m1 + m2 - 2, -prod));
        p1p2.push_back(Triplet(l1 + l2, m1 + m2 + 2, -prod));
      } else {
        p1p2.push_back(Triplet(l1 + l2, m1 + m2, prod));
      }
    }
  }
}

/**
Compute the *sparse* change of basis matrix `A1`.

*/
template <typename Scalar>
inline void computeA1(int lmax, Eigen::SparseMatrix<Scalar> &A1) {
  using Triplet = Eigen::Triplet<Scalar>;
  using Triplets = std::vector<Triplet>;

  int N = (lmax + 1) * (lmax + 1);
  Scalar norm = 2.0 / sqrt(M_PI);

  // Amplitude
  ColMatrix<Scalar, Dynamic, Dynamic> C(N, N);
  amp(lmax, C);

  // Z terms
  std::vector<Triplets> t_Z(N);
  legendre(lmax, t_Z);

  // XY terms
  std::vector<Triplets> t_XY(N);
  theta(lmax, t_XY);

  // Construct the change of basis matrix
  Triplets t_M, coeffs;
  for (int col = 0; col < N; ++col) {
    // Multiply Z and XY
    computeSparsePolynomialProduct(t_Z[col], t_XY[col], t_M);

    // Parse the terms and store in `coeffs`
    for (Triplet term : t_M) {
      int l = term.row();
      int m = term.col();
      int row = l * l + l + m;
      Scalar value = term.value() * norm * C(row, col);
      coeffs.push_back(Triplet(row, col, value));
    }
  }
  A1.resize(N, N);
  A1.setFromTriplets(coeffs.begin(), coeffs.end());
}

/**
Compute the starry flux operator `rTA1`.

*/
template <typename VECTOR> inline void computerTA1(VECTOR &rTA1) {
  using Scalar = typename VECTOR::Scalar;
  RowVector<Scalar, SP__N> rT;
  computerT(rT);
  Eigen::SparseMatrix<Scalar> A1;
  computeA1(SP__LMAX, A1);
  rTA1 = rT * A1;
}

} // namespace flux
} // namespace sp

#endif