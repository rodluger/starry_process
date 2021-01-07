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
template <typename Scalar>
void computerT(const int deg, RowVector<Scalar, Dynamic> &rT) {
  Scalar amp0, amp, lfac1, lfac2;

  int mu, nu;
  rT.setZero((deg + 1) * (deg + 1));
  amp0 = M_PI;
  lfac1 = 1.0;
  lfac2 = 2.0 / 3.0;
  for (int l = 0; l < deg + 1; l += 4) {
    amp = amp0;
    for (int m = 0; m < l + 1; m += 4) {
      mu = l - m;
      nu = l + m;
      rT(l * l + l + m) = amp * lfac1;
      rT(l * l + l - m) = amp * lfac1;
      if (l < deg) {
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
  for (int l = 2; l < deg + 1; l += 4) {
    amp = amp0;
    for (int m = 2; m < l + 1; m += 4) {
      mu = l - m;
      nu = l + m;
      rT(l * l + l + m) = amp * lfac1;
      rT(l * l + l - m) = amp * lfac1;
      if (l < deg) {
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
inline void polymulz(int deg, const Eigen::MatrixBase<T1> &p,
                     Eigen::MatrixBase<T2> &pz) {
  int n = 0;
  int lz, nz;
  bool odd1;
  pz.setZero();
  for (int l = 0; l < deg + 1; ++l) {
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
inline void legendre(int deg,
                     std::vector<std::vector<Eigen::Triplet<Scalar>>> &M) {
  // Compute densely
  int N = (deg + 1) * (deg + 1);
  int ip, im;
  Scalar term = 1.0, fac = 1.0;
  Vector<Scalar, Dynamic> colvec(N);
  ColMatrix<Scalar, Dynamic, Dynamic> dnsM(N, N);
  dnsM.setZero();
  for (int m = 0; m < deg + 1; ++m) {
    // 1
    ip = m * m + 2 * m;
    im = m * m;
    dnsM(0, ip) = fac;
    dnsM(0, im) = fac;
    if (m < deg) {
      // z
      ip = m * m + 4 * m + 2;
      im = m * m + 2 * m + 2;
      dnsM(2, ip) = (2 * m + 1) * dnsM(m * m + 2 * m, 0);
      dnsM(2, im) = dnsM(2, ip);
    }
    for (int l = m + 1; l < deg + 1; ++l) {
      // Recurse
      ip = l * l + l + m;
      im = l * l + l - m;
      polymulz(deg - 1, dnsM.col((l - 1) * (l - 1) + l - 1 + m), colvec);
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
    for (int l = 0; l < deg + 1; ++l) {
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
inline void theta(int deg,
                  std::vector<std::vector<Eigen::Triplet<Scalar>>> &M) {
  int N = (deg + 1) * (deg + 1);
  M.resize(N);
  Scalar term1, term2;
  int n1, n2;
  for (int m = 0; m < deg + 1; ++m) {
    term1 = 1.0;
    term2 = m;
    for (int j = 0; j < m + 1; j += 2) {
      if (j > 0) {
        term1 *= -(m - j + 1.0) * (m - j + 2.0) / (j * (j - 1.0));
        term2 *= -(m - j) * (m - j + 1.0) / (j * (j + 1.0));
      }
      for (int l = m; l < deg + 1; ++l) {
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
inline void amp(int deg, Eigen::MatrixBase<Derived> &M) {
  M.setZero();
  typename Derived::Scalar inv_root_two = sqrt(0.5);
  for (int l = 0; l < deg + 1; ++l) {
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
inline void computeA1(int deg, Eigen::SparseMatrix<Scalar> &A1) {
  using Triplet = Eigen::Triplet<Scalar>;
  using Triplets = std::vector<Triplet>;

  int N = (deg + 1) * (deg + 1);
  Scalar norm = 2.0 / sqrt(M_PI);

  // Amplitude
  ColMatrix<Scalar, Dynamic, Dynamic> C(N, N);
  amp(deg, C);

  // Z terms
  std::vector<Triplets> t_Z(N);
  legendre(deg, t_Z);

  // XY terms
  std::vector<Triplets> t_XY(N);
  theta(deg, t_XY);

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

template <typename Scalar>
inline void computeA1Inv(int deg, const Eigen::SparseMatrix<Scalar> &A1,
                         Eigen::SparseMatrix<Scalar> &A1Inv) {
  int N = (deg + 1) * (deg + 1);
  Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
  solver.compute(A1);

  if (solver.info() != Eigen::Success)
    throw std::runtime_error(
        "Error computing the change of basis matrix `A1Inv`.");
  Eigen::SparseMatrix<Scalar> I =
      RowMatrix<Scalar, Dynamic, Dynamic>::Identity(N, N).sparseView();
  A1Inv = solver.solve(I);
}

/**
Compute the starry flux operator `rTA1`.

*/
template <typename VECTOR> inline void computerTA1(VECTOR &rTA1) {
  using Scalar = typename VECTOR::Scalar;
  RowVector<Scalar, Dynamic> rT;
  computerT(SP__LMAX, rT);
  Eigen::SparseMatrix<Scalar> A1;
  computeA1(SP__LMAX, A1);
  rTA1 = rT * A1;
}

/**
Limb darkening class.

*/
template <typename Scalar> class LimbDark {

public:
  RowVector<Scalar, Dynamic> rT;
  Eigen::SparseMatrix<Scalar> A1;
  Eigen::SparseMatrix<Scalar> A1Inv;
  Eigen::SparseMatrix<Scalar> A1NbyN;
  Eigen::SparseMatrix<Scalar> U1;
  Vector<Scalar, Dynamic> p;
  RowMatrix<Scalar, Dynamic, Dynamic> Lp;
  RowMatrix<Scalar, Dynamic, Dynamic> DDp;

  /**
  Compute the change of basis matrix `U1` from limb darkening coefficients
  to polynomial coefficients.

  */
  void computeU1() {
    Scalar twol, amp, lfac, lchoosek, fac0, fac;
    Scalar norm = 2.0 / sqrt(M_PI);
    ColMatrix<Scalar, Dynamic, Dynamic> U0;
    ColMatrix<Scalar, Dynamic, Dynamic> LT, YT;
    LT.setZero(SP__LUMAX + 1, SP__LUMAX + 1);
    YT.setZero(SP__LUMAX + 1, SP__LUMAX + 1);

    // Compute L^T
    for (int l = 0; l < SP__LUMAX + 1; ++l) {
      lchoosek = 1;
      for (int k = 0; k < l + 1; ++k) {
        if ((k + 1) % 2 == 0)
          LT(k, l) = lchoosek;
        else
          LT(k, l) = -lchoosek;
        lchoosek *= (l - k) / (k + 1.0);
      }
    }

    // Compute Y^T
    // Even terms
    twol = 1.0;
    lfac = 1.0;
    fac0 = 1.0;
    for (int l = 0; l < SP__LUMAX + 1; l += 2) {
      amp = twol * sqrt((2 * l + 1) / (4 * M_PI)) / lfac;
      lchoosek = 1;
      fac = fac0;
      for (int k = 0; k < l + 1; k += 2) {
        YT(k, l) = amp * lchoosek * fac;
        fac *= (k + l + 1.0) / (k - l + 1.0);
        lchoosek *= (l - k) * (l - k - 1) / ((k + 1.0) * (k + 2.0));
      }
      fac0 *= -0.25 * (l + 1) * (l + 1);
      lfac *= (l + 1.0) * (l + 2.0);
      twol *= 4.0;
    }
    // Odd terms
    twol = 2.0;
    lfac = 1.0;
    fac0 = 0.5;
    for (int l = 1; l < SP__LUMAX + 1; l += 2) {
      amp = twol * sqrt((2 * l + 1) / (4 * M_PI)) / lfac;
      lchoosek = l;
      fac = fac0;
      for (int k = 1; k < l + 1; k += 2) {
        YT(k, l) = amp * lchoosek * fac;
        fac *= (k + l + 1.0) / (k - l + 1.0);
        lchoosek *= (l - k) * (l - k - 1) / ((k + 1.0) * (k + 2.0));
      }
      fac0 *= -0.25 * (l + 2) * l;
      lfac *= (l + 1.0) * (l + 2.0);
      twol *= 4.0;
    }

    // Compute U0
    Eigen::HouseholderQR<ColMatrix<Scalar, Dynamic, Dynamic>> solver(
        SP__LUMAX + 1, SP__LUMAX + 1);
    solver.compute(YT);
    U0 = solver.solve(LT);

    // Normalize it. Since we compute `U0` from the *inverse*
    // of `A1`, we must *divide* by the normalization constant
    U0 /= norm;

    // Compute U1
    ColMatrix<Scalar, Dynamic, Dynamic> X(SP__NLU, SP__LUMAX + 1);
    X.setZero();
    for (int l = 0; l < SP__LUMAX + 1; ++l)
      X(l * (l + 1), l) = 1;
    Eigen::SparseMatrix<Scalar> XU0 = (X * U0).sparseView();
    U1 = A1 * XU0;

    // TODO: I think most of the computation above is wasteful!
    // Check whether we actually need to go up to `SP__LUMAX`.
    U1 = U1.block(0, 0, (SP__UMAX + 1) * (SP__UMAX + 1), SP__UMAX + 1);
  }

  /**
  Compute the limb darkening operator in the polynomial basis.

  */
  inline void computeLp() {
    bool odd1;
    int l, n;
    int n1 = 0, n2 = 0;
    Lp.setZero(SP__NLU, SP__N);
    for (int l1 = 0; l1 < SP__LMAX + 1; ++l1) {
      for (int m1 = -l1; m1 < l1 + 1; ++m1) {
        odd1 = (l1 + m1) % 2 == 0 ? false : true;
        n2 = 0;
        for (int l2 = 0; l2 < SP__UMAX + 1; ++l2) {
          for (int m2 = -l2; m2 < l2 + 1; ++m2) {
            l = l1 + l2;
            n = l * l + l + m1 + m2;
            if (odd1 && ((l2 + m2) % 2 != 0)) {
              Lp(n - 4 * l + 2, n1) += p(n2);
              Lp(n - 2, n1) -= p(n2);
              Lp(n + 2, n1) -= p(n2);
            } else {
              Lp(n, n1) += p(n2);
            }
            ++n2;
          }
        }
        ++n1;
      }
    }
  }

  /**
  Compute the gradient of the polynomial product matrix.
  This is independent of any user coefficients, so we can
  just pre-compute it!

  */
  inline void computeDDp() {
    bool odd1;
    int l, n;
    int n1 = 0, n2 = 0;
    Vector<RowMatrix<Scalar, Dynamic, Dynamic>, Dynamic> D;
    D.resize((SP__UMAX + 1) * (SP__UMAX + 1));
    for (n = 0; n < (SP__UMAX + 1) * (SP__UMAX + 1); ++n)
      D(n).setZero(SP__NLU, SP__N);
    for (int l1 = 0; l1 < SP__LMAX + 1; ++l1) {
      for (int m1 = -l1; m1 < l1 + 1; ++m1) {
        odd1 = (l1 + m1) % 2 == 0 ? false : true;
        n2 = 0;
        for (int l2 = 0; l2 < SP__UMAX + 1; ++l2) {
          for (int m2 = -l2; m2 < l2 + 1; ++m2) {
            l = l1 + l2;
            n = l * l + l + m1 + m2;
            if (odd1 && ((l2 + m2) % 2 != 0)) {
              D[n2](n - 4 * l + 2, n1) += 1;
              D[n2](n - 2, n1) -= 1;
              D[n2](n + 2, n1) -= 1;
            } else {
              D[n2](n, n1) += 1;
            }
            ++n2;
          }
        }
        ++n1;
      }
    }
    DDp.resize((SP__UMAX + 1) * (SP__UMAX + 1), SP__N);
    for (n = 0; n < (SP__UMAX + 1) * (SP__UMAX + 1); ++n)
      DDp.row(n) = rT * D(n) * A1NbyN;
  }

  explicit LimbDark() {

    if (SP__UMAX > 0) {
      // Pre-compute a bunch of stuff
      computerT(SP__LUMAX, rT);
      computeA1(SP__LUMAX, A1);
      computeA1Inv(SP__LUMAX, A1, A1Inv);
      computeU1();
      A1NbyN = A1.topLeftCorner(SP__N, SP__N);
      computeDDp();
    }
  };

  /**
  Compute the starry flux operator `rTA1` under limb darkening.

  */
  template <typename ROWVECTOR>
  void computerTA1L(const Vector<Scalar, SP__UMAX> &u, ROWVECTOR &rTA1L) {

    if (SP__UMAX == 0) {
      throw std::runtime_error("Limb darkening is currently disabled.");
    }

    // Compute the limb darkening polynomial
    Vector<Scalar, SP__UMAX + 1> u_;
    u_(0) = -1.0;
    u_.template tail<SP__UMAX>() = u;
    p = U1 * u_;
    Scalar norm = Scalar(1.0) /
                  rT.template head<(SP__UMAX + 1) * (SP__UMAX + 1)>().dot(p);
    p *= norm * M_PI;

    // Compute the limb darkening operator `L` in the polynomial basis
    computeLp();

    // Change of basis
    rTA1L = (rT * Lp) * A1NbyN;
  }

  /**
  Compute the starry flux operator `rTA1` under limb darkening (backprop pass).

  */
  template <typename ROWVECTOR, typename VECTOR>
  void computerTA1L(const Vector<Scalar, SP__UMAX> &u, ROWVECTOR &bf,
                    VECTOR &bu) {

    if (SP__UMAX == 0) {
      throw std::runtime_error("Limb darkening is currently disabled.");
    }

    // Compute the limb darkening polynomial
    Vector<Scalar, SP__UMAX + 1> u_;
    u_(0) = -1.0;
    u_.template tail<SP__UMAX>() = u;
    p = U1 * u_;
    Scalar norm = Scalar(1.0) /
                  rT.template head<(SP__UMAX + 1) * (SP__UMAX + 1)>().dot(p);
    p *= norm * M_PI;

    // Backprop p
    RowVector<Scalar, (SP__UMAX + 1) * (SP__UMAX + 1)> bp;
    bp = DDp * bf.transpose();

    // Compute the limb darkening derivatives
    RowMatrix<Scalar, Dynamic, Dynamic> DpDu =
        M_PI * norm * U1 -
        p * rT.template head<(SP__UMAX + 1) * (SP__UMAX + 1)>() * U1 * norm;
    bu = (bp * DpDu).template tail<SP__UMAX>();
  }

  /**
  Compute the limb darkening operator `L` (ylm -> ylm).

  This is not currently used in the code, but is made available
  for reference.

  */
  template <typename ROWMATRIX>
  void computeL(const Vector<Scalar, SP__UMAX> &u, ROWMATRIX &L) {

    if (SP__UMAX == 0) {
      throw std::runtime_error("Limb darkening is currently disabled.");
    }

    // Compute the limb darkening polynomial
    Vector<Scalar, SP__UMAX + 1> u_;
    u_(0) = -1.0;
    u_.template tail<SP__UMAX>() = u;
    p = U1 * u_;
    Scalar norm = Scalar(1.0) /
                  rT.template head<(SP__UMAX + 1) * (SP__UMAX + 1)>().dot(p);
    p *= norm * M_PI;

    // Compute the limb darkening operator `Lp` in the polynomial basis
    computeLp();

    // Change of basis
    L = (A1Inv * Lp) * A1NbyN;
  }
};

/**
  Compute the polynomial basis at a vector of points.

*/
template <typename ROWVECTOR, typename ROWMATRIX>
inline void computepT(const ROWVECTOR &x, const ROWVECTOR &y,
                      const ROWVECTOR &z, ROWMATRIX &pT) {
  // Dimensions
  using Scalar = typename ROWVECTOR::Scalar;
  size_t npts = x.size();
  pT.resize(npts, SP__N);

  // Optimized polynomial basis computation
  // A little opaque, sorry...
  RowMatrix<Scalar, Dynamic, Dynamic> xarr(npts, SP__N), yarr(npts, SP__N);
  RowVector<Scalar, Dynamic> xterm(npts), yterm(npts);
  xterm.setOnes();
  yterm.setOnes();
  xterm += 0.0 * z; // Ensures we get `nan`s off the disk
  yterm += 0.0 * z; // Ensures we get `nan`s off the disk
  int i0 = 0, di0 = 3, j0 = 0, dj0 = 2;
  int i, j, di, dj, n;
  for (n = 0; n < SP__LMAX + 1; ++n) {
    i = i0;
    di = di0;
    xarr.col(i) = xterm;
    j = j0;
    dj = dj0;
    yarr.col(j) = yterm;
    i = i0 + di - 1;
    j = j0 + dj - 1;
    while (i + 1 < SP__N) {
      xarr.col(i) = xterm;
      xarr.col(i + 1) = xterm;
      di += 2;
      i += di;
      yarr.col(j) = yterm;
      yarr.col(j + 1) = yterm;
      dj += 2;
      j += dj - 1;
    }
    xterm = xterm.cwiseProduct(x);
    i0 += 2 * n + 1;
    di0 += 2;
    yterm = yterm.cwiseProduct(y);
    j0 += 2 * (n + 1) + 1;
    dj0 += 2;
  }
  n = 0;
  for (int l = 0; l < SP__LMAX + 1; ++l) {
    for (int m = -l; m < l + 1; ++m) {
      pT.col(n) = xarr.col(n).cwiseProduct(yarr.col(n));
      if ((l + m) % 2 != 0)
        pT.col(n) = pT.col(n).cwiseProduct(z.transpose());
      ++n;
    }
  }
}

} // namespace flux
} // namespace sp

#endif
