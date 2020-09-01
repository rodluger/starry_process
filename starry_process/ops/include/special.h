/**
\file special.h
\brief Special functions.

*/

#ifndef _SP_SPECIAL_H
#define _SP_SPECIAL_H

#include "constants.h"
#include "utils.h"

//! Euler constant
#define EULER 0.577215664901532860606512090082402431

namespace sp {
namespace special {

using namespace utils;

// Evaluation of the digamma function, adapted from `scipy`, which is in
// turn adapted from `boost`. Original code at
//    https://github.com/scipy/scipy/blob/
//    59347ae8b86bcc92c339efe213128f64ab6df98c/
//    scipy/special/cephes/psi.c
namespace digamma {

/**
 * Evaluates polynomial of degree N:
 *
 *                     2          N
 * y  =  C  + C x + C x  +...+ C x
 *        0    1     2          N
 *
 * Coefficients are stored in reverse order:
 *
 * coef[0] = C  , ..., coef[N] = C  .
 *            N                   0
*/
inline double polevl(const double &x, const double coef[], int N) {
  double ans;
  int i;
  const double *p;

  p = coef;
  ans = *p++;
  i = N;

  do
    ans = ans * x + *p++;
  while (--i);

  return (ans);
}

/**
 * Rational approximation on [1, 2] taken from Boost.
 *
 * Now for the approximation, we use the form:
 *
 * digamma(x) = (x - root) * (Y + R(x-1))
 *
 * Where root is the location of the positive root of digamma,
 * Y is a constant, and R is optimised for low absolute error
 * compared to Y.
 *
 * Maximum Deviation Found:               1.466e-18
 * At double precision, max error found:  2.452e-17
*/
inline double digamma_imp_1_2(const double &x) {
  double r, g;
  static const float Y = 0.99558162689208984f;
  static const double root1 = 1569415565.0 / 1073741824.0;
  static const double root2 = (381566830.0 / 1073741824.0) / 1073741824.0;
  static const double root3 = 0.9016312093258695918615325266959189453125e-19;
  static double P[] = {-0.0020713321167745952, -0.045251321448739056,
                       -0.28919126444774784,   -0.65031853770896507,
                       -0.32555031186804491,   0.25479851061131551};
  static double Q[] = {-0.55789841321675513e-6,
                       0.0021284987017821144,
                       0.054151797245674225,
                       0.43593529692665969,
                       1.4606242909763515,
                       2.0767117023730469,
                       1.0};
  g = x - root1;
  g -= root2;
  g -= root3;
  r = polevl(x - 1.0, P, 5) / polevl(x - 1.0, Q, 6);
  return g * Y + g * r;
}

/**
 * Asymptotic series expansion of psi(x)
*/
inline double psi_asy(const double &x) {
  double y, z;
  static double A[] = {8.33333333333333333333E-2, -2.10927960927960927961E-2,
                       7.57575757575757575758E-3, -4.16666666666666666667E-3,
                       3.96825396825396825397E-3, -8.33333333333333333333E-3,
                       8.33333333333333333333E-2};
  if (x < 1.0e17) {
    z = 1.0 / (x * x);
    y = z * polevl(z, A, 6);
  } else {
    y = 0.0;
  }
  return log(x) - (0.5 / x) - y;
}

/**
 * psi(x), the digamma function.
*/
inline double psi(const double &x_) {
  double y = 0.0;
  double q, r;
  int i, n;
  double x = x_;

  if (isnan(x)) {
    return x;
  } else if (isinf(x) && x > 0) {
    return x;
  } else if (isinf(x) && x < 0) {
    return NAN;
  } else if (x == 0) {
    return NAN;
  } else if (x < 0.0) {
    /* argument reduction before evaluating tan(pi * x) */
    r = modf(x, &q);
    if (r == 0.0) {
      return NAN;
    }
    y = -M_PI / tan(M_PI * r);
    x = 1.0 - x;
  }

  /* check for positive integer up to 10 */
  if ((x <= 10.0) && (x == floor(x))) {
    n = (int)x;
    for (i = 1; i < n; i++) {
      y += 1.0 / i;
    }
    y -= EULER;
    return y;
  }

  /* use the recurrence relation to move x into [1, 2] */
  if (x < 1.0) {
    y -= 1.0 / x;
    x += 1.0;
  } else if (x < 10.0) {
    while (x > 2.0) {
      x -= 1.0;
      y += 1.0 / x;
    }
  }
  if ((1.0 <= x) && (x <= 2.0)) {
    y += digamma_imp_1_2(x);
    return y;
  }

  /* x is large, use the asymptotic series */
  y += psi_asy(x);
  return y;
}
} // namespace digamma

/**
 * The Gauss hypergeometric function 2F1 and its `b` and `c` derivs.
*/
template <typename T>
inline T hyp2f1(const T &a_, const T &b_, const T &c_, const T &z, T &dfdb,
                T &dfdc) {

  T a = a_;
  T b = b_;
  T c = c_;
  T term = a * b * z / c;
  T dtermdb = a * z / c;
  T dtermdc = -term / c;
  T value = 1.0 + term;
  T fac1, fac2, fac3;
  dfdb = dtermdb;
  dfdc = dtermdc;
  int n = 1;
  while (((abs(term / value) > SP_2F1_MAXTOL) ||
          (abs(dtermdb / dfdb) > SP_2F1_MAXDTOL) ||
          (abs(dtermdc / dfdc) > SP_2F1_MAXDTOL)) &&
         (n < SP_2F1_MAXITER)) {
    a += 1;
    b += 1;
    c += 1;
    n += 1;
    fac1 = a * z / c / n;
    fac2 = fac1 * b;
    fac3 = -fac2 / c;
    dtermdb *= fac2;
    dtermdb += fac1 * term;
    dtermdc *= fac2;
    dtermdc += fac3 * term;
    term *= fac2;
    value += term;
    dfdb += dtermdb;
    dfdc += dtermdc;
  }
  if (n == SP_2F1_MAXITER) {
    std::stringstream args, msg;
    args << "a_ = " << a_ << ", "
         << "b_ = " << b_ << ", "
         << "c_ = " << c_ << ", "
         << "z = " << z;
    if (abs(term / value) > SP_2F1_MINTOL) {
      msg << "Series for 2F1 did not converge "
          << "(value = " << std::setprecision(9) << value
          << ", frac. error = " << abs(term / value) << ").";
    } else if (abs(dtermdb / dfdb) > SP_2F1_MINTOL) {
      msg << "Series for d2F1db did not converge "
          << "(value = " << std::setprecision(9) << dfdb
          << ", frac. error = " << abs(dtermdb / dfdb) << ").";
    } else {
      msg << "Series for d2F1dc did not converge "
          << "(value = " << std::setprecision(9) << dfdc
          << ", frac. error = " << abs(dtermdc / dfdc) << ").";
    }
    throw StarryProcessException(msg.str(), "special.h", "hyp2f1", args.str());
  }

  return value;
}

/**
 * The Euler Beta function.
*/
template <typename T>
inline T EulerBeta(const T &alpha, const T &beta, T &dfda, T &dfdb) {
  using digamma::psi;
  T EB = exp(lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta));
  T P = psi(alpha + beta);
  dfda = EB * (psi(alpha) - P);
  dfdb = EB * (psi(beta) - P);
  return EB;
}

} // namespace special
} // namespace sp

#endif