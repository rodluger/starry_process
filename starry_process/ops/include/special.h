/**
\file special.h
\brief Special functions.

*/

#ifndef _SP_SPECIAL_H
#define _SP_SPECIAL_H

#include "constants.h"
#include "utils.h"

namespace sp {
namespace special {

using namespace utils;

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
    if (abs(term / value) > SP_2F1_MINTOL) {
      std::stringstream msg;
      msg << "Series for 2F1 did not converge (value = " << std::setprecision(9)
          << value << ", frac. error = " << abs(term / value) << ").";
      std::stringstream args;
      args << "a_ = " << a_ << ", "
           << "b_ = " << b_ << ", "
           << "c_ = " << c_ << ", "
           << "z = " << z;
      throw StarryProcessException(msg.str(), "special.h", "hyp2f1",
                                   args.str());
    } else if (abs(dtermdb / dfdb) > SP_2F1_MINTOL) {
      std::stringstream msg;
      msg << "Series for d2F1/db did not converge (value = "
          << std::setprecision(9) << dfdb
          << ", frac. error = " << abs(dtermdb / dfdb) << ").";
      std::stringstream args;
      args << "a_ = " << a_ << ", "
           << "b_ = " << b_ << ", "
           << "c_ = " << c_ << ", "
           << "z = " << z;
      throw StarryProcessException(msg.str(), "special.h", "hyp2f1",
                                   args.str());
    } else {
      std::stringstream msg;
      msg << "Series for d2F1/dc did not converge (value = "
          << std::setprecision(9) << dfdc
          << ", frac. error = " << abs(dtermdc / dfdc) << ").";
      std::stringstream args;
      args << "a_ = " << a_ << ", "
           << "b_ = " << b_ << ", "
           << "c_ = " << c_ << ", "
           << "z = " << z;
      throw StarryProcessException(msg.str(), "special.h", "hyp2f1",
                                   args.str());
    }
  }

  return value;
}

/**
 * The Euler Beta function.
*/
template <typename T> inline T EulerBeta(const T &alpha, const T &beta) {
  return exp(lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta));
}

} // namespace special
} // namespace sp

#endif