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
 * The Gauss hypergeometric function 2F1.
*/
template <typename T>
inline T hyp2f1(const T &a_, const T &b_, const T &c_, const T &z) {

  // Compute the value
  T a = a_;
  T b = b_;
  T c = c_;
  T term = a * b * z / c;
  T value = 1.0 + term;
  int n = 1;
  while ((abs(term / value) > SP_2F1_MAXTOL) && (n < SP_2F1_MAXITER)) {
    a += 1;
    b += 1;
    c += 1;
    n += 1;
    term *= a * b * z / c / n;
    value += term;
  }
  if ((n == SP_2F1_MAXITER) && (abs(term / value) > SP_2F1_MINTOL)) {
    std::stringstream msg;
    msg << "Series for 2F1 did not converge (value = " << std::setprecision(9)
        << value << ", frac. error = " << abs(term / value) << ").";
    std::stringstream args;
    args << "a_ = " << a_ << ", "
         << "b_ = " << b_ << ", "
         << "c_ = " << c_ << ", "
         << "z = " << z;
    throw StarryProcessException(msg.str(), "special.h", "hyp2f1", args.str());
  }
  return value;
}

/**
 * The Euler Beta function.
*/
template <typename T> inline T EulerBeta(const T &alpha, const T &beta) {
  // TODO: Check the numerical precision of this!
  return exp(lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta));
}

} // namespace special
} // namespace sp

#endif