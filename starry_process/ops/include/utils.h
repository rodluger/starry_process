/**
 * \file utils.h
 * \brief Miscellaneous utilities and definitions used throughout the code.
 */

#ifndef _SP_UTILS_H_
#define _SP_UTILS_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

namespace sp {
namespace utils {

//! Commonly used stuff
using std::abs;
using std::isinf;
using std::isnan;
using std::max;

//! Eigen shorthand
using Eigen::Dynamic;
using Eigen::Map;
template <typename Scalar, int M, int N>
using RowMatrix = Eigen::Matrix<Scalar, M, N, Eigen::RowMajor>;
template <typename Scalar, int M, int N>
using ColMatrix = Eigen::Matrix<Scalar, M, N, Eigen::ColMajor>;
template <typename Scalar, int N> using Vector = Eigen::Matrix<Scalar, N, 1>;
template <typename Scalar, int N> using RowVector = Eigen::Matrix<Scalar, 1, N>;

//! Check if a number is even (or doubly, triply, quadruply... even)
inline bool is_even(int n, int ntimes = 1) {
  for (int i = 0; i < ntimes; i++) {
    if ((n % 2) != 0)
      return false;
    n /= 2;
  }
  return true;
}

template <size_t Lower, size_t Upper> struct for_bounds {
  static constexpr const size_t lower = Lower;
  static constexpr const size_t upper = Upper;
};

namespace for_constexpr_detail {
template <size_t lower, size_t... Is, class F>
void for_constexpr_impl(F &&f, std::index_sequence<Is...> /*meta*/) {
  (void)std::initializer_list<char>{
      ((void)f(std::integral_constant<size_t, Is + lower>{}), '0')...};
}
} // namespace for_constexpr_detail

/**
 * Compile-time for loop. From https://nilsdeppe.com/posts/for-constexpr
 * Requires C++14
 */
template <class Bounds0, class F> void for_constexpr(F &&f) {
  for_constexpr_detail::for_constexpr_impl<Bounds0::lower>(
      std::forward<F>(f),
      std::make_index_sequence<Bounds0::upper - Bounds0::lower>{});
}

/**
 * Generic starry exception class.
 */
class StarryProcessException : public std::exception {

  std::string m_msg;

  std::string bold(const char *msg) {
    std::stringstream boldmsg;
    boldmsg << "\e[1m" << msg << "\e[0m";
    return boldmsg.str();
  }

  std::string url(const char *msg) {
    std::stringstream urlmsg;
    urlmsg << "\e[1m\e[34m" << msg << "\e[0m\e[39m";
    return urlmsg.str();
  }

public:
  StarryProcessException(const std::string &msg, const std::string &file,
                         const std::string &function, const std::string &args)
      : m_msg(std::string("Something went wrong in starry! \n\n") +
              bold("Error: ") + msg + std::string("\n") + bold("File: ") +
              file + std::string("\n") + bold("Function: ") + function +
              std::string("\n") + bold("Arguments: ") + args +
              std::string("\n") +
              std::string(
                  "If you believe this is a bug, please open an issue at ") +
              url("https://github.com/rodluger/starry_process/issues/new. ") +
              std::string("Include the information above and a minimum working "
                          "example. \n")) {}

  virtual const char *what() const throw() { return m_msg.c_str(); }
};

} // namespace utils
} // namespace sp

#endif