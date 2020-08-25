/**
\file utils.h
\brief Miscellaneous utilities and definitions used throughout the code.

*/

#ifndef _SP_UTILS_H_
#define _SP_UTILS_H_

#include <Eigen/Core>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <vector>

namespace sp {
namespace utils {

using Eigen::Map;

template <typename Scalar, int M, int N>
using RowMatrix = Eigen::Matrix<Scalar, M, N, Eigen::RowMajor>;

template <typename Scalar, int N> using Vector = Eigen::Matrix<Scalar, N, 1>;

/**
  Generic starry exception class.

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