/**
 * \file size.h
 * \brief Size integral functions.
*/

#ifndef _SP_SIZE_H
#define _SP_SIZE_H

#include "constants.h"
#include "utils.h"

namespace sp {
namespace size {

using namespace utils;
using special::hyp2f1;
using special::EulerBeta;

/**
 * Compute the mean `q` and variance `Q` size integrals.
*/
template <int LMAX, typename S, typename V, typename M>
inline void ComputeSizeIntegrals(const S &alpha, const S &beta, V &q, V &dqda,
                                 V &dqdb, M &Q, M &dQda, M &dQdb) {

  // Dimensions
  const int N = (LMAX + 1) * (LMAX + 1);

  // Initialize the output
  q.setZero();
  dqda.setZero();
  dqdb.setZero();
  Q.setZero();
  dQda.setZero();
  dQdb.setZero();

  // TODO
}

} // namespace size
} // namespace sp

#endif