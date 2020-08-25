/**
\file latitude.h
\brief Latitude integral functions.

*/

#ifndef _SP_LATITUDE_H
#define _SP_LATITUDE_H

#include <constants.h>
#include <utils.h>

namespace sp {
namespace latitude {

using namespace utils;

/**
  The Gauss hypergeometric function 2F1.

*/
template <typename Scalar, typename Vector, typename RowMatrix>
inline void ComputeLatitudeIntegrals(const Scalar &alpha, const Scalar &beta,
                                     Vector &q, Vector &dqda, Vector &dqdb,
                                     RowMatrix &Q, RowMatrix &dQda,
                                     RowMatrix &dQdb) {

  q.setZero();
  dqda.setZero();
  dqdb.setZero();
  Q.setZero();
  dQda.setZero();
  dQdb.setZero();

  // TODO
}

} // namespace latitude
} // namespace sp

#endif