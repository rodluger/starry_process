/**
 * \file constants.h
 * \brief Hard-coded constants.
*/

#ifndef _SP_CONSTANTS_H_
#define _SP_CONSTANTS_H_

// ----------------
// SYSTEM CONSTANTS
// ----------------

//! Degree of the Ylm expansion
#ifndef SP__LMAX
#define SP__LMAX 15
#endif

//! Number of Ylm terms
#define SP__N ((SP__LMAX + 1) * (SP__LMAX + 1))

//! Number of terms in the Wigner rotation matrix
#define SP__NWIG                                                               \
  (((SP__LMAX + 1) * (2 * SP__LMAX + 1) * (2 * SP__LMAX + 3)) / 3)

//! Spot profile correction constant
#ifndef SP__C0
#define SP__C0 0.152490623061794
#endif

//! Spot profile correction constant
#ifndef SP__C1
#define SP__C1 4.584365957466125
#endif

//! Spot profile correction constant
#ifndef SP__C2
#define SP__C2 0.124027269107698
#endif

//! Spot profile correction constant
#ifndef SP__C3
#define SP__C3 61.471223120013967
#endif

//! Spot profile correction constant
#define SP__Q (SP__C1 / SP__C0)

//! Spot profile correction constant
#define SP__P (1.0 + SP__C0 + SP__C1)

//! Spot profile correction constant
#define SP__Z (-SP__C1 / (1.0 + SP__C0))

//! Spot profile correction constant
#define SP__ZBAR (SP__C1 / SP__P)

//! Eigendecomposition tolerance
#ifndef SP__EIGH_MINDIFF
#define SP__EIGH_MINDIFF 1.0e-15
#endif

// --------------
// USER CONSTANTS
// --------------

//! Maximum number of iterations when computing 2F1
#ifndef SP_2F1_MAXITER
#define SP_2F1_MAXITER 500
#endif

//! Tolerance (max) when computing 2F1
#ifndef SP_2F1_MAXTOL
#define SP_2F1_MAXTOL 1e-15
#endif

//! Tolerance (max) when computing 2F1 derivs
#ifndef SP_2F1_MAXDTOL
#define SP_2F1_MAXDTOL 1e-13
#endif

//! Tolerance (min) when computing 2F1
#ifndef SP_2F1_MINTOL
#define SP_2F1_MINTOL 1e-12
#endif

//! Tolerance (min) when computing 2F1 derivs
#ifndef SP_2F1_MINDTOL
#define SP_2F1_MINDTOL 1e-10
#endif

//! Compute all Gjkm numerically? Slower but more stable.
#ifndef SP_COMPUTE_G_NUMERICALLY
#define SP_COMPUTE_G_NUMERICALLY 1
#endif

//! Tolerance to avoid div by zero when computing G
#ifndef SP_G_DIV_BY_ZERO_TOL
#define SP_G_DIV_BY_ZERO_TOL 1e-6
#endif

//! Recurse upward in m? Disabled by default, since it can be unstable
#ifndef SP_G_RECURSE_UPWARD_IN_M
#define SP_G_RECURSE_UPWARD_IN_M 0
#endif

//! Wigner rotation matrix tolerance
#ifndef SP_WIGNER_TOL
#define SP_WIGNER_TOL 1.0e-14
#endif

#endif