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

//! Degree of limb darkening
#ifndef SP__UMAX
#define SP__UMAX 2
#endif

//! Total degree
#define SP__LUMAX (SP__LMAX + SP__UMAX)

//! Number of Ylm terms
#define SP__N ((SP__LMAX + 1) * (SP__LMAX + 1))

//! Number of Ylm + LD terms
#define SP__NLU ((SP__LMAX + SP__UMAX + 1) * (SP__LMAX + SP__UMAX + 1))

//! Number of terms in the Wigner rotation matrix
#define SP__NWIG                                                               \
  (((SP__LMAX + 1) * (2 * SP__LMAX + 1) * (2 * SP__LMAX + 3)) / 3)

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

//! Wigner rotation matrix tolerance
#ifndef SP_WIGNER_TOL
#define SP_WIGNER_TOL 1.0e-14
#endif

#endif