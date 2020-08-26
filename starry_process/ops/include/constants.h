/**
 * \file constants.h
 * \brief Hard-coded constants.
*/

#ifndef _SP_CONSTANTS_H_
#define _SP_CONSTANTS_H_

//! Degree of the Ylm expansion
#ifndef SP_LMAX
#define SP_LMAX 15
#endif

//! Spot profile correction constant
#ifndef SP_C0
#define SP_C0 0.5
#endif

//! Spot profile correction constant
#ifndef SP_C1
#define SP_C1 1.0
#endif

//! Spot profile correction constant
#ifndef SP_C2
#define SP_C2 0.0
#endif

//! Spot profile correction constant
#ifndef SP_C3
#define SP_C3 0.0
#endif

//! Spot profile correction constant
#define SP_Q (SP_C1 / SP_C0)

//! Spot profile correction constant
#define SP_P (1 + SP_C0 + SP_C1)

//! Spot profile correction constant
#define SP_Z (-SP_C1 / (1 + SP_C0))

//! Spot profile correction constant
#define SP_ZBAR (SP_C1 / (1 + SP_C0 + SP_C1))

//! Maximum number of iterations when computing 2F1
#ifndef SP_2F1_MAXITER
#define SP_2F1_MAXITER 200
#endif

//! Tolerance (max) when computing 2F1
#ifndef SP_2F1_MAXTOL
#define SP_2F1_MAXTOL 1e-15
#endif

//! Tolerance (min) when computing 2F1
#ifndef SP_2F1_MINTOL
#define SP_2F1_MINTOL 1e-12
#endif

//! Tolerance to avoid div by zero when computing G
#ifndef SP_G_DIV_BY_ZERO_TOL
#define SP_G_DIV_BY_ZERO_TOL 1e-6
#endif

#endif