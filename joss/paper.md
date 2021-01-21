---
title: "starry_process: Interpretable Gaussian processes for stellar light curves"
tags:
  - Python
  - astronomy
authors:
  - name: Rodrigo Luger
    orcid: 0000-0002-0296-3826
    affiliation: "1, 2"
  - name: Daniel Foreman-Mackey
    orcid: 0000-0002-9328-5652
    affiliation: 1
  - name: Christina Hedges
    orcid: 0000-0002-3385-8391
    affiliation: "3, 4"
affiliations:
  - name: Center for Computational Astrophysics, Flatiron Institute, New York, NY
    index: 1
  - name: Virtual Planetary Laboratory, University of Washington, Seattle, WA
    index: 2
  - name: Bay Area Environmental Research Institute, P.O. Box 25, Moffett Field, CA 94035, USA
    index: 3
  - name: NASA Ames Research Center, Moffett Field, CA
    index: 4

date: 21 January 2021
bibliography: bib.bib
---

# Statement of need

Mapping the surfaces of stars using time series measurements is a fundamental
problem in modern time-domain stellar astrophysics. This inverse problem is
ill-posed and computationally intractable, but in the associated AAS Journals
publication submitted in parallel to this paper, we derive an interpretable
effective Gaussian Process (GP) model for this problem that enables robust
probabilistic characterization of stellar surfaces using photometric time series
observations. Implementation of this model requires the efficient evaluation of
a set of special functions and recursion relations that are not readily
available in existing probabilistic programming frameworks. The `starry_process`
package provides the necessary elements to perform this analysis with existing
and forthcoming astronomical datasets.

# Summary

We implement our interpretable GP in the open-source, user-friendly `Python`
package `starry_process`, which can be installed via `pip` or from source on
[GitHub](https://github.com/rodluger/starry_process). The code is thoroughly
unit-tested and well documented, with examples on how to use the GP in custom
inference problems. As discussed in the associated AAS Journals publication,
users can choose, among other options, whether or not to marginalize over
inclination and whether or not to model a normalized process. Users can also
choose the spherical harmonic degree of the expansion, although it is
recommended to use $l_\mathrm{max} = 15$ (see below). Users may compute the mean
vector and covariance matrix in either the spherical harmonic basis or the flux
basis, or they may sample from it or use it to compute marginal likelihoods.
Arbitrary order limb darkening is implemented following @Agol2020.

The code was designed to maximize the speed and numerical stability of the
computation. Although most of the computations of the expectation integrals
involve many layers of nested sums over all spherical harmonic coefficients,
these may be expressed as high-dimensional tensor products, which can be
evaluated efficiently on modern hardware. Many of the expressions can also be
either pre-computed or computed recursively. To maximize the speed of the
algorithm, the code is implemented in hybrid `C++`/`Python` using the
just-in-time compilation capability of the `Theano` package [@Theano2016]. Since
all equations derived here have closed form expressions, these can be
differentiated in a straightforward and numerically stable manner, enabling the
computation of backpropagated gradients within `Theano`. As such,
`starry_process` is designed to work out-of-the box with `Theano`-based
inference tools such as `PyMC3` for NUTS/HMC or ADVI sampling [@Salvatier2016].

![Evaluation time in seconds for a single log-likelihood evaluation as a
function of the number of points $K$ in each light curve when conditioning on a
value of the inclination (blue) and when marginalizing over the inclination
(orange). At $l_\mathrm{max} = 15$, computation of the covariance matrix of the
GP takes about 20ms on a 2018 MacBook Pro. The dashed line shows the asymptotic
scaling of the algorithm, which is due to the Cholesky factorization and solve
operations.\label{fig:speed}](figures/speed.pdf)

Figure \ref{fig:speed} shows the computational scaling of the `Python`
implementation of the algorithm for the case where we condition the GP on a
specific value of the inclination (blue) and the case where we marginalize over
inclination (orange). Both curves show the time in seconds to compute the
likelihood (averaged over many trials) as a function of the number of points $K$
in a single light curve. For $K \lesssim 100$, the computation time is constant
at $10-20$ ms for both algorithms. This is the approximate time (on a typical
modern laptop) taken to compute the GP covariance matrix given a set of
hyperparameters $\pmb{\theta}_\bullet$. For larger values of $K$, the cost
approaches a scaling of $K^{2.6}$, which is dominated by the factorization of
the covariance matrix and the solve operation to compute the likelihood. The
likelihood marginalized over inclination is only slightly slower to compute,
thanks to the tricks discussed in the associated publication in the AAS
Journals.

Many modern GP packages [e.g., @Ambikasaran2015; @ForemanMackey2017] have
significantly better asymptotic scalings, but these are usually due to specific
structure imposed on the kernel functions, such as the assumption of
stationarity. Our kernel structure is determined by the physics (or perhaps more
accurately, the geometry) of stellar surfaces, and its nonstationarity is a
consequence of the normalization step in relative photometry. Moreover, and
unlike the typical kernels used for GP regression, our kernel is a nontrivial
function of the hyperparameters $\pmb{\theta}_\bullet$, so its computation is
necessarily more expensive. Nevertheless, the fact that our GP may be used for
likelihood evaluation in a small fraction of a second for typical datasets ($K
\sim 1{,000}$) makes it extremely useful for inference. Recall that we are
implicitly marginalizing over all of the properties of \emph{every spot} on the
surface of the star.

In ensemble analyses, we must compute the likelihood of each of the $M$ light
curves conditioned on $\pmb{\theta}_\bullet$. In practice, each star will have a
different rotation period, different limb darkening coefficients, and different
photometric uncertainty, meaning we must factorize $M$ different covariance
matrices. Fortunately, the spherical harmonic covariance,
$\pmb{\Sigma}_\mathbf{y}$ need only be computed once; this can then be linearly
transformed into the flux basis for each light curve. As the evaluation of
$\pmb{\Sigma}_\mathbf{y}$ is the computationally-intensive step, the likelihood
evaluation typically scales sub-linearly with $M$. Furthermore, it is possible
to marginalize over the period and limb darkening coefficients, which would
remove the scaling with $M$ entirely if the photometric precision were the same
for all light curves. Even if it is not, the algorithm can still be greatly sped
up, although an implementation of this is deferred to the next paper.

![Log of the condition number of the covariance in the spherical harmonic basis,
$\pmb{\Sigma}_{\mathbf{y}}$, as a function of the spherical harmonic degree of
the expansion, $l_\mathrm{max}$. Different lines correspond to different values
of $\pmb{\theta}_\bullet$ drawn from a uniform prior (see text for details). In
the majority of the cases, the matrix becomes ill-conditioned above
$l_\mathrm{max} = 15$\label{fig:stability}](figures/stability.pdf)

Our algorithm is also numerically stable over nearly all of the prior volume up
to $l_\mathrm{max} = 15$. Figure \ref{fig:stability} shows the log of the
condition number of the covariance matrix in the spherical harmonic basis,
$\pmb{\Sigma}_\mathbf{y}$, as a function of the spherical harmonic degree of the
expansion for 100 draws from a uniform prior over $r \in [10^\circ, 45^\circ]$,
$\mu_\phi \in [0^\circ, 85^\circ]$, $\sigma_\phi \in [5^\circ, 40^\circ]$, and
$n \in [1, 50]$. The condition number is nearly constant up to $l_\mathrm{max} =
15$ in almost all cases; above this value, the algorithm suddenly becomes
unstable and the covariance is ill-conditioned. The instability occurs within
the computation of the latitude and longitude moment integrals and is likely due
to the large number of operations involving linear combinations of
hypergeometric and gamma functions. While it may be possible to achieve
stability at higher values of $l_\mathrm{max}$ via careful reparametrization of
some of those equations, we find that $l_\mathrm{max} = 15$ is high enough for
most practical purposes. We plan to revisit this point in future work.

Finally, instabilities can also occur if $\sigma_\phi$ is too small and/or $n$
is too large. Values of $\sigma_\phi \lesssim 1^\circ$ lead to instabilities in
the computation of the hypergeometric functions, while values of $n \gtrsim 50$
can sometimes cause the Cholesky factorization of the covariance to fail
(although this can be mitigated by adding a small quantity to the diagonal to
ensure positive-semidefinitess). In cases where the algorithm goes (very)
unstable, the log-likelihood evaluation returns $-\infty$: in other words, they
are silently rejected by an implicit prior. Fortunately, these cases are likely
unphysical: in practice, there should always be some finite amount of variance
in the latitudes of spots, and stars with more than 50 spots are likely too
spotted for individual spots to be discernible in the first place. Instead, we
are likely sensitive to \emph{groups} of spots, which our GP is flexible enough
to model.

# Acknowledgements

# References
