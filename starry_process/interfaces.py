from .compat import USE_AESARA
import pymc3 as pm
from pymc3.blocking import ArrayOrdering, DictToArrayBijection
from pymc3.model import Point

if USE_AESARA:
    from pymc3.aesaraf import inputvars
else:
    from pymc3.theanof import inputvars
import pymc3_ext as pmx
from pymc3_ext import (
    optimize,
    get_theano_function_for_var,
    get_args_for_theano_function,
)
from pymc3.util import update_start_vals, get_default_varnames
from tqdm.auto import tqdm
import numpy as np


__all__ = ["MCMCInterface"]


class MCMCInterface:
    """
    An interface for using a ``pymc3`` model with a plain vanilla MCMC sampler.



    Args:
        model (optional): The ``pymc3`` model. If ``None`` (default), uses the
            current model on the stack.

    """

    def __init__(self, model=None):

        # Get the model
        self.model = pm.modelcontext(model)

        # Get the variables
        self.varnames = get_default_varnames(self.model.unobserved_RVs, False)

        # Get the starting point
        self.start = Point(self.model.test_point, model=self.model)
        self.ndim = len(self.start)
        self.mean = None
        self.cov = None

        # Compile the log probability function
        self.vars = inputvars(self.model.cont_vars)
        self.bij = DictToArrayBijection(ArrayOrdering(self.vars), self.start)
        self.func = get_theano_function_for_var(
            self.model.logpt, model=self.model
        )

    def optimize(self, **kwargs):
        """
        Maximize the log probability of a ``pymc3`` model.

        This routine wraps ``pymc3_ext.optimize``, which in turn
        wraps the ``scipy.optimize.minimize`` function. This method
        accepts any of the keywords accepted by either of those
        two functions.

        Returns:
            The array of parameter values at the optimum point.

        """
        self.map_soln, self.info = optimize(
            model=self.model, return_info=True, **kwargs
        )
        self.mean = self.info["x"]
        self.cov = self.info["hess_inv"]
        return self.mean

    def get_initial_state(
        self, nwalkers=30, var=None, check_finite=True, max_tries=100
    ):
        """
        Generate random initial points for sampling.

        If the ``optimize`` method was called beforehand, this method
        returns samples from a multidimensional Gaussian centered on
        the maximum a posteriori (MAP) solution with covariance equal
        to the inverse of the Hessian matrix at that point, unless
        ``var`` is provided, in which case that is used instead.
        If the optimizer was not called, this method
        returns samples from a Gaussian with mean equal to the
        model's test point (``model.test_point``) and variance equal to
        ``var``.

        Args:
            var (float, array, or matrix, optional): Variance of the
                multidimensional Gaussian used to draw samples.
                This quantity is optional if ``optimize`` was called
                beforehand, otherwise it must be provided.
                Default is ``None``.

        Returns:
            An array of shape ``(nwalkers, ndim)`` where ``ndim``
            is the number of free model parameters.

        """
        if var is None:
            if self.mean is not None and self.cov is not None:
                # User ran `optimize`, so let's sample from
                # the Laplacian approximation at the MAP point
                mean = self.mean
                cov = self.cov
            else:
                raise ValueError(
                    "Please provide a variance `var`, or run `optimize` before calling this method."
                )
        else:
            if self.mean is not None:
                # User ran `optimize`, so let's sample around
                # the MAP point
                mean = self.mean
            else:
                # Sample around the test value
                mean = self.bij.map(self.start)
            cov = var * np.eye(len(mean))

        # Sample from the Gaussian
        p0 = np.random.multivariate_normal(mean, cov, size=nwalkers)

        # Ensure the log probability is finite everywhere
        if check_finite:
            for k in range(nwalkers):
                n = 0
                while not np.isfinite(self.logp(p0[k])):
                    if n > max_tries:
                        raise ValueError(
                            "Unable to initialize walkers at a point with finite `logp`. "
                            "Try reducing `var` or running `optimize()`."
                        )
                    p0[k] = p.random.multivariate_normal(mean, cov)

        return p0

    def logp(self, x):
        """
        Return the log probability evaluated at a point.

        Args:
            x (array): The array of parameter values.

        Returns:
            The value of the log probability function evaluated at ``x``.
        """
        try:
            res = self.func(
                *get_args_for_theano_function(
                    self.bij.rmap(x), model=self.model
                )
            )
        except Exception:
            import traceback

            print("array:", x)
            print("point:", self.bij.rmap(x))
            traceback.print_exc()
            raise

        return res

    def transform(self, samples, varnames=None, progress=True):
        """
        Transform samples from the internal to the user parametrization.

        Args:
            samples (array or matrix): The set of points to transform.
            varnames (list, optional): The names of the parameters to
                transform to. These may either be strings or the actual
                ``pymc3`` model variables. If ``None`` (default), these
                are determined automatically and may be accessed as the
                ``varnames`` attribute of this class.
            progress (bool, optional): Display a progress bar? Default ``True``.

        Returns:
            An array of shape ``(..., len(varnames))``, where
            ``... = samples.shape[:-1]``, containing the transformed
            samples.
        """
        is_1d = len(np.shape(samples)) == 1
        samples = np.atleast_2d(samples)
        if varnames is None:
            varnames = self.varnames
        varnames = [v.name if not type(v) is str else v for v in varnames]
        shape = list(samples.shape)
        shape[-1] = len(varnames)
        x = np.zeros(shape)
        for k in tqdm(range(len(samples)), disable=not progress):
            point = pmx.optim.get_point(self, samples[k])
            for j, name in enumerate(varnames):
                x[k, j] = point[name]
        if is_1d:
            return x.flatten()
        else:
            return x
