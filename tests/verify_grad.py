import theano
from theano.gradient import *
from theano.tests.unittest_tools import seed_rng


def raise_error(msg):
    raise ValueError(msg)


def verify_grad(
    fun,
    pt,
    n_tests=1,
    rng=None,
    eps=None,
    out_type=None,
    abs_tol=None,
    rel_tol=None,
    mode=None,
    cast_to_output_type=False,
    no_debug_ref=True,
):
    """
    A version of `theano.tests.unittest_tools.verify_grad` with
    simpler error messages.

    """
    if rng is None:
        seed_rng()
        rng = np.random

    # The import is here to prevent circular import.
    from theano import compile, shared
    import theano.tensor
    from theano.tensor import as_tensor_variable, TensorType

    assert isinstance(pt, (list, tuple))
    pt = [np.array(p) for p in pt]

    for i, p in enumerate(pt):
        if p.dtype not in ("float16", "float32", "float64"):
            raise TypeError(
                (
                    "verify_grad can work only with floating point "
                    'inputs, but input %i has dtype "%s".'
                )
                % (i, p.dtype)
            )

    _type_tol = dict(  # relative error tolerances for different types
        float16=5e-2, float32=1e-2, float64=1e-4
    )

    if abs_tol is None:
        abs_tol = builtins.max(_type_tol[str(p.dtype)] for p in pt)
    if rel_tol is None:
        rel_tol = builtins.max(_type_tol[str(p.dtype)] for p in pt)

    if rng is None:
        raise TypeError(
            (
                "rng should be a valid instance of "
                "numpy.random.RandomState. You may "
                "want to use theano.tests.unittest"
                "_tools.verify_grad instead of "
                "theano.gradient.verify_grad."
            )
        )

    # We allow input downcast in function, because numeric_grad works in the
    # most precise dtype used among the inputs, so we may need to cast some.
    def function(inputs, output, name, mode=mode):
        f = compile.function(
            inputs,
            output,
            accept_inplace=True,
            allow_input_downcast=True,
            mode=mode,
            on_unused_input="ignore",
            name=name,
        )
        return f

    tensor_pt = [
        TensorType(
            as_tensor_variable(p).dtype, as_tensor_variable(p).broadcastable
        )(name="input %i" % i)
        for i, p in enumerate(pt)
    ]

    # fun can be either a function or an actual Op instance
    o_output = fun(*tensor_pt)

    if isinstance(o_output, list):
        raise NotImplementedError(
            ("cant (yet) autotest gradient of fun " "with multiple outputs")
        )
        # we could make loop over outputs making random projections R for each,
        # but this doesn't handle the case where not all the outputs are
        # differentiable... so I leave this as TODO for now -JB.

    o_fn = function(tensor_pt, o_output, name="gradient.py fwd")
    o_fn_out = o_fn(*[p.copy() for p in pt])

    if isinstance(o_fn_out, tuple) or isinstance(o_fn_out, list):
        raise TypeError(
            "It seems like you are trying to use verify_grad "
            "on an op or a function which outputs a list: there should"
            " be a single (array-like) output instead"
        )

    # random_projection should not have elements too small,
    # otherwise too much precision is lost in numerical gradient
    def random_projection():
        plain = rng.rand(*o_fn_out.shape) + 0.5
        if cast_to_output_type and o_output.dtype == "float32":
            return np.array(plain, o_output.dtype)
        return plain

    t_r = shared(random_projection(), borrow=True)
    t_r.name = "random_projection"

    # random projection of o onto t_r
    # This sum() is defined above, it's not the builtin sum.
    cost = theano.tensor.sum(t_r * o_output)

    if no_debug_ref:
        mode_for_cost = mode_not_slow(mode)
    else:
        mode_for_cost = mode

    cost_fn = function(
        tensor_pt, cost, name="gradient.py cost", mode=mode_for_cost
    )

    symbolic_grad = grad(cost, tensor_pt, disconnected_inputs="ignore")

    grad_fn = function(
        tensor_pt, symbolic_grad, name="gradient.py symbolic grad"
    )

    for test_num in xrange(n_tests):

        num_grad = numeric_grad(cost_fn, [p.copy() for p in pt], eps, out_type)

        analytic_grad = grad_fn(*[p.copy() for p in pt])

        # Since `tensor_pt` is a list, `analytic_grad` should be one too.
        assert isinstance(analytic_grad, list)

        try:

            max_arg, max_err_pos, max_abs_err, max_rel_err = num_grad.max_err(
                analytic_grad, abs_tol, rel_tol
            )

        except ValueError:

            # usually b/c NaNs

            raise_error(
                "Numerical: {}\nAnalytic: {}".format(
                    num_grad.gf, analytic_grad
                )
            )

        if max_abs_err > abs_tol and max_rel_err > rel_tol:

            raise_error(
                "Numerical: {}\nAnalytic: {}".format(
                    num_grad.gf, analytic_grad
                )
            )

        # get new random projection for next test
        if test_num < n_tests - 1:
            t_r.set_value(random_projection(), borrow=True)
