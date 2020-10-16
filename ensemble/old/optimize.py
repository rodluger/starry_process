from .generate import generate
import starry
from starry_process import StarryProcess
from starry_process import optimize
import numpy as np
import theano
import theano.tensor as tt
import os
from tqdm import tqdm

PATH = os.path.abspath(os.path.dirname(__file__))


def optimize(runid):

    # File names
    DATA_FILE = os.path.join(PATH, "{:02d}".format(runid), "data.npz")
    TRUTH_FILE = os.path.join(PATH, "{:02d}".format(runid), "truth.npz")
    INPUT_FILE = os.path.join(PATH, "{:02d}".format(runid), "input.json")

    # Generate (and save) the data
    if clobber or not os.path.exists(DATA_FILE):
        data, truth = generate(runid)
        np.savez(DATA_FILE, **data)
        np.savez(TRUTH_FILE, **truth)
    else:
        data = np.load(DATA_FILE)
        truth = np.load(TRUTH_FILE)
    t = data["t"]
    flux = data["flux"]
    ferr = data["ferr"]
    nlc = len(flux)

    # Settings
    with open(INPUT_FILE, "r") as f:
        inputs = json.load(f).get("optimize", {})
    baseline_var = inputs.get("baseline_var", 1e-2)
    niter = inputs.get("niter", 5000)
    kwargs = dict(lr=inputs.get("lr", 1e-2))
    optimizer = getattr(optimize, inputs.get("optimizer", "NAdam"))
    s0 = inputs.get("s0", 25.0)
    la0 = inputs.get("la0", 0.5)
    lb0 = inputs.get("lb0", 0.5)
    c0 = inputs.get("c0", 0.5)
    N0 = inputs.get("N0", 10.0)
    incs0 = inputs.get("incs0", 60.0)

    # Vars
    s = theano.shared(truth["s"])
    la = theano.shared(truth["la"])
    lb = theano.shared(truth["lb"])
    c = theano.shared(truth["c"])
    N = theano.shared(truth["N"])
    incs = theano.shared(truth["incs"])
    theano_vars = [s, la, lb, c, N, incs]

    # Loss function
    sp = StarryProcess(size=s, latitude=[la, lb], contrast=[c, N])
    log_like = []
    for k in range(nlc):
        log_like.append(
            sp.log_likelihood(
                t,
                flux[k],
                ferr ** 2,
                baseline_var=baseline_var,
                period=truth["periods"][k],
                inc=incs[k],
            )
        )
    loss = -(
        tt.sum(log_like)
        + tt.sum(tt.log(tt.sin(incs * np.pi / 180)))
        + sp.log_jac()
    )

    # Set up the optimizer
    upd = optimizer(loss, theano_vars, **kwargs)
    train = theano.function([], theano_vars + [loss], updates=upd)

    # Call it once to get the "true" loss
    *_, true_loss = train()

    # Initial (bad) guesses
    s.set_value(s0)
    la.set_value(la0)
    lb.set_value(lb0)
    c.set_value(c0)
    N.set_value(N0)
    incs.set_value(incs0 * np.ones(nlc))

    # Run
    loss_val = np.zeros(niter)
    best_loss = np.inf
    best_params = []
    iterator = tqdm(np.arange(niter))
    for n in iterator:
        *params, loss_val[n] = train()
        iterator.set_postfix(
            {"loss": "{:.1f} / {:.1f}".format(best_loss, true_loss)}
        )
        if loss_val[n] < best_loss:
            best_loss = loss_val[n]
            best_params = params

    # Return
    optim = dict(
        loss=loss_val,
        s=best_params[0],
        la=best_params[1],
        lb=best_params[2],
        c=best_params[3],
        N=best_params[4],
        incs=best_params[5:],
    )
    return optim
