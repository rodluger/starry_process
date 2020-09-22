from starry_process import StarryProcess
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import theano
import theano.tensor as tt
from generate import generate
import time
import datetime
from scipy.optimize import minimize


# DEBUG
plt.switch_backend("MacOSX")


def _log_probability_function(
    t, flux, ferr, params, baseline_mean, baseline_var
):
    sp = StarryProcess(**params, angle_unit="rad")
    log_prior = tt.sum(tt.log(tt.sin(params["incs"])))
    log_prior += sp.log_jac()
    log_likes = [None for k in range(len(flux))]
    for k in range(len(flux)):
        sp.design._set_params(
            period=params["periods"][k], inc=params["incs"][k]
        )
        log_likes[k] = sp.log_likelihood(
            t,
            flux[k],
            ferr ** 2,
            baseline_mean=baseline_mean,
            baseline_var=baseline_var,
        )
    log_like = tt.sum(log_likes)
    return log_prior + log_like


def get_log_probability(
    t,
    flux,
    ferr,
    params,
    gradient=True,
    baseline_mean=0.0,
    baseline_var=0.0,
    **kwargs
):
    free_params = []
    for key, value in params.items():
        if value == "free":
            x = tt.dscalar()
            params[key] = x
            free_params.append(x)

    print("Compiling the likelihood function...")
    tstart = time.time()

    fn = _log_probability_function(
        t, flux, ferr, params, baseline_mean, baseline_var
    )
    if gradient:
        fn = [fn] + theano.grad(fn, free_params, disconnected_inputs="ignore")
    log_probability = theano.function(free_params, fn)

    telapsed = int(time.time() - tstart)
    print("Done ({:s}).".format(str(datetime.timedelta(seconds=telapsed))))

    return log_probability


# Generate the data
data, truth, fig = generate(
    cmu=1.0 / 30,
    nspots=30,
    nlc=30,
    periods=1.0,
    plot=True,
    ferr=1e-4,
    use_starry_process=False,
    normalize=True,
)
t = data["t"]
ferr = data["ferr"]
flux = data["flux"]
plt.show()


# Set up the inference
params = {
    "sa": truth["sa"],
    "sb": truth["sb"],
    "la": "free",  # truth["la"],
    "lb": "free",  # truth["lb"],
    "ca": truth["ca"],
    "cb": truth["cb"],
    "incs": truth["incs"],
    "periods": truth["periods"],
}
log_prob = get_log_probability(
    t, flux, ferr, params, gradient=False, baseline_var=1e-2
)

# Grid search
npts = 30
a = np.linspace(0, 1, npts)
b = np.linspace(0, 1, npts)
logp = np.zeros((npts, npts))
for i in tqdm(range(npts)):
    for j in range(npts):
        logp[j, i] = log_prob(a[i], b[j])
prob = np.exp(logp - np.max(logp))
plt.imshow(prob, origin="lower", extent=(0, 1, 0, 1), cmap="plasma")
plt.axvline(truth["la"], color="w")
plt.axhline(truth["lb"], color="w")
plt.show()
breakpoint()
