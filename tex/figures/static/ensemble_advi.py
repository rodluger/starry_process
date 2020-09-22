from generate import generate
from starry_process import StarryProcess
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pymc3 as pm
import exoplanet as xo
import theano
import theano.tensor as tt
import os

plt.switch_backend("MacOSX")

# Options
nlc = 10
nadvi = 10000
nsamples = 10000
seed = 0
baseline_var = 1e-2
compute = True

# Get the data
data, truth, _ = generate(nlc=nlc, plot=True)
plt.show()
t = data["t"]
flux = data["flux"]
ferr = data["ferr"]

# Set up the model
with pm.Model() as model:

    # Priors
    sa = pm.Uniform("sa", 0, 1, testval=truth["sa"])
    sb = pm.Uniform("sb", 0, 1, testval=truth["sb"])
    la = pm.Uniform("la", 0, 1, testval=truth["la"])
    lb = pm.Uniform("lb", 0, 1, testval=truth["lb"])
    ca = pm.Uniform("ca", 0, 5, testval=truth["ca"])
    cb = truth["cb"]
    incs = pm.Uniform(
        "incs", 0, 0.5 * np.pi, shape=(nlc,), testval=truth["incs"]
    )
    periods = truth["periods"]

    # Set up the GP
    sp = StarryProcess(
        sa=sa, sb=sb, la=la, lb=lb, ca=ca, cb=cb, angle_unit="rad"
    )

    # Likelihood for each light curve
    log_like = []
    for k in range(nlc):
        sp.design._set_params(period=periods[k], inc=incs[k])
        log_like.append(
            sp.log_likelihood(t, flux[k], ferr ** 2, baseline_var=baseline_var)
        )
    pm.Potential("marginal", tt.sum(log_like))

    # Priors
    pm.Potential("sini", tt.sum(tt.log(tt.sin(incs))))
    pm.Potential("jacobian", sp.log_jac())

    # Fit
    print("Fitting...")
    advi_fit = pm.fit(
        n=nadvi, method=pm.ADVI(), random_seed=seed, start=model.test_point
    )

    # Sample
    print("Sampling...")
    advi_trace = advi_fit.sample(nsamples)

    # Display the summary
    print(pm.summary(advi_trace))

    # Save the trace
    pm.save_trace(trace=advi_trace, overwrite=True)
