from starry_process import MCMCInterface
import pymc3 as pm
import numpy as np

# Generate a dataset
npts = 100
yerr = 1e-2
x = np.linspace(0, 2 * np.pi, npts)
u_true = 0.5
v_true = 0.3
y = np.cos(u_true * x + v_true) + yerr * np.random.randn(npts)

# Define the model
with pm.Model() as model:
    u = pm.Uniform("u", 0, 1)
    v = pm.Normal("v", 0, 1)
    f = pm.math.cos(u * x + v)
    pm.Normal("obs", mu=f, sd=yerr, observed=y)

# Instantiate our interface
mci = MCMCInterface(model=model)

# Generate initial samples (centered at the test point)
nwalkers = 100
p0 = mci.get_initial_state(nwalkers=nwalkers, var=1.0)

# Check that we can compute logp everywhere
for k in range(nwalkers):
    assert np.isfinite(mci.logp(p0[k]))

# Now optimize via gradient descent
p = mci.optimize()

# Convert to user parametrization and check we found the optimum
u, v = mci.transform(p, varnames=["u", "v"])
assert np.allclose(u, u_true, atol=1e-2)
assert np.allclose(v, v_true, atol=1e-2)

# Generate initial samples (this time centered at the MAP)
p0 = mci.get_initial_state(nwalkers=nwalkers)

# Check that we can compute logp everywhere
for k in range(nwalkers):
    assert np.isfinite(mci.logp(p0[k]))
