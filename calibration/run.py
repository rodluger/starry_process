from starry_process import calibrate
import pickle
import numpy as np
import os


# Inputs
clobber = True
kwargs = dict()

# Generate
if clobber or not os.path.exists("data.npz"):
    data = calibrate.generate(**kwargs)
    np.savez("data.npz", **data)
else:
    data = np.load("data.npz")

# Plot the data
fig = calibrate.plot_data(data, **kwargs)
fig.savefig("data.pdf", bbox_inches="tight")

# Sample
if clobber or not os.path.exists("results.pkl"):
    results = calibrate.sample(data, **kwargs)
    pickle.dump(results, open("results.pkl", "wb"))
else:
    results = pickle.load(open("results.pkl", "rb"))

# Plot the results
fig = calibrate.plot_latitude_pdf(results, **kwargs)
fig.savefig("latitude.pdf", bbox_inches="tight")

fig = calibrate.plot_trace(results, **kwargs)
fig.savefig("trace.pdf", bbox_inches="tight")

fig = calibrate.plot_corner(results, **kwargs)
fig.savefig("corner.pdf", bbox_inches="tight")
