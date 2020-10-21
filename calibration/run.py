from starry_process import calibrate
import pickle
import numpy as np

# Inputs
kwargs = dict()

# Generate
data = calibrate.generate(**kwargs)
np.savez("data.npz", data=data)
fig = calibrate.plot_data(data, **kwargs)
fig.savefig("data.pdf", bbox_inches="tight")

# Sample
results = calibrate.sample(data, **kwargs)
pickle.dump(results, open("results.pkl", "wb"))

# Plot the results
fig = calibrate.plot_latitude_pdf(results, **kwargs)
fig.savefig("latitude.pdf", bbox_inches="tight")

fig = calibrate.plot_trace(results.samples, **kwargs)
fig.savefig("trace.pdf", bbox_inches="tight")

fig = calibrate.plot_corner(results.samples, **kwargs)
fig.savefig("corner.pdf", bbox_inches="tight")
