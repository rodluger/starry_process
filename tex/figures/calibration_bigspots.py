from starry_process import calibrate
import numpy as np
import os
import shutil

# Utility funcs to move figures to this directory
abspath = lambda *args: os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *args
)
copy = lambda name, src, dest: shutil.copyfile(
    abspath("data", name, src), abspath(dest)
)

# Run
calibrate.run(
    path=abspath("data/bigspots"),
    generate=dict(radius=dict(mu=45, sigma=5), nspots=dict(mu=1)),
    sample=dict(rmax=60),
    plot_data=False,
    plot_inclination_pdf=False,
)

# Copy output to this directory
copy("bigspots", "corner_transformed.pdf", "calibration_bigspots_corner.pdf")
copy("bigspots", "latitude.pdf", "calibration_bigspots_latitude.pdf")
