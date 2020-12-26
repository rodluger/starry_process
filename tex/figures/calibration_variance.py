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
    path=abspath("data/variance"),
    generate=dict(
        nlc=1000,
        nspots=dict(sigma=3),
        radius=dict(sigma=3),
        contrast=dict(sigma=0.01),
    ),
    sample=dict(compute_inclination_pdf=False),
    plot_data=False,
    plot_inclination_pdf=False,
)

# Copy output to this directory
copy("variance", "corner_transformed.pdf", "calibration_variance_corner.pdf")
copy("variance", "latitude.pdf", "calibration_variance_latitude.pdf")
