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
    path=abspath("data/hicontrast"),
    generate=dict(nspots=dict(mu=2), contrast=dict(mu=0.5)),
    sample=dict(compute_inclination_pdf=False),
)

# Copy output to this directory
copy(
    "hicontrast",
    "corner_transformed.pdf",
    "calibration_hicontrast_corner.pdf",
)
