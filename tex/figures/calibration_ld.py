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
    path=abspath("data/ld"),
    generate=dict(u=[0.5, 0.25]),
    sample=dict(u=[0.5, 0.25]),
    plot_data=False,
    plot_inclination_pdf=False,
)

# Copy output to this directory
copy(
    "ld", "corner_transformed.pdf", "calibration_ld_corner.pdf",
)
copy("ld", "latitude.pdf", "calibration_ld_latitude.pdf")
