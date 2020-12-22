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
    path=abspath("data/ld_1000"),
    generate=dict(nlc=1000, u=[0.5, 0.25]),
    sample=dict(u=[0.5, 0.25], compute_inclination_pdf=False),
    plot_data=False,
    plot_inclination_pdf=False,
)

# Copy output to this directory
copy(
    "ld_1000", "corner_transformed.pdf", "calibration_ld_1000_corner.pdf",
)
copy("ld_1000", "latitude.pdf", "calibration_ld_1000_latitude.pdf")
