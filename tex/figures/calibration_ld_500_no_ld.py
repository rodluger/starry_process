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
    path=abspath("data/ld_500_no_ld"),
    generate=dict(nlc=500, u=[0.5, 0.25]),
    sample=dict(u=[0.0, 0.0], compute_inclination_pdf=False),
    plot_data=False,
    plot_inclination_pdf=False,
)

# Copy output to this directory
copy(
    "ld_500_no_ld",
    "corner_transformed.pdf",
    "calibration_ld_500_no_ld_corner.pdf",
)
copy("ld_500_no_ld", "latitude.pdf", "calibration_ld_500_no_ld_latitude.pdf")
