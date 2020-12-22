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
    path=abspath("data/isotropic"),
    generate=dict(latitude=dict(sigma=np.inf)),
    plot_data=False,
    plot_inclination_pdf=False,
)

# Copy output to this directory
copy("isotropic", "corner_transformed.pdf", "calibration_isotropic_corner.pdf")
copy("isotropic", "latitude.pdf", "calibration_isotropic_latitude.pdf")
