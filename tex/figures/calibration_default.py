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
calibrate.run(path=abspath("data/default"), ncols=7, clip=True)

# Copy output to this directory
copy("default", "data.pdf", "calibration_default_data.pdf")
copy("default", "corner_transformed.pdf", "calibration_default_corner.pdf")
copy("default", "latitude.pdf", "calibration_default_latitude.pdf")
copy("default", "inclination.pdf", "calibration_default_inclination.pdf")
