from starry_process import calibrate
import numpy as np
import os
import shutil

# TODO: Not yet ready for CI runs
if not int(os.getenv("CI", 0)):

    # Utility funcs to move figures to this directory
    abspath = lambda *args: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), *args
    )
    copy = lambda name, src, dest: shutil.copyfile(
        abspath("data", name, src), abspath(dest)
    )

    # Run
    calibrate.run(
        path=abspath("data/onespot"),
        generate=dict(contrast=dict(mu=0.9), nspots=dict(mu=1)),
        plot_data=True,  # TODO
        plot_inclination_pdf=False,
    )

    # Copy output to this directory
    copy("onespot", "corner_transformed.pdf", "calibration_onespot_corner.pdf")
    copy("onespot", "latitude.pdf", "calibration_onespot_latitude.pdf")
