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
        path=abspath("data/variance"),
        generate=dict(
            nspots=dict(mu=5), radius=dict(mu=3), contrast=dict(mu=0.01)
        ),
        plot_data=False,
        plot_inclination_pdf=False,
    )

    # Copy output to this directory
    copy(
        "variance", "corner_transformed.pdf", "calibration_variance_corner.pdf"
    )
    copy("variance", "latitude.pdf", "calibration_variance_latitude.pdf")
