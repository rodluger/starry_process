from starry_process import calibrate
import numpy as np
import os


# TODO: Not yet ready for CI runs
if not bool(os.getenv("CI", False)):

    # Default run
    calibrate.run_batch(
        path="data/batch_default", compute_inclination_pdf=False
    )

