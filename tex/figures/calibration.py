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

# TODO: Not yet ready for CI runs
if not int(os.getenv("CI", 0)):

    # Equatorial spots
    calibrate.run(
        path="data/equatorial", generate=dict(latitude=dict(mu=0)),
    )

    # High latitude spots
    calibrate.run(
        path="data/hilat", generate=dict(latitude=dict(mu=60)),
    )

    # Isotropic spots
    calibrate.run(
        path="data/isotropic", generate=dict(latitude=dict(sigma=np.inf)),
    )

    # Tiny spots
    calibrate.run(
        path="data/tinyspots",
        generate=dict(radius=dict(mu=3.0, sigma=0.25), contrast=dict(mu=1)),
    )

    # Only 5 +/- 1 dark spots
    calibrate.run(
        path="data/fivespots",
        generate=dict(nspots=dict(mu=5, sigma=1), contrast=dict(mu=0.2)),
    )

    # 1 light curve
    calibrate.run(
        path="data/default_1",
        generate=dict(nlc=1),
        sample=dict(compute_inclination_pdf=False),
    )

    # 1000 light curves
    calibrate.run(
        path="data/default_1000",
        generate=dict(nlc=1000),
        sample=dict(compute_inclination_pdf=False),
    )

    # Limb-darkened
    calibrate.run(
        path="data/ld",
        generate=dict(u=[0.5, 0.25]),
        sample=dict(u=[0.5, 0.25]),
    )

    # Limb-darkened, 500 light curves
    calibrate.run(
        path="data/ld_500",
        generate=dict(nlc=500, u=[0.5, 0.25]),
        sample=dict(u=[0.5, 0.25], compute_inclination_pdf=False),
        plot_data=False,
    )

    # Limb-darkened, 1000 light curves
    calibrate.run(
        path="data/ld_1000",
        generate=dict(nlc=1000, u=[0.5, 0.25]),
        sample=dict(u=[0.5, 0.25], compute_inclination_pdf=False),
        plot_data=False,
    )

    # Limb-darkened, 500 light curves, but assuming no limb
    # darkening when running inference
    calibrate.run(
        path="data/ld_500_no_ld",
        generate=dict(nlc=500, u=[0.5, 0.25]),
        sample=dict(u=[0.0, 0.0], compute_inclination_pdf=False),
        plot_data=False,
    )

