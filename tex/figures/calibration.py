from starry_process import calibrate
import os

# TODO: Not yet ready for Azure runs
if not bool(os.getenv("MAKEFILE", False)):

    # calibrate.run(path="data/default")

    calibrate.run(
        path="data/ld_unnorm_1000",
        plot_data=False,
        compute_inclination_pdf=False,
        generate=dict(u=[0.5, 0.25], normalized=False),
        sample=dict(compute_inclination_pdf=False, u=[0.5, 0.25]),
    )

