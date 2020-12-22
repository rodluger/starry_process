import numpy as np
import os
import shutil
import corner

old_corner = corner.corner


def new_corner(*args, **kwargs):
    fig = old_corner(*args, **kwargs)
    ax = np.reshape(fig.axes, (5, 5))
    for axis in ax[:, 0]:
        axis.axvspan(0, 10, hatch="//", facecolor="none", lw=0.5, alpha=0.25)
    return fig


corner.corner = new_corner

from starry_process import calibrate

# Utility funcs to move figures to this directory
abspath = lambda *args: os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *args
)
copy = lambda name, src, dest: shutil.copyfile(
    abspath("data", name, src), abspath(dest)
)

# Run
calibrate.run(
    path=abspath("data/tinyspots"),
    generate=dict(radius=dict(mu=3.0, sigma=0.25), contrast=dict(mu=1)),
    plot_data=False,
    plot_inclination_pdf=False,
)

# Copy output to this directory
copy("tinyspots", "corner_transformed.pdf", "calibration_tinyspots_corner.pdf")
copy("tinyspots", "latitude.pdf", "calibration_tinyspots_latitude.pdf")
