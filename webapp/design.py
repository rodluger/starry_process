import numpy as np
import os

__all__ = ["get_design_matrix"]


def get_design_matrix(ydeg, npix=50):
    # Cached file name
    file = "A{:02d}-{:03d}.npz".format(ydeg, npix)

    # If it doesn't exist, create it
    if not os.path.exists(file):
        import starry

        Nx = 2 * npix
        x, y = np.meshgrid(
            2 * np.sqrt(2) * np.linspace(-1, 1, Nx),
            np.sqrt(2) * np.linspace(-1, 1, npix),
        )
        a = np.sqrt(2)
        b = 2 * np.sqrt(2)
        idx = (y / a) ** 2 + (x / b) ** 2 > 1
        y[idx] = np.nan
        x[idx] = np.nan
        theta = np.arcsin(y / np.sqrt(2))
        lat = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)
        lon = np.pi * x / (2 * np.sqrt(2) * np.cos(theta))
        lat = lat.flatten() * 180 / np.pi
        lon = lon.flatten() * 180 / np.pi
        map = starry.Map(ydeg, lazy=False)
        A = np.pi * map.intensity_design_matrix(lat=lat, lon=lon)
        np.savez(file, A=A)

    # Else load from disk
    else:

        A = np.load(file)["A"]

    return A
