import numpy as np
import os

__all__ = ["get_intensity_design_matrix", "get_flux_design_matrix"]


def get_intensity_design_matrix(ydeg, npix=100, clobber=False):
    # Cached file name
    file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "A_I{:02d}-{:03d}.npz".format(ydeg, npix),
    )

    # If it doesn't exist, create it
    if clobber or not os.path.exists(file):
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
        A_I = np.pi * map.intensity_design_matrix(lat=lat, lon=lon)
        A_I = np.array(A_I, dtype="float32")

        try:
            np.savez_compressed(file, A_I=A_I)
        except:
            # fail silently; we might not have write permissions
            pass

    # Else load from disk
    else:

        try:

            A_I = np.load(file)["A_I"]

        except:

            return get_intensity_design_matrix(ydeg, npix=npix, clobber=True)

    return A_I


def get_flux_design_matrix(ydeg, npts=300):
    # Cached file name
    file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "A_F{:02d}-{:03d}.npz".format(ydeg, npts),
    )

    # If it doesn't exist, create it
    if not os.path.exists(file):
        import starry

        map = starry.Map(ydeg, lazy=False)
        incs = [15, 30, 45, 60, 75, 90]
        A_F = np.empty((len(incs), npts, (ydeg + 1) ** 2))
        for i, inc in enumerate(incs):
            map.inc = inc
            theta = np.linspace(0, 360, npts) * 2
            A_F[i] = map.design_matrix(theta=theta)

        try:
            np.savez(file, A_F=A_F)
        except:
            # fail silently; we might not have write permissions
            pass

    # Else load from disk
    else:

        try:

            A_F = np.load(file)["A_F"]

        except IOError:

            return get_flux_design_matrix(ydeg, npts=npts, clobber=True)

    return A_F
