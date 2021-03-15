from .ops import pTA1Op
from .compat import tt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

try:
    from IPython.display import HTML
except:
    pass


def RAxisAngle(axis, theta):
    axis = np.array(axis)
    axis /= np.sqrt(np.sum(axis ** 2))
    cost = np.cos(theta)
    sint = np.sin(theta)
    return np.array(
        [
            [
                cost + axis[0] * axis[0] * (1 - cost),
                axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
                axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
            ],
            [
                axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
                cost + axis[1] * axis[1] * (1 - cost),
                axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
            ],
            [
                axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
                axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
                cost + axis[2] * axis[2] * (1 - cost),
            ],
        ]
    )


def latlon_to_xyz(lat, lon):
    """Convert lat-lon points in radians to Cartesian points."""
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)
    R1 = RAxisAngle([1.0, 0.0, 0.0], -lat)
    R2 = RAxisAngle([0.0, 1.0, 0.0], lon)
    return np.einsum("ij...,jl...,l->i...", R2, R1, np.array([0.0, 0.0, 1.0]))


def compute_moll_grid(my, mx):
    """Compute the polynomial basis on a Mollweide grid."""
    x, y = np.meshgrid(
        np.sqrt(2) * np.linspace(-2, 2, mx),
        np.sqrt(2) * np.linspace(-1, 1, my),
    )

    # Make points off-grid nan
    a = np.sqrt(2)
    b = 2 * np.sqrt(2)
    y[(y / a) ** 2 + (x / b) ** 2 > 1] = np.nan

    # https://en.wikipedia.org/wiki/Mollweide_projection
    theta = np.arcsin(y / np.sqrt(2))
    lat = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)
    lon0 = 3 * np.pi / 2
    lon = lon0 + np.pi * x / (2 * np.sqrt(2) * np.cos(theta))

    # Back to Cartesian, this time on the *sky*
    x = np.reshape(np.cos(lat) * np.cos(lon), [1, -1])
    y = np.reshape(np.cos(lat) * np.sin(lon), [1, -1])
    z = np.reshape(np.sin(lat), [1, -1])
    R = RAxisAngle([1.0, 0.0, 0.0], -np.pi / 2)
    return R @ np.concatenate((x, y, z))


def mollweide_transform(my=150, mx=300):
    x, y, z = compute_moll_grid(my=my, mx=mx)
    M = np.pi * pTA1Op()(x, y, z).eval()
    return M


def latlon_transform(lat, lon):
    x, y, z = latlon_to_xyz(lat, lon)
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    M = np.pi * pTA1Op()(x, y, z).eval()
    return M


def get_moll_latitude_lines(dlat=np.pi / 6, npts=1000, niter=100):
    res = []
    latlines = np.arange(-np.pi / 2, np.pi / 2, dlat)[1:]
    for lat in latlines:
        theta = lat
        for n in range(niter):
            theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (
                2 + 2 * np.cos(2 * theta)
            )
        x = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), npts)
        y = np.ones(npts) * np.sqrt(2) * np.sin(theta)
        a = np.sqrt(2)
        b = 2 * np.sqrt(2)
        y[(y / a) ** 2 + (x / b) ** 2 > 1] = np.nan
        res.append((x, y))
    return res


def get_moll_longitude_lines(dlon=np.pi / 6, npts=1000, niter=100):
    res = []
    lonlines = np.arange(-np.pi, np.pi, dlon)[1:]
    for lon in lonlines:
        lat = np.linspace(-np.pi / 2, np.pi / 2, npts)
        theta = np.array(lat)
        for n in range(niter):
            theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (
                2 + 2 * np.cos(2 * theta)
            )
        x = 2 * np.sqrt(2) / np.pi * lon * np.cos(theta)
        y = np.sqrt(2) * np.sin(theta)
        res.append((x, y))
    return res


def visualize(image, **kwargs):
    # Get kwargs
    cmap = kwargs.pop("cmap", "plasma")
    grid = kwargs.pop("grid", True)
    interval = kwargs.pop("interval", 75)
    file = kwargs.pop("file", None)
    html5_video = kwargs.pop("html5_video", True)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    dpi = kwargs.pop("dpi", None)
    figsize = kwargs.pop("figsize", None)
    bitrate = kwargs.pop("bitrate", None)
    colorbar = kwargs.pop("colorbar", False)
    shrink = kwargs.pop("shrink", 0.01)
    ax = kwargs.pop("ax", None)
    if ax is None:
        custom_ax = False
    else:
        custom_ax = True

    # Animation
    nframes = image.shape[0]
    animated = nframes > 1
    borders = []
    latlines = []
    lonlines = []

    # Set up the plot
    if figsize is None:
        figsize = (7, 3.75)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.figure

    # Mollweide
    dx = 2.0 / image.shape[1]
    extent = (1 + shrink) * np.array(
        [
            -(1 + dx) * 2 * np.sqrt(2),
            2 * np.sqrt(2),
            -(1 + dx) * np.sqrt(2),
            np.sqrt(2),
        ]
    )
    ax.axis("off")
    ax.set_xlim(-2 * np.sqrt(2) - 0.05, 2 * np.sqrt(2) + 0.05)
    ax.set_ylim(-np.sqrt(2) - 0.05, np.sqrt(2) + 0.05)

    # Anti-aliasing at the edges
    x = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), 10000)
    y = np.sqrt(2) * np.sqrt(1 - (x / (2 * np.sqrt(2))) ** 2)
    borders += [ax.fill_between(x, 1.1 * y, y, color="w", zorder=-1)]
    borders += [
        ax.fill_betweenx(0.5 * x, 2.2 * y, 2 * y, color="w", zorder=-1)
    ]
    borders += [ax.fill_between(x, -1.1 * y, -y, color="w", zorder=-1)]
    borders += [
        ax.fill_betweenx(0.5 * x, -2.2 * y, -2 * y, color="w", zorder=-1)
    ]

    if grid:
        x = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), 10000)
        a = np.sqrt(2)
        b = 2 * np.sqrt(2)
        y = a * np.sqrt(1 - (x / b) ** 2)
        borders += ax.plot(x, y, "k-", alpha=1, lw=1.5, zorder=0)
        borders += ax.plot(x, -y, "k-", alpha=1, lw=1.5, zorder=0)
        lats = get_moll_latitude_lines()
        latlines = [None for n in lats]
        for n, l in enumerate(lats):
            (latlines[n],) = ax.plot(
                l[0], l[1], "k-", lw=0.5, alpha=0.5, zorder=0
            )
        lons = get_moll_longitude_lines()
        lonlines = [None for n in lons]
        for n, l in enumerate(lons):
            (lonlines[n],) = ax.plot(
                l[0], l[1], "k-", lw=0.5, alpha=0.5, zorder=0
            )

    # Plot the first frame of the image
    if vmin is None:
        vmin = np.nanmin(image)
    if vmax is None:
        vmax = np.nanmax(image)
    # Set a minimum contrast
    if np.abs(vmin - vmax) < 1e-12:
        vmin -= 1e-12
        vmax += 1e-12

    img = ax.imshow(
        image[0],
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        animated=animated,
        zorder=-3,
    )

    # Add a colorbar
    if colorbar:
        if not custom_ax:
            fig.subplots_adjust(right=0.85)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax, orientation="vertical")

    # Display or save the image / animation
    if animated:

        def updatefig(i):
            img.set_array(image[i])
            return (img, *borders, *latlines, *lonlines)

        ani = FuncAnimation(
            fig, updatefig, interval=interval, blit=True, frames=image.shape[0]
        )

        # Business as usual
        if (file is not None) and (file != ""):
            if file.endswith(".mp4"):
                ani.save(file, writer="ffmpeg", dpi=dpi, bitrate=bitrate)
            elif file.endswith(".gif"):
                ani.save(file, writer="imagemagick", dpi=dpi, bitrate=bitrate)
            else:
                # Try and see what happens!
                ani.save(file, dpi=dpi, bitrate=bitrate)
            if not custom_ax:
                if not plt.isinteractive():
                    plt.close()
        else:  # if not custom_ax:
            try:
                if "zmqshell" in str(type(get_ipython())):
                    plt.close()
                    with matplotlib.rc_context(
                        {
                            "savefig.dpi": dpi
                            if dpi is not None
                            else "figure",
                            "animation.bitrate": bitrate
                            if bitrate is not None
                            else -1,
                        }
                    ):
                        if html5_video:
                            display(HTML(ani.to_html5_video()))
                        else:
                            display(HTML(ani.to_jshtml()))
                else:
                    raise NameError("")
            except NameError:
                plt.show()
                if not plt.isinteractive():
                    plt.close()

        # Matplotlib generates an annoying empty
        # file when producing an animation. Delete it.
        try:
            os.remove("None0000000.png")
        except FileNotFoundError:
            pass

    else:
        if (file is not None) and (file != ""):
            fig.savefig(file, bbox_inches="tight")
            if not custom_ax:
                if not plt.isinteractive():
                    plt.close()
        elif not custom_ax:
            plt.show()
