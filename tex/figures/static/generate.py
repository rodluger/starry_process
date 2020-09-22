from starry_process import StarryProcess
import starry
from scipy.stats import norm as Normal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm import tqdm
import os


FILE = os.path.abspath(__file__)


def lat2y(lat):
    # Returns the fractional y position (in [0, 1])
    # corresponding to a given latitude on a Mollweide grid
    theta = lat
    niter = 100
    for n in range(niter):
        theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (
            2 + 2 * np.cos(2 * theta)
        )
    return np.sin(theta)


class Star(object):
    def __init__(self, nlon=300, ydeg=30):
        # Generate a uniform intensity grid
        self.nlon = nlon
        self.nlat = nlon // 2
        self.lon = np.linspace(-180, 180, self.nlon)
        self.lat = np.linspace(-90, 90, self.nlat)
        self.lon, self.lat = np.meshgrid(self.lon, self.lat)
        self.intensity = np.ones_like(self.lat)

        # Instantiate a starry map
        self.map = starry.Map(ydeg, lazy=False)

    def _angular_distance(self, lam1, lam2, phi1, phi2):
        # https://en.wikipedia.org/wiki/Great-circle_distance
        return (
            np.arccos(
                np.sin(phi1 * np.pi / 180) * np.sin(phi2 * np.pi / 180)
                + np.cos(phi1 * np.pi / 180)
                * np.cos(phi2 * np.pi / 180)
                * np.cos((lam2 - lam1) * np.pi / 180)
            )
            * 180
            / np.pi
        )

    def reset(self):
        self.intensity = np.zeros_like(self.lat)

    def add_spot(self, lon, lat, radius, contrast, linear=True):
        idx = self._angular_distance(lon, self.lon, lat, self.lat) <= radius
        if linear:
            self.intensity[idx] -= contrast
        else:
            self.intensity[idx] = -contrast

    def flux(self, t, period=1.0, inc=60.0, smoothing=0.1, normalize=True):
        # Expand in Ylms
        self.map.load(self.intensity)
        self.map.inc = inc

        # Smooth to get rid of ringing
        if smoothing > 0:
            l = np.concatenate(
                [np.repeat(l, 2 * l + 1) for l in range(self.map.ydeg + 1)]
            )
            s = np.exp(-0.5 * l * (l + 1) * smoothing ** 2)
            self.map._y *= s

        # Get the flux
        flux = 1 + self.map.flux(theta=360.0 / period * t)

        # Median-normalize it
        if normalize:
            flux /= np.median(flux)

        # Remove the baseline
        flux -= 1

        return flux


def generate(
    use_starry_process=False,
    normalize=True,
    nlon=300,
    ydeg=30,
    smoothing=0.1,
    seed=0,
    nlc=10,
    npts=1000,
    tmax=4.0,
    periods=1.0,
    ferr=1e-4,
    nspots=20,
    smu=20.0,
    ssig=5.0,
    lmu=30.0,
    lsig=5.0,
    cmu=0.05,
    csig=0.0,
    clobber=False,
    plot=False,
    nrows=1,
    vmin=0.5,
    vmax=1.0,
):
    # Get the hash
    periods = periods * np.ones(nlc)
    args = (
        use_starry_process,
        normalize,
        nlon,
        ydeg,
        smoothing,
        seed,
        nlc,
        npts,
        tmax,
        *periods,
        ferr,
        nspots,
        smu,
        ssig,
        lmu,
        lsig,
        cmu,
        csig,
    )
    ID = hex(abs(hash(args)))
    BASE = os.path.join(os.path.dirname(os.path.abspath(FILE)), ID)

    if clobber or not os.path.exists(BASE + "_data.npz"):

        # Convert to standard params
        sp = StarryProcess()
        sa, sb = sp.size.transform.transform(smu, ssig)
        la, lb = sp.latitude.transform.transform(lmu, lsig)
        ca, cb = cmu * nspots, csig * nspots
        sp = StarryProcess(sa=sa, sb=sb, la=la, lb=lb, ca=ca, cb=cb)

        # Generate `nlc` light curves
        np.random.seed(seed)
        t = np.linspace(0, tmax, npts)
        flux0 = np.empty((nlc, npts))
        flux = np.empty((nlc, npts))
        images = [None for k in range(nlc)]
        incs = np.arccos(np.random.uniform(0, 1, size=nlc))

        if use_starry_process:

            # Draw Ylms
            y = sp.sample_ylm(nsamples=nlc).eval()

            # Compute the fluxes
            map = starry.Map(15, lazy=False)
            for k in tqdm(range(nlc)):

                # Get the light curve
                map[:, :] = y[k]
                map.inc = incs[k] * 180 / np.pi
                flux0[k] = 1 + map.flux(theta=360.0 / periods[k] * t)

                # Median-normalize it
                if normalize:
                    flux0[k] /= np.median(flux0[k])

                # Remove the baseline
                flux0[k] -= 1

                # Render the surface
                images[k] = 1.0 + map.render(projection="moll", res=150)

        else:

            star = Star(nlon=nlon, ydeg=ydeg)
            for k in tqdm(range(nlc)):

                # Generate the stellar map
                star.reset()
                for n in range(nspots):
                    radius = Normal.rvs(smu, ssig)
                    sign = 1 if np.random.random() < 0.5 else -1
                    lat = sign * Normal.rvs(lmu, lsig)
                    lon = np.random.uniform(-180, 180)
                    contrast = Normal.rvs(cmu, csig)
                    star.add_spot(lon, lat, radius, contrast)

                # Get the light curve
                flux0[k] = star.flux(
                    t,
                    period=periods[k],
                    inc=incs[k] * 180 / np.pi,
                    smoothing=smoothing,
                    normalize=normalize,
                )

                # Render the surface
                images[k] = 1.0 + star.map.render(projection="moll", res=150)

        # Add photon noise
        for k in tqdm(range(nlc)):
            flux[k] = flux0[k] + ferr * np.random.randn(npts)

        # Save the "true" params
        truth = dict(
            sa=sa,
            sb=sb,
            la=la,
            lb=lb,
            ca=ca,
            cb=cb,
            periods=periods,
            incs=incs,
        )
        np.savez_compressed(BASE + "_truth.npz", **truth)

        # Save the data
        data = dict(
            t=t,
            flux0=flux0,
            flux=flux,
            ferr=ferr,
            nspots=nspots,
            smu=smu,
            ssig=ssig,
            lmu=lmu,
            lsig=lsig,
            cmu=cmu,
            csig=csig,
            images=images,
        )
        np.savez_compressed(BASE + "_data.npz", **data)

    else:

        data = np.load(BASE + "_data.npz")
        data = {key: data[key] for key in data.keys()}
        t = data["t"]
        flux0 = data["flux0"]
        flux = data["flux"]
        images = data["images"]

        truth = np.load(BASE + "_truth.npz")
        truth = {key: truth[key] for key in truth.keys()}
        incs = truth["incs"]

    # Plot the data
    if plot:

        assert nlc % nrows == 0

        wr = np.ones(nlc // nrows)
        wr[-1] = 1.17
        gridspec = {"width_ratios": wr}
        fig, ax = plt.subplots(
            2 * nrows,
            nlc // nrows,
            figsize=(12, 2 * nrows),
            gridspec_kw=gridspec,
        )
        fig.subplots_adjust(hspace=0.4)
        axtop = ax.transpose().flatten()[::2]
        axbot = ax.transpose().flatten()[1::2]
        yrng = 1.1 * np.max(np.abs(1e3 * (flux0)))
        ymin = -yrng
        ymax = yrng
        xe = 2 * np.linspace(-1, 1, 1000)
        ye = np.sqrt(1 - (0.5 * xe) ** 2)
        eps = 0.01
        xe = (1 - eps) * xe
        ye = (1 - 0.5 * eps) * ye
        for k in range(nlc):
            im = axtop[k].imshow(
                images[k],
                origin="lower",
                extent=(-2, 2, -1, 1),
                cmap="plasma",
                vmin=vmin,
                vmax=vmax,
            )
            axtop[k].plot(xe, ye, "k-", lw=1, clip_on=False)
            axtop[k].plot(xe, -ye, "k-", lw=1, clip_on=False)
            axtop[k].plot(0, lat2y(0.5 * np.pi - incs[k]), "kx", ms=3)
            axtop[k].axis("off")
            axbot[k].plot(t, 1e3 * (flux[k]), "k.", alpha=0.3, ms=1)
            axbot[k].plot(t, 1e3 * (flux0[k]), "C0-", lw=1)
            axbot[k].set_ylim(ymin, ymax)
            if k < nrows:
                div = make_axes_locatable(axtop[nlc - k - 1])
                cax = div.append_axes("right", size="7%", pad="10%")
                cbar = fig.colorbar(im, cax=cax, orientation="vertical")
                cbar.set_label("intensity", fontsize=8)
                cbar.set_ticks([0, 0.5, 1])
                cbar.ax.tick_params(labelsize=6)
                axbot[k].spines["top"].set_visible(False)
                axbot[k].spines["right"].set_visible(False)
                axbot[k].set_xlabel("rotations", fontsize=8)
                axbot[k].set_ylabel("flux [ppt]", fontsize=8)
                axbot[k].set_xticks([0, 1, 2, 3, 4])
                for tick in (
                    axbot[k].xaxis.get_major_ticks()
                    + axbot[k].yaxis.get_major_ticks()
                ):
                    tick.label.set_fontsize(6)
                axbot[k].tick_params(direction="in")
            else:
                axbot[k].axis("off")

        return data, truth, fig

    return data, truth
