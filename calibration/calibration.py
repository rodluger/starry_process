from starry_process.calibrate import run
from starry_process.calibrate.defaults import update_with_defaults
from starry_process.calibrate.log_prob import get_log_prob
from starry_process.latitude import beta2gauss
import os
import subprocess
import glob
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import json
from tqdm.auto import tqdm
from matplotlib import colors
import pickle
from dynesty import utils as dyfunc
from scipy.stats import norm as Normal


# Path to this directory
HERE = os.path.dirname(os.path.abspath(__file__))


# User email
try:
    EMAIL = (
        subprocess.check_output(["git", "config", "user.email"])
        .decode()
        .split("\n")[0]
    )
except:
    EMAIL = None


def _compute_inclination_pdf(path, clobber=False):
    """

    """
    # Check if we already did this
    if clobber or not os.path.exists(os.path.join(path, "inclinations.npz")):

        # Get kwargs
        with open(os.path.join(path, "kwargs.json"), "r") as f:
            kwargs = json.load(f)
        kwargs = update_with_defaults(**kwargs)
        sample_kwargs = kwargs["sample"]
        gen_kwargs = kwargs["generate"]
        plot_kwargs = kwargs["plot"]
        ninc_pts = plot_kwargs["ninc_pts"]
        ninc_samples = plot_kwargs["ninc_samples"]
        ydeg = sample_kwargs["ydeg"]
        baseline_var = sample_kwargs["baseline_var"]
        apply_jac = sample_kwargs["apply_jac"]
        normalized = gen_kwargs["normalized"]

        # Get the data
        with open(os.path.join(path, "results.pkl"), "rb") as f:
            results = pickle.load(f)
        data = np.load(os.path.join(path, "data.npz"))
        t = data["t"]
        ferr = data["ferr"]
        period = data["period"]
        flux = data["flux"]
        nlc = len(flux)

        # Array of inclinations & log prob for each light curve
        inc = np.linspace(0, 90, ninc_pts)
        lp = np.empty((nlc, ninc_samples, ninc_pts))

        # Compile the likelihood function for a given inclination
        log_prob = get_log_prob(
            t,
            flux=None,
            ferr=ferr,
            p=period,
            ydeg=ydeg,
            baseline_var=baseline_var,
            apply_jac=apply_jac,
            normalized=normalized,
            marginalize_over_inclination=False,
        )

        # Resample posterior samples to equal weight
        samples = np.array(results.samples)
        try:
            weights = np.exp(results["logwt"] - results["logz"][-1])
        except:
            weights = results["weights"]
        samples = dyfunc.resample_equal(samples, weights)

        # Compute the posteriors
        for n in tqdm(range(nlc)):
            for j in range(ninc_samples):
                idx = np.random.randint(len(samples))
                lp[n, j] = np.array(
                    [
                        log_prob(flux[n].reshape(1, -1), *samples[idx], i)
                        for i in inc
                    ]
                )

        # Save
        np.savez(os.path.join(path, "inclinations.npz"), inc=inc, lp=lp)

    else:

        data = np.load(os.path.join(path, "data.npz"))
        results = np.load(os.path.join(path, "inclinations.npz"))
        inc = results["inc"]
        lp = results["lp"]

    # Plot
    fig, ax = plt.subplots(5, 10, figsize=(20, 10), sharex=True, sharey=True)
    ax = ax.flatten()
    for n in range(lp.shape[0]):
        for j in range(lp.shape[1]):
            ax[n].plot(
                inc, np.exp(lp[n, j] - lp[n, j].max()), "C0-", lw=1, alpha=0.25
            )
        ax[n].axvline(data["incs"][n], color="C1")
        ax[n].margins(0.1, 0.1)
        if n == 40:
            ax[n].spines["top"].set_visible(False)
            ax[n].spines["right"].set_visible(False)
            ax[n].set_xlabel("inclination", fontsize=10)
            ax[n].set_ylabel("probability", fontsize=10)
            ax[n].set_xticks([0, 30, 60, 90])
            ax[n].set_yticks([])
        else:
            ax[n].axis("off")

    fig.savefig(os.path.join(path, "inclinations.pdf"), bbox_inches="tight")


def run_batch(name, nodes=20, tasks=100, queue="cca", walltime=30, **kwargs):
    """
    Do inference on synthetic datasets.

    This generates `ntasks` datasets (each containing many light curves)
    and runs the full inference problem on each one on the SLURM cluster.
    This is useful for calibrating the model: we use this to show that our
    posterior estimates are unbiased and capture the true variance
    correctly.

    """
    # Output paths
    path = os.path.abspath(name)
    for i in range(tasks):
        if not os.path.exists(os.path.join(path, "{}".format(i))):
            os.makedirs(os.path.join(path, "{}".format(i)))

    # Slurm script
    slurmfile = os.path.join(HERE, "run.sh")
    tasks_per_node = int(np.ceil(tasks / nodes))
    with open(slurmfile, "w") as f:
        print(
            """#!/bin/sh\n"""
            """cd {}\n"""
            """module load disBatch\n"""
            """disBatch.py -t {} taskfile""".format(HERE, tasks_per_node),
            file=f,
        )

    # Script to run each task in disBatch
    taskfile = os.path.join(HERE, "taskfile")
    with open(taskfile, "w") as f:
        print(
            (
                """#DISBATCH REPEAT {} start 0 """
                """python -c "from starry_process.calibrate import run; """
                """run(path='{}/$DISBATCH_REPEAT_INDEX', seed=$DISBATCH_REPEAT_INDEX, **{})" """
                """&> {}/$DISBATCH_REPEAT_INDEX/batch.log"""
            ).format(tasks, path, kwargs, path),
            file=f,
        )

    # Slurm args
    sbatch_args = [
        "sbatch",
        "--partition={}".format(queue),
        "-N {}".format(nodes),
        "--output={}".format(os.path.join(path, "batch.log")),
        "--job-name={}".format(name),
        "--time={}:00:00".format(walltime),
        "--exclusive",
    ]
    if EMAIL is not None:
        sbatch_args.extend(
            ["--mail-user={}".format(EMAIL), "--mail-type=END,FAIL"]
        )

    # Submit!
    sbatch_args.append(slurmfile)
    print("Submitting the job...")
    print(" ".join(sbatch_args))
    subprocess.call(sbatch_args)


def process_batch(name, nodes=20, queue="cca", walltime=24, clobber=False):
    """
    After running a batch, go back and draw samples from the inclination
    PDF for each light curve in each dataset. This is fairly computationally
    expensive, so we also run this on the cluster.

    """
    # Output paths
    path = os.path.abspath(name)
    results_files = glob.glob(os.path.join(path, "*", "results.pkl"))
    tasks = len(results_files)

    # Slurm script
    slurmfile = os.path.join(HERE, "run.sh")
    tasks_per_node = int(np.ceil(tasks / nodes))
    with open(slurmfile, "w") as f:
        print(
            """#!/bin/sh\n"""
            """cd {}\n"""
            """module load disBatch\n"""
            """disBatch.py -t {} taskfile""".format(HERE, tasks_per_node),
            file=f,
        )

    # Script to run each task in disBatch
    taskfile = os.path.join(HERE, "taskfile")
    with open(taskfile, "w") as f:
        print(
            (
                """#DISBATCH REPEAT {} start 0 """
                """cd {}; """
                """python -c "from calibration import _compute_inclination_pdf; """
                """_compute_inclination_pdf(path='{}/$DISBATCH_REPEAT_INDEX', clobber={})" """
                """&> {}/$DISBATCH_REPEAT_INDEX/batch_inc.log"""
            ).format(tasks, HERE, path, clobber, path),
            file=f,
        )

    # Slurm args
    sbatch_args = [
        "sbatch",
        "--partition={}".format(queue),
        "-N {}".format(nodes),
        "--output={}".format(os.path.join(path, "batch_inc.log")),
        "--job-name={}".format(name),
        "--time={}:00:00".format(walltime),
        "--exclusive",
    ]
    if EMAIL is not None:
        sbatch_args.extend(
            ["--mail-user={}".format(EMAIL), "--mail-type=END,FAIL"]
        )

    # Submit!
    sbatch_args.append(slurmfile)
    print("Submitting the job...")
    print(" ".join(sbatch_args))
    subprocess.call(sbatch_args)


def plot_batch(name, bins=10, alpha=0.25, nsig=4):
    """
    Plot the results of a batch run.

    """
    # Get the posterior means and covariances
    path = os.path.abspath(name)
    files = glob.glob(os.path.join(path, "*", "mean_and_cov.npz"))
    mean = np.empty((len(files), 5))
    cov = np.empty((len(files), 5, 5))
    for k, file in enumerate(files):
        data = np.load(file)
        mean[k] = data["mean"]
        cov[k] = data["cov"]

    # Get the true values
    kwargs = update_with_defaults(
        **json.load(
            open(files[0].replace("mean_and_cov.npz", "kwargs.json"), "r")
        )
    )
    truths = [
        kwargs["generate"]["radius"]["mu"],
        kwargs["generate"]["latitude"]["mu"],
        kwargs["generate"]["latitude"]["sigma"],
        kwargs["generate"]["contrast"]["mu"],
        kwargs["generate"]["nspots"]["mu"],
    ]
    labels = [r"$r$", r"$\mu$", r"$\sigma$", r"$c$", r"$n$"]

    # Plot the distribution of posterior means & variances
    fig, ax = plt.subplots(
        2,
        len(truths) + 1,
        figsize=(16, 6),
        gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1, 0.01]),
    )
    fig.subplots_adjust(hspace=0.4)
    for n in range(len(truths)):

        # Distribution of means
        ax[0, n].hist(
            mean[:, n], histtype="step", bins=bins, lw=2, density=True
        )
        ax[0, n].axvline(np.mean(mean[:, n]), color="C0", ls="--")
        ax[0, n].axvline(truths[n], color="C1")

        # Distribution of errors (should be ~ std normal)
        deltas = (mean[:, n] - truths[n]) / np.sqrt(cov[:, n, n])
        ax[1, n].hist(
            deltas,
            density=True,
            histtype="step",
            range=(-4, 4),
            bins=bins,
            lw=2,
        )
        ax[1, n].hist(
            np.random.randn(10000),
            density=True,
            range=(-4, 4),
            bins=bins,
            histtype="step",
            lw=2,
        )

        ax[0, n].set_title(labels[n], fontsize=16)
        ax[0, n].set_xlabel("posterior mean")
        ax[1, n].set_xlabel("posterior error")
        ax[0, n].set_yticks([])
        ax[1, n].set_yticks([])

    # Tweak appearance
    ax[0, -1].axis("off")
    ax[0, -1].plot(0, 0, "C0", ls="--", label="mean")
    ax[0, -1].plot(0, 0, "C1", ls="-", label="truth")
    ax[0, -1].legend(loc="center left")
    ax[1, -1].axis("off")
    ax[1, -1].plot(0, 0, "C0", ls="-", lw=2, label="measured")
    ax[1, -1].plot(0, 0, "C1", ls="-", lw=2, label=r"$\mathcal{N}(0, 1)$")
    ax[1, -1].legend(loc="center left")
    fig.savefig(
        os.path.join(path, "calibration_bias.pdf"), bbox_inches="tight"
    )

    # Now let's plot all of the posteriors on a corner plot
    files = glob.glob(os.path.join(path, "*", "results.pkl"))
    samples = [None for k in range(len(files))]
    ranges = [None for k in range(len(files))]
    for k in tqdm(range(len(files))):

        # Get the samples
        with open(files[k], "rb") as f:
            results = pickle.load(f)
        samples[k] = np.array(results.samples)
        samples[k][:, 1], samples[k][:, 2] = beta2gauss(
            samples[k][:, 1], samples[k][:, 2]
        )
        try:
            weights = np.exp(results["logwt"] - results["logz"][-1])
        except:
            weights = results["weights"]
        samples[k] = dyfunc.resample_equal(samples[k], weights)

        # Get the 4-sigma ranges
        mu = np.mean(samples[k], axis=0)
        std = np.std(samples[k], axis=0)
        ranges[k] = np.array([mu - nsig * std, mu + nsig * std]).T

    # Set plot limits to the maximum of the ranges
    ranges = np.array(ranges)
    ranges = np.array(
        [np.min(ranges[:, :, 0], axis=0), np.max(ranges[:, :, 1], axis=0)]
    ).T

    # Go!
    color = lambda i, alpha: "{}{}".format(
        colors.to_hex("C{}".format(i)),
        ("0" + hex(int(alpha * 256)).split("0x")[-1])[-2:],
    )
    fig = None
    for k in tqdm(range(len(mean))):

        # Plot the 2d hist
        fig = corner(
            samples[k],
            fig=fig,
            labels=labels,
            plot_datapoints=False,
            plot_density=False,
            fill_contours=True,
            no_fill_contours=True,
            color=color(k, alpha),
            contourf_kwargs=dict(),
            contour_kwargs=dict(alpha=0),
            bins=20,
            hist_bin_factor=5,
            smooth=2.0,
            hist_kwargs=dict(alpha=0),
            levels=(1.0 - np.exp(-0.5 * np.array([1.0]) ** 2)),
            range=ranges,
        )

        # Plot the 1d hist
        if k == len(mean) - 1:
            truths_ = truths
        else:
            truths_ = None
        fig = corner(
            samples[k],
            fig=fig,
            labels=labels,
            plot_datapoints=False,
            plot_density=False,
            plot_contours=False,
            fill_contours=False,
            no_fill_contours=True,
            color=color(k, alpha),
            bins=500,
            smooth1d=10.0,
            hist_kwargs=dict(alpha=0.5 * alpha),
            range=ranges,
            truths=truths_,
            truth_color="k",
        )

    # Fix the axis limits
    for k in range(5):
        ax = fig.axes[6 * k]
        ymax = np.max([line._y.max() for line in ax.lines])
        ax.set_ylim(0, 1.1 * ymax)

    # We're done
    fig.savefig(
        os.path.join(path, "calibration_corner.pdf"), bbox_inches="tight"
    )

    # Get the inclination posterior means and covariances (if present)
    inc_files = glob.glob(os.path.join(path, "*", "inclinations.npz"))
    if len(inc_files):

        data_files = [
            file.replace("inclinations", "data") for file in inc_files
        ]
        x = np.linspace(-90, 90, 1000)
        deltas = []

        # Compute the "posterior error" histogram
        for k in tqdm(range(len(inc_files))):
            data = np.load(data_files[k])
            results = np.load(inc_files[k])
            truths = data["incs"]
            inc = results["inc"]
            lp = results["lp"]
            nlc = len(truths)
            nsamples = lp.shape[1]
            for n in range(nlc):
                for j in range(nsamples):
                    pdf = np.exp(lp[n, j] - np.max(lp[n, j]))
                    pdf /= np.trapz(pdf)
                    mean = np.trapz(inc * pdf)
                    var = np.trapz(inc ** 2 * pdf) - mean ** 2
                    deltas.append((mean - truths[n]) / np.sqrt(var))

        # Plot it
        fig, ax = plt.subplots(1, figsize=(6, 3))
        ax.hist(deltas, bins=50, range=(-5, 5), density=True, label="measured")
        ax.hist(
            Normal.rvs(size=len(deltas)),
            bins=50,
            range=(-5, 5),
            histtype="step",
            color="C1",
            density=True,
            label=r"$\mathcal{N}(0, 1)$",
        )
        ax.legend(loc="top right")
        ax.set_xlabel("posterior error")
        ax.set_yticks([])
        fig.savefig(
            os.path.join(path, "inclinations.pdf"), bbox_inches="tight"
        )

