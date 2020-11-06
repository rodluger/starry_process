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


def run_single(name, seed=0, queue="cca", walltime=8, **kwargs):
    """
    
    """
    # Output path
    path = os.path.abspath(name)
    if not os.path.exists(os.path.join(path, "{}".format(seed))):
        os.makedirs(os.path.join(path, "{}".format(seed)))

    # Slurm script
    slurmfile = os.path.join(HERE, "run.sh")
    with open(slurmfile, "w") as f:
        print(
            (
                """#!/bin/sh\n"""
                """python -c "from starry_process.calibrate import run; """
                """run(path='{}/{}', plot=False, seed={}, **{})" """
                """&> {}/{}/single.log"""
            ).format(path, seed, seed, kwargs, path, seed),
            file=f,
        )

    # Slurm args
    sbatch_args = [
        "sbatch",
        "--partition={}".format(queue),
        "-N 1",
        "--output={}".format(
            os.path.join(path, "{}".format(seed), "batch.log")
        ),
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


def run_batch(name, nodes=20, tasks=100, queue="cca", walltime=30, **kwargs):
    """
    

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
                """run(path='{}/$DISBATCH_REPEAT_INDEX', plot=False, seed=$DISBATCH_REPEAT_INDEX, **{})" """
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


def plot_batch(name, bins=10, alpha=0.25, nsig=4):
    """

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


def process_batch_inclinations(name, nodes=20, queue="cca", walltime=5):
    """
    
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
                """python -c "from calibration import process_inclinations; """
                """process_inclinations(path='{}/$DISBATCH_REPEAT_INDEX')" """
                """&> {}/$DISBATCH_REPEAT_INDEX/batch_inc.log"""
            ).format(tasks, HERE, path, path),
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


def process_inclinations(path):
    """

    """

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
