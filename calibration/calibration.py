from starry_process.calibrate import run
from starry_process.calibrate.defaults import update_with_defaults
import os
import subprocess
import glob
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import json


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


def run_batch(name, nodes=5, tasks=10, queue="cca", walltime=8, **kwargs):
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


def plot_batch(name, bins=10):
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

    ax[0, -1].axis("off")
    ax[0, -1].plot(0, 0, "C0", ls="--", label="mean")
    ax[0, -1].plot(0, 0, "C1", ls="-", label="truth")
    ax[0, -1].legend(loc="center left")

    ax[1, -1].axis("off")
    ax[1, -1].plot(0, 0, "C0", ls="-", lw=2, label="measured")
    ax[1, -1].plot(0, 0, "C1", ls="-", lw=2, label=r"$\mathcal{N}(0, 1)$")
    ax[1, -1].legend(loc="center left")

    fig.savefig(os.path.join(path, "calibration.pdf"), bbox_inches="tight")
