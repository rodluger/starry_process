import os
import subprocess
import glob
import numpy as np


def run_batch(
    path,
    datasets=100,
    clobber=False,
    slurm_nodes=10,
    slurm_queue="cca",
    slurm_walltime=60,
    slurm_email=None,
    **kwargs
):
    """
    Do inference on synthetic datasets.

    This generates ``datasets`` datasets (each containing many light curves)
    and runs the full inference problem on each one (optionally) on a SLURM 
    cluster. This is useful for calibrating the model: we use this to show 
    that our posterior estimates are unbiased and capture the true variance
    correctly.

    """
    # Output paths
    path = os.path.abspath(path)
    for i in range(datasets):
        if not os.path.exists(os.path.join(path, "{}".format(i))):
            os.makedirs(os.path.join(path, "{}".format(i)))

    # Do we have slurm?
    try:
        subprocess.check_output(["sbatch", "--version"])
        use_slurm = True
    except:
        use_slurm = False

    if use_slurm:

        # Script to run each task in disBatch
        taskfile = os.path.join(path, "taskfile")
        with open(taskfile, "w") as f:
            print(
                (
                    """#DISBATCH REPEAT {} start 0 """
                    """python -c "from starry_process.calibrate import run; """
                    """run(path='{}/$DISBATCH_REPEAT_INDEX', seed=$DISBATCH_REPEAT_INDEX, clobber={}, **{})" """
                    """&> {}/$DISBATCH_REPEAT_INDEX/batch.log"""
                    """\n"""
                    """#DISBATCH BARRIER"""
                    """\n"""
                    """python -c "from starry_process.calibrate import plot_batch; plot_batch('{}')" """
                ).format(datasets, path, clobber, kwargs, path, path),
                file=f,
            )

        # Slurm script
        slurmfile = os.path.join(path, "run.sh")
        tasks_per_node = int(np.ceil(datasets / slurm_nodes))
        with open(slurmfile, "w") as f:
            print(
                """#!/bin/sh\n"""
                """cd {}\n"""
                """module load disBatch\n"""
                """disBatch.py -t {} taskfile""".format(path, tasks_per_node),
                file=f,
            )

        # Slurm args
        sbatch_args = [
            "sbatch",
            "--partition={}".format(slurm_queue),
            "-N {}".format(slurm_nodes),
            "--output={}".format(os.path.join(path, "batch.log")),
            "--job-name={}".format(path.split("/")[-1]),
            "--time={}:00:00".format(slurm_walltime),
            "--exclusive",
        ]
        try:
            if email is None:
                email = (
                    subprocess.check_output(["git", "config", "user.email"])
                    .decode()
                    .split("\n")[0]
                )
            sbatch_args.extend(
                ["--mail-user={}".format(email), "--mail-type=END,FAIL"]
            )
        except:
            pass

        # Submit!
        sbatch_args.append(slurmfile)
        print("Submitting the job...")
        print(" ".join(sbatch_args))
        subprocess.call(sbatch_args)

    else:

        # Run locally in serial (will take a loooong time)
        from . import run
        from .plot import plot_batch
        from tqdm.auto import tqdm

        for k in tqdm(
            range(datasets), disable=bool(int(os.getenv("NOTQDM", "0")))
        ):
            run(
                path="{}/{}".format(path, k), seed=k, clobber=clobber, **kwargs
            )
        plot_batch(path)
