from starry_process.calibrate import run
import os
import subprocess
import glob
import numpy as np


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


def clean():
    for file in glob.glob(os.path.join(HERE, "taskfile*")):
        os.remove(file)
    if os.path.exists("run.sh"):
        os.remove("run.sh")
