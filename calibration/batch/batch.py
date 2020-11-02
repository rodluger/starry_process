from starry_process import calibrate
import pickle
import numpy as np
from dynesty import utils as dyfunc
import os
import subprocess

__PATH__ = os.path.dirname(os.path.abspath(__file__))


# Global settings
kwargs = {}


def perform(seed):
    """
    Run the task on a cluster node.

    """
    # Set the seed
    kwargs["seed"] = seed

    # Generate
    data = calibrate.generate(**kwargs)
    np.savez("data{:04d}.npz".format(seed), **data)

    # Sample
    results = calibrate.sample(data, **kwargs)
    pickle.dump(results, open("results{:04d}.pkl".format(seed), "wb"))

    # Get posterior mean and cov
    samples = np.array(results.samples)
    try:
        weights = np.exp(results["logwt"] - results["logz"][-1])
    except:
        weights = results["weights"]
    samples = dyfunc.resample_equal(samples, weights)
    mean, cov = dyfunc.mean_and_cov(results.samples, weights)
    np.savez("results{:04d}.npz".format(seed), mean=mean, cov=cov)


def submit(
    nodes=2,
    tasks=10,
    cpus_per_task=2,
    queue="cca",
    email=None,
    walltime=8,
    **kwargs
):
    """
    Submit the cluster job.

    """
    slurmfile = os.path.join(__PATH__, "perform.sh")
    sbatch_args = [
        "sbatch",
        "--partition=%s" % queue,
        "--array=0-{}".format(tasks),
        "-N{}".format(nodes),
        "-cpus_per_task={}".format(cpus_per_task),
        "--export=ALL,DIRNAME={}".format(__PATH__),
        "--output={}".format(os.path.join(__PATH__, "batch_%A_%a.log")),
        "--job-name=batch",
        "--time={}:00:00".format(walltime),
        "--exclusive",
    ]
    if email is not None:
        sbatch_args.append(
            ["--mail-user={}".format(email), "--mail-type=END,FAIL"]
        )
    sbatch_args.append(slurmfile)
    print("Submitting the job...")
    print(" ".join(sbatch_args))
    subprocess.call(sbatch_args)


if __name__ == "__main__":
    submit()
