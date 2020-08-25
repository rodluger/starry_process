from starry_process.gp import YlmGP
from line_profiler import LineProfiler
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


def run_profiler():
    gp = YlmGP(10)
    profile = LineProfiler()
    profile.add_function(gp.set_params)
    profile.run("gp.set_params()")
    profile.print_stats()


def run_timing(lmax=20, nruns=10):
    ls = np.arange(1, lmax + 1)
    t = np.zeros(lmax)
    std = np.zeros(lmax)
    for i, l in tqdm(enumerate(ls), total=lmax):
        gp = YlmGP(l)
        t_ = np.zeros(nruns)
        for k in range(nruns):
            tstart = time.time()
            gp.set_params()
            t_[k] = time.time() - tstart
        t[i] = np.median(t_)
        std[i] = 1.4826 * np.median(np.abs(t_ - t[i]))

    plt.plot(ls, t)
    plt.fill_between(ls, t - std, t + std, color="C0", alpha=0.25)
    plt.xlabel("spherical harmonic degree")
    plt.ylabel("time per covariance evaluation [s]")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    plt.style.use("default")
    run_timing()
