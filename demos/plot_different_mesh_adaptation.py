from opt_adapt.opt import _implemented_methods
from opt_adapt.utils import create_directory
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
pwd = os.path.abspath(os.path.dirname(__file__))
choices = [name for name in os.listdir(pwd) if os.path.isdir(name)]
parser.add_argument("demo", type=str, choices=choices)
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--method", type=str, choices=_implemented_methods.keys())
args = parser.parse_args()
demo = args.demo
n = args.n
method = args.method
plot_dir = create_directory(f"{demo}/plots")

runs = ["uniform", "hessian", "go"]
fig, axes = plt.subplots()
for run in runs:
    J = np.load(f"{demo}/data/{run}_progress_J_{n}_{method}.npy")
    t = np.load(f"{demo}/data/{run}_progress_t_{n}_{method}.npy")
    times = [sum(t[:i]) for i in range(len(t))]
    axes.loglog(times, J, label=method)
axes.grid(True)
axes.set_xlabel("Cumulative CPU time (seconds)")
axes.set_ylabel("Objective function value")
axes.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/qoi_vs_time_{method}_{n}.png")
plt.close()
fig, axes = plt.subplots()
for run in runs:
    J = np.load(f"{demo}/data/{run}_progress_J_{n}_{method}.npy")
    it = np.array(range(len(J))) + 1
    axes.loglog(it, J, label=run)
axes.grid(True)
axes.set_xlabel("Iteration")
axes.set_ylabel("Objective function value")
axes.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/qoi_vs_it_{method}_{n}.png")
plt.close()