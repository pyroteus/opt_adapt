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
args = parser.parse_args()
demo = args.demo
n = args.n
plot_dir = create_directory(f"{demo}/plots")

fig, axes = plt.subplots()
for method in _implemented_methods:
    J = np.load(f"{demo}/data/uniform_progress_J_{n}_{method}.npy")
    it = np.array(range(len(J))) + 1
    axes.loglog(it, J, label=method)
axes.grid(True)
axes.set_xlabel("Iteration")
axes.set_ylabel("Objective function value")
axes.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/convergence_{n}.png")
