import matplotlib.pyplot as plt
import numpy as np
import argparse
from thetis import create_directory


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--uniform",
    nargs="+",
    type=int,
    default=[],
)
parser.add_argument(
    "--hessian",
    nargs="+",
    type=int,
    default=[],
)
args = parser.parse_args()


def plot_progress(axes, method, ext, gradients=True, **kwargs):
    """
    Plot the progress of an optimisation run, in terms
    of control value vs. objective functional value.

    :arg axes: the figure axes to plot on
    :arg method: the optimisation method
    :kwarg gradients: should gradients also be plotted?
    :kwargs: to be passed to matplotlib's plotting functions
    """
    m = np.load(f"data/{method}_progress_m_{ext}.npy")
    off = np.abs(m - 250.0)
    J = -np.load(f"data/{method}_progress_J_{ext}.npy") / 1000
    print(f"Converged offset ({method}): {off[-1]:.2f}m offset => {J[-1]:.2f} kW")
    axes.plot(off, J, "x", **kwargs)
    kwargs.pop("label")
    axes.arrow(x=off[-1], y=J[-1] + 20, dx=0, dy=-10, width=0.5, **kwargs)
    if gradients:
        dJdm = np.load(f"data/{method}_progress_dJdm_{ext}.npy") / 1000
        if m[-1] > 250:
            dJdm = -dJdm
        dc = 2.0
        for c, f, g in zip(off, J, dJdm):
            x = np.array([c - dc, c + dc])
            axes.plot(x, f + g * (x - c), "-", **kwargs)


fig, axes = plt.subplots()
i = 0
for n in args.uniform:
    plot_progress(axes, "uniform", n, label=f"Uniform {n}", color=f"C{i}")
    i += 1
for target in args.hessian:
    plot_progress(
        axes,
        "hessian",
        target,
        label=f"Hessian-based {target}",
        color=f"C{i}",
    )
    i += 1
axes.set_xlabel(r"$y$-offset of second turbine ($\mathrm{m}$)")
axes.set_ylabel(r"Power output ($\mathrm{kW}$)")
# axes.set_xlim([0, 60])
axes.grid(True, which="both")
axes.legend()
plt.tight_layout()
create_directory("plots")
plt.savefig("plots/progress.pdf")
