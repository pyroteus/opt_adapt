from pyroteus.log import pyrint
from pyroteus.utility import create_directory, File
from firedrake.meshadapt import RiemannianMetric, adapt
from firedrake_adjoint import *
from pyroteus.metric import space_normalise, enforce_element_constraints
from opt_adapt.opt import *
import argparse
import importlib
import numpy as np
import os
from time import perf_counter
from firedrake import triplot
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
pwd = os.path.abspath(os.path.dirname(__file__))
choices = [name for name in os.listdir(pwd) if os.path.isdir(name)]
parser.add_argument("demo", type=str, choices=choices)
parser.add_argument("--method", type=str, default="gradient_descent")
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--target", type=float, default=1000.0)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--gtol", type=float, default=1.0e-05)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--lr_lowerbound", type=float, default=1e-8)
parser.add_argument("--check_lr", action="store_true")
parser.add_argument("--disp", type=int, default=2)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()
demo = args.demo
method = args.method
n = args.n
target = args.target

# Setup initial mesh
setup = importlib.import_module(f"{demo}.setup")
mesh = setup.initial_mesh(n=n)

# Setup parameter class
params = OptAdaptParameters(
    method,
    options={
        "disp": args.disp,
        "lr": args.lr,
        "lr_lowerbound": args.lr_lowerbound,
        "check_lr": args.check_lr,
        "gtol": args.gtol,
        "maxiter": args.maxiter,
        "target_base": 0.2 * target,
        "target_inc": 0.1 * target,
        "target_max": target,
        "model_options": {
            "no_exports": True,
            "outfile": File(
                f"{demo}/outputs_hessian/{method}/solution.pvd", adaptive=True
            ),
        },
    },
    Rspace=setup.initial_control(mesh).ufl_element().family() == "Real",
)
pyrint(f"Using method {method}")


def adapt_hessian_based(mesh, target=1000.0, norm_order=1.0, **kwargs):
    """
    Adapt the mesh w.r.t. the intersection of the Hessians of
    each component of velocity and pressure.
    :kwarg target: Desired metric complexity (continuous
        analogue of mesh vertex count).
    :kwarg norm_order: Normalisation order :math:`p` for the
    :math:`L^p` normalisation routine.
    """
    metric = space_normalise(setup.hessian(mesh), target, norm_order)
    enforce_element_constraints(metric, 1.0e-05, 500.0, 1000.0)
    if args.disp > 2:
        pyrint("Metric construction complete.")
    newmesh = adapt(mesh, RiemannianMetric(mesh).assign(metric))
    if args.disp > 2:
        pyrint("Mesh adaptation complete.")
    return newmesh


cpu_timestamp = perf_counter()
op = OptimisationProgress()
failed = False
if args.debug:
    m_opt = minimise(
        setup.forward_run,
        mesh,
        setup.initial_control,
        adapt_fn=adapt_hessian_based,
        method=method,
        params=params,
        op=op,
    )
    cpu_time = perf_counter() - cpu_timestamp
    print(f"Uniform optimisation completed in {cpu_time:.2f}s")
else:
    try:
        m_opt = minimise(
            setup.forward_run,
            mesh,
            setup.initial_control,
            adapt_fn=adapt_hessian_based,
            method=method,
            params=params,
            op=op,
        )
        cpu_time = perf_counter() - cpu_timestamp
        print(f"Hessian-based optimisation completed in {cpu_time:.2f}s")
    except Exception as exc:
        cpu_time = perf_counter() - cpu_timestamp
        print(f"Hessian-based optimisation failed after {cpu_time:.2f}s")
        print(f"Reason: {exc}")
        failed = True
create_directory(f"{demo}/data")
t = op.t_progress
m = np.array([m.dat.data[0] for m in op.m_progress]).flatten()
J = op.J_progress
dJ = np.array([dj.dat.data[0] for dj in op.dJ_progress]).flatten()
nc = op.nc_progress
np.save(f"{demo}/data/hessian_progress_t_{n}_{method}", t)
np.save(f"{demo}/data/hessian_progress_m_{n}_{method}", m)
np.save(f"{demo}/data/hessian_progress_J_{n}_{method}", J)
np.save(f"{demo}/data/hessian_progress_dJ_{n}_{method}", dJ)
np.save(f"{demo}/data/hessian_progress_nc_{n}_{method}", nc)
with open(f"{demo}/data/hessian_{target:.0f}_{method}.log", "w+") as f:
    note = " (FAIL)" if failed else ""
    f.write(f"cpu_time: {cpu_time}{note}\n")


plot_dir = create_directory(f"{demo}/plots")
fig, axes = plt.subplots()
triplot(op.mesh_progress[-1], axes=axes)
axes.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/mesh_hessian_{method}.png")