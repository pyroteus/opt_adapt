import firedrake as fd
import firedrake_adjoint as fd_adj
from firedrake.adjoint import get_solve_blocks
import ufl

from opt_adapt.utils import pprint

import numpy as np
from time import perf_counter

from opt_adapt.matrix import *

__all__ = ["OptimisationProgress", "OptAdaptParameters", "identity_mesh", "get_state", "minimise"]


class OptimisationProgress:
    """
    Class for stashing progress of the optimisation
    routine.
    """

    def __init__(self):
        self.J_progress = []
        self.m_progress = []
        self.dJdm_progress = []


class OptAdaptParameters:
    """
    Class for holding parameters associated with the
    combined optimisation-adaptation routine.
    """

    def __init__(self, options={}):
        self.model_options = {}
        self.disp = 0
        self.lr = 0.001  # Step length / learning rate
        self.transfer_fn = fd.project  # Mesh-to-mesh interpolation method

        self.maxiter = 101  # Maximum iteration count
        self.gtol = 1.0e-05  # Gradient relative tolerance
        self.dtol = 1.0001  # Divergence tolerance i.e. 0.01% increase
        self.element_rtol = 0.005  # Element count relative tolerance
        self.qoi_rtol = 0.005  # QoI relative tolerance

        self.target_base = 200.0  # Base target metric complexity
        self.target_inc = 200.0  # Increment for target metric complexity
        self.target_max = 1000.0  # Eventual target metric complexity

        # Apply user-specified values
        for key, value in options.items():
            if not hasattr(self, key):
                raise ValueError(f"Option {key} not recognised")
            self.__setattr__(key, value)

# Only works for problem with control paramter in r-space
def line_search(forward_run, mesh, u, P, J, dJ, Rspace, alpha=1e-4, max_search_iter=100):
    u = u.copy(deepcopy=True)
    
    lr = 1
    if Rspace:
        initial_slope = float(dJ) * float(P)
    else:
        initial_slope = np.dot(dJ.dat.data, P.dat.data)

    if initial_slope==0.0:
        return 1.0
    
    pprint(f"  Applying line search with alpha = {alpha} and tau = {tau}")
    ext = ""
    for i in range(max_search_iter):
        pprint(f"  {i:3d}:      lr = {lr:.4e}{ext}")
        u_plus = u + lr*P 
        J_plus, u_plus = forward_run(mesh, u_plus)
        ext = f"  diff {J_plus - J:.4e}"
        # check Armijo rule:
        if J_plus-J <= alpha*lr*initial_slope:
            break
        lr /= 2
    else:
        raise Exception("Line search did not converge")
    pprint(f"  converged lr = {lr:.4e}")
    return lr


def _gradient_descent(it, forward_run, m, params, u, u_, dJ_, Rspace=False):
    """
    Take one gradient descent iteration.
    :arg it: the current iteration number
    :arg forward_run: a Python function that
        implements the forward model and
        computes the objective functional
    :arg m: the current mesh
    :kwarg params: :class:`OptAdaptParameters` instance
        containing parameters for the optimisation and
        adaptation routines
    :arg u: the current control value
    :arg u_: the previous control value
    :arg dJ_: the previous gradient value
    :kwarg Rspace: is the prognostic function
        space of type 'Real'?
    """

    # Annotate the tape and compute the gradient
    J, u = forward_run(m, u, **params.model_options)
    dJ = fd_adj.compute_gradient(J, fd_adj.Control(u))
    yield {"J": J, "u": u.copy(deepcopy=True), "dJ": dJ.copy(deepcopy=True)}

    # Choose step length
    if u_ is None or dJ_ is None:
        lr = params.lr
    else:
        if Rspace:
            dJ_ = fd.Function(dJ).assign(dJ_)
            u_ = fd.Function(u).assign(u_)
        else:
            dJ_ = params.transfer_fn(dJ_, dJ.function_space())
            u_ = params.transfer_fn(u_, u.function_space())
        dJ_diff = fd.assemble(ufl.inner(dJ_ - dJ, dJ_ - dJ) * ufl.dx)
        lr = abs(fd.assemble(ufl.inner(u_ - u, dJ_ - dJ) * ufl.dx) / dJ_diff)

    # Take a step downhill
    u -= lr * dJ
    yield {"lr": lr, "u+": u, "u-": u_, "dJ-": dJ_, "B": None}


def _BFGS(it, forward_run, m, params, u, u_, dJ_, B, Rspace=False):
    """
    A second order routine
    """
    J, u = forward_run(m, u, **params.model_options)
    dJ = fd_adj.compute_gradient(J, fd_adj.Control(u))
    yield {"J": J, "u": u.copy(deepcopy=True), "dJ": dJ.copy(deepcopy=True)}
    
    if Rspace:
            if u_ is None or dJ_ is None:
                B = 1 
            else:
                dJ_ = fd.Function(dJ).assign(dJ_)
                u_ = fd.Function(u).assign(u_)
                s = float(u) - float(u_)
                y = float(dJ) - float(dJ_)
                B = y / s
            P = -float(dJ) / B
            lr = line_search(forward_run, m, u, P, J, dJ, Rspace)
            u += lr * P
            yield {"lr": lr, "u+": u, "u-": u_, "dJ-": dJ_, "B": B}
            return
    
    if B is None:
        B = Matrix(u.function_space())

    P = B.scale(-1).solve(dJ)
    lr = params.lr
    u += lr * P

    if u_ is not None and dJ_ is not None:
        dJ_ = params.transfer_fn(dJ_, dJ.function_space())
        u_ = params.transfer_fn(u_, u.function_space())
        
        s = u.copy(deepcopy=True)
        s -= u_
        y = dJ.copy(deepcopy=True)
        y -= dJ_
        
        y_star_s = np.dot(y.dat.data, s.dat.data) 
        y_y_star = OuterProductMatrix(y,y)
        second_term = y_y_star.scale(1/y_star_s)
        
        Bs = B.multiply(s) 
        sBs = np.dot(s.dat.data, Bs.dat.data)
        sB = B.multiply(s, side="left")
        BssB = OuterProductMatrix(Bs,sB)
        third_term = BssB.scale(1/sBs)
        
        B.add(second_term)
        B.subtract(third_term)

    yield {"lr": lr, "u+": u, "u-": u_, "dJ-": dJ_, "B": B}


def _newton(it, forward_run, m, params, u, u_, dJ_, B, Rspace=False):
    """
    A second order routine
    """
    J, u = forward_run(m, u, **params.model_options)
    dJ = fd_adj.compute_gradient(J, fd_adj.Control(u))
    H = compute_full_hessian(J, fd_adj.Control(u))
    yield {"J": J, "u": u.copy(deepcopy=True), "dJ": dJ.copy(deepcopy=True)}
    
    try: 
        P = H.scale(-1).solve(dJ)
    except np.linalg.LinAlgError:
        raise Exception("Hessian is singular, please try the other methods")
    
    lr = line_search(forward_run, m, u, P, J, dJ, Rspace)
    u += lr * P
    yield {"lr": lr, "u+": u, "u-": None, "dJ-": None, "B": None} 


_implemented_methods = {
    "gradient_descent": {"func": _gradient_descent, "order": 1},
    "BFGS": {"func": _BFGS, "order": 2},
    "newton": {"func": _newton, "order": 2},
}


def identity_mesh(mesh, **kwargs):
    """
    The simplest possible adaptation function: the
    identity function.
    """
    return mesh


def get_state(adjoint=False, tape=None):
    """
    Extract the current state from the tape (velocity and
    elevation).
    :kwarg adjoint: If ``True``, return the corresponding
        adjoint state variables.
    """
    solve_block = get_solve_blocks()[0]
    return solve_block.adj_sol if adjoint else solve_block._outputs[0].saved_output


def minimise(
    forward_run,
    mesh,
    initial_control_fn,
    adapt_fn=identity_mesh,
    params=OptAdaptParameters(),
    method="gradient_descent",
    op=None,
):
    """
    Custom minimisation routine, where the tape is
    re-annotated each iteration in order to support
    mesh adaptation.
    :arg forward_run: a Python function that
        implements the forward model and
        computes the objective functional
    :arg mesh: the initial mesh
    :arg init_control_fn: a Python function that takes
        a mesh as input and initialises the control
    :kwarg adapt_fn: a Python function that takes a
        mesh as input and adapts it to get a new mesh
    :kwarg params: :class:`OptAdaptParameters` instance
        containing parameters for the optimisation and
        adaptation routines
    :kwarg method: the optimisation method
    :kwarg op: optional :class:`OptimisationProgress`
        instance
    """
    try:
        step, order = _implemented_methods[method].values()
    except KeyError:
        raise ValueError(f"Method '{method}' not recognised")
    op = op or OptimisationProgress()
    tape = fd_adj.get_working_tape()
    tape.clear_tape()
    u_plus = initial_control_fn(mesh)
    Rspace = u_plus.ufl_element().family() == "Real"
    dJ_init = None
    target = params.target_base
    B = None

    # Enter the optimisation loop
    nc_ = mesh.num_cells()
    adaptor = adapt_fn
    mesh_conv_it = []
    for it in range(1, params.maxiter + 1):
        term_msg = f"Terminated after {it} iterations due to "
        u_ = None if it == 1 else op.m_progress[-1]
        dJ_ = None if it == 1 else op.dJdm_progress[-1]
        if order == 1:
            args = (u_plus, u_, dJ_)
        elif order == 2:
            args = (u_plus, u_, dJ_, B)
        else:
            raise NotImplementedError(f"Method {method} unavailable")

        # Take a step
        cpu_timestamp = perf_counter()
        out = {}
        for o in step(it, forward_run, mesh, params, *args, Rspace=Rspace):
            out.update(o)
        J, u, dJ = out["J"], out["u"], out["dJ"]
        lr, u_plus, u_, dJ_, B = out["lr"], out["u+"], out["u-"], out["dJ-"], out["B"]

        # Print to screen, if requested
        if params.disp > 0:
            t = perf_counter() - cpu_timestamp
            g = dJ.dat.data[0] if Rspace else fd.norm(dJ)
            msgs = [f"{it:3d}:  J = {J:9.4e}"]
            if Rspace:
                msgs.append(f"m = {u_plus.dat.data[0]:9.4e}")
            if Rspace:
                msgs.append(f"dJdm = {g:11.4e}")
            else:
                msgs.append(f"||dJdm|| = {g:9.4e}")
            msgs.append(f"step length = {lr:9.4e}")
            msgs.append(f"#elements = {nc_:5d}")
            msgs.append(f"time = {t:.2f}s")
            pprint(",  ".join(msgs))

        # Stash progress
        op.J_progress.append(J)
        op.m_progress.append(u)
        op.dJdm_progress.append(dJ)

        # Check for QoI divergence
        if it > 1 and np.abs(J / np.min(op.J_progress)) > params.dtol:
            raise fd.ConvergenceError(term_msg + "dtol divergence")

        # Check for gradient convergence
        if it == 1:
            dJ_init = fd.norm(dJ)
        elif fd.norm(dJ) / dJ_init < params.gtol:
            if params.disp > 0:
                pprint(term_msg + "gtol convergence")
            break

        # Check for reaching maximum iteration count
        if it == params.maxiter:
            raise fd.ConvergenceError(term_msg + "reaching maxiter")

        # Ramp up the target complexity
        target = min(target + params.target_inc, params.target_max)

        # Adapt the mesh
        mesh = adaptor(mesh, target=target, control=u_plus)
        nc = mesh.num_cells()

        # Check for mesh convergence
        if adaptor != identity_mesh and np.abs(nc - nc_) < params.element_rtol * nc_:
            conv = np.array([op.J_progress[i] for i in mesh_conv_it])
            if (np.abs(J - conv) < params.qoi_rtol * np.abs(conv)).any():
                pprint(term_msg + "qoi_rtol convergence")
                break
            mesh_conv_it.append(it)
            if params.disp > 1:
                pprint("NOTE: turning adaptation off due to element_rtol convergence")
            adaptor = identity_mesh
            continue
        else:
            adaptor = adapt_fn
        nc_ = nc

        # Check for QoI convergence
        if it > 1:
            J_ = op.J_progress[-2]
            if np.abs(J - J_) < params.qoi_rtol * np.abs(J_):
                if adaptor != identity_mesh:
                    conv = np.array([op.J_progress[i] for i in mesh_conv_it])
                    if (np.abs(J - conv) < params.qoi_rtol * np.abs(conv)).any():
                        pprint(term_msg + "qoi_rtol convergence")
                        break
                mesh_conv_it.append(it)
                if params.disp > 1:
                    pprint("NOTE: turning adaptation off due to qoi_rtol convergence")
                adaptor = identity_mesh
                continue
            else:
                adaptor = adapt_fn

        # Clean up
        tape.clear_tape()
    return u_plus
