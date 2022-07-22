from opt_adapt.matrix import *
from opt_adapt.utils import pprint
import firedrake as fd
import firedrake_adjoint as fd_adj
from firedrake.adjoint import get_solve_blocks
import ufl
import numpy as np
from time import perf_counter


__all__ = [
    "OptimisationProgress",
    "OptAdaptParameters",
    "identity_mesh",
    "get_state",
    "minimise",
]


class OptimisationProgress:
    """
    Class for stashing progress of the optimisation
    routine.
    """

    def __init__(self):
        self.t_progress = []
        self.J_progress = []
        self.m_progress = []
        self.dJ_progress = []
        self.ddJ_progress = []
        self.nc_progress = []
        self.mesh_progress = []


class OptAdaptParameters:
    """
    Class for holding parameters associated with the
    combined optimisation-adaptation routine.
    """

    def __init__(self, method, Rspace=False, options={}):
        """
        :arg method: the optimisation method
        :kwarg Rspace: is the control in R-space?
        :kwarg options: a dictionary of parameters and values
        """
        if method not in _implemented_methods:
            raise ValueError(f"Method {method} not recognised")
        self.method = method
        self.Rspace = Rspace
        self.disp = 0
        self.model_options = {}

        """
        Mesh-to-mesh interpolation method
        """
        if Rspace:
            self.transfer_fn = lambda f, fs: fd.Function(fs).assign(f)
        else:
            self.transfer_fn = fd.project

        """
        Initial step length
        """
        lr = options.pop("lr", None)
        if lr is not None:
            self.lr = lr
        elif method == "gradient_descent":
            self.lr = 0.001
        else:
            self.lr = 0.1
        self.lr_lowerbound = 1e-8

        """
        Whether needs to check lr
        """
        check_lr = options.pop("check_lr", None)
        if lr is not None:
            self.check_lr = check_lr
        elif "quasi_newton" in _implemented_methods[method]["type"]:
            self.check_lr = True
        else:
            self.check_lr = False

        """
        Parameters for combined optimisation-adaptation routine
        """
        self.maxiter = 101  # Maximum iteration count
        self.gtol = 1.0e-05  # Gradient relative tolerance
        self.gtol_loose = 1.0e-03
        self.dtol = 1.01  # Divergence tolerance i.e. 1% increase
        self.element_rtol = 0.005  # Element count relative tolerance
        self.qoi_rtol = 0.005  # QoI relative tolerance

        """
        Parameters of Adam
        """
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8

        """
        Line search parameters
        """
        self.line_search = True
        self.ls_rtol = 0.1  # Relative tolerance
        self.ls_frac = 0.5  # Fraction to reduce the step by
        self.ls_maxiter = 100  # Maximum iteration count

        """
        Parameters related to target metric complexity
        """
        self.target_base = 200.0  # Base target metric complexity
        self.target_inc = 200.0  # Increment for target metric complexity
        self.target_max = 1000.0  # Eventual target metric complexity

        # Apply user-specified values
        for key, value in options.items():
            if not hasattr(self, key):
                raise ValueError(f"Option {key} not recognised")
            self.__setattr__(key, value)


def dotproduct(f, g):
    """
    The dotproduct of two varaible in the same functionspace.
    """
    return np.dot(f.dat.data, g.dat.data)


def line_search(forward_run, m, u, P, J, dJ, params):
    """
    Apply a backtracking line search method to
    compute the step length / learning rate (lr).

    :arg forward_run: a Python function that
        implements the forward model and
        computes the objective functional
    :arg m: the current mesh
    :arg u: the current control value
    :arg P: the current descent direction
    :arg J: the current value of objective function
    :arg dJ: the current gradient value
    :kwarg params: :class:`OptAdaptParameters` instance
        containing parameters for the optimisation and
        adaptation routines
    """

    lr = params.lr
    if not params.line_search:
        return lr
    alpha = params.ls_rtol
    tau = params.ls_frac
    maxiter = params.ls_maxiter
    disp = params.disp

    # Compute initial slope
    initial_slope = dotproduct(dJ, P)
    if np.isclose(initial_slope, 0.0):
        return params.lr

    # Perform line search
    if disp > 0:
        pprint(f"  Applying line search with alpha = {alpha} and tau = {tau}")
    ext = ""
    for i in range(maxiter):
        if disp > 0:
            pprint(f"  {i:3d}:      lr = {lr:.4e}{ext}")
        u_plus = u + lr * P
        J_plus, u_plus = forward_run(m, u_plus)
        ext = f"  diff {J_plus - J:.4e}"

        # Check Armijo rule:
        if J_plus - J <= alpha * lr * initial_slope:
            break
        lr *= tau
    else:
        raise Exception("Line search did not converge")
    if disp > 0:
        pprint(f"  converged lr = {lr:.4e}")
    return lr


def _gradient_descent(it, forward_run, m, params, u, u_, dJ_):
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
    """

    # Annotate the tape and compute the gradient
    J, u = forward_run(m, u, **params.model_options)
    dJ = fd_adj.compute_gradient(J, fd_adj.Control(u))
    yield {"J": J, "u": u.copy(deepcopy=True), "dJ": dJ.copy(deepcopy=True)}

    # Choose step length
    if u_ is None or dJ_ is None:
        lr = params.lr
    else:
        dJ_ = params.transfer_fn(dJ_, dJ.function_space())
        u_ = params.transfer_fn(u_, u.function_space())
        dJ_diff = fd.assemble(ufl.inner(dJ_ - dJ, dJ_ - dJ) * ufl.dx)
        lr = abs(fd.assemble(ufl.inner(u_ - u, dJ_ - dJ) * ufl.dx) / dJ_diff)

    # Take a step downhill
    u -= lr * dJ
    yield {"lr": lr, "u+": u}


def _adam(it, forward_run, m, params, u, a_, b_):
    """
    Take one Adam iteration.

    :arg it: the current iteration number
    :arg forward_run: a Python function that
        implements the forward model and
        computes the objective functional
    :arg m: the current mesh
    :kwarg params: :class:`OptAdaptParameters` instance
        containing parameters for the optimisation and
        adaptation routines
    :arg u: the current control value
    :arg a_: the previous first moment variable value
    :arg b_: the previous second moment variable value
    """

    lr = params.lr
    beta_1 = params.beta_1
    beta_2 = params.beta_2
    epsilon = params.epsilon

    # Annotate the tape and compute the gradient
    J, u = forward_run(m, u, **params.model_options)
    dJ = fd_adj.compute_gradient(J, fd_adj.Control(u))
    yield {"J": J, "u": u.copy(deepcopy=True), "dJ": dJ.copy(deepcopy=True)}

    if a_ is None or b_ is None:
        a_ = fd.Function(dJ.function_space())
        b_ = fd.Function(dJ.function_space())

    # Find the descent direction
    dJ_square = fd.Function(dJ)
    dJ_square *= dJ
    a = fd.Function(dJ).assign(beta_1 * a_ + (1 - beta_1) * dJ)
    b = fd.Function(dJ).assign(beta_2 * b_ + (1 - beta_2) * dJ_square)
    a_hat = fd.Function(dJ).assign(a / (1 - pow(beta_1, it)))
    b_hat = fd.Function(dJ).assign(b / (1 - pow(beta_2, it)))
    P = fd.Function(dJ).assign(-1 * a_hat / (pow(b_hat, 0.5) + epsilon))

    # Take a step downhill
    lr = line_search(forward_run, m, u, P, J, dJ, params)
    u += lr * P

    yield {"lr": lr, "u+": u, "a-": a, "b-": b}


def _lbfgs(it, forward_run, m, params, u, rho, s, y, n=5):
    """
    Take one L-BFGS iteration.

    :arg it: the current iteration number
    :arg forward_run: a Python function that
        implements the forward model and
        computes the objective functional
    :arg m: the current mesh
    :kwarg params: :class:`OptAdaptParameters` instance
        containing parameters for the optimisation and
        adaptation routines
    :arg u: the current control value
    :arg rho: the list to store previous n rhok values
    :arg s: the list to store previous n sk values
    :arg y: the list to store previous n yk values
    :arg n: the history size of rho, s, y
    """

    # Annotate the tape and compute the gradient
    J_, u_ = forward_run(m, u, **params.model_options)
    dJ_ = fd_adj.compute_gradient(J_, fd_adj.Control(u_))
    yield {"J": J_, "u": u_.copy(deepcopy=True), "dJ": dJ_.copy(deepcopy=True)}

    if s is None or y is None or rho is None:
        rho = []
        s = []
        y = []

    # Update the descent direction P
    n_ = len(s)
    q = dJ_.copy(deepcopy=True)
    if n_ == 0:
        H = Matrix(dJ_.function_space())
        P = H.multiply(q)
    else:
        a = np.empty((n_,))
        for i in range(n_ - 1, -1, -1):
            a[i] = rho[i] * dotproduct(s[-1], q)
            q -= a[i] * y[i]
        H = Matrix(dJ_.function_space()).scale(
            dotproduct(s[-1], y[-1]) / dotproduct(y[-1], y[-1])
        )
        P = H.multiply(q)
        for i in range(n_):
            b = rho[i] * dotproduct(y[-1], P)
            P += s[i] * (a[i] - b)
    P = Matrix(dJ_.function_space()).scale(-1).multiply(P)

    # Take a step downhill
    lr = line_search(forward_run, m, u_, P, J_, dJ_, params)
    u = u_ + lr * P

    J, u = forward_run(m, u, **params.model_options)
    dJ = fd_adj.compute_gradient(J, fd_adj.Control(u))
    sk = u.copy(deepcopy=True)
    sk -= u_
    yk = dJ.copy(deepcopy=True)
    yk -= dJ_

    # Update three lists
    if dotproduct(sk, yk) > 0:
        s.append(sk)
        y.append(yk)
        rho.append(1.0 / dotproduct(sk, yk))
    if len(s) > n:
        rho.pop(0)
        s.pop(0)
        y.pop(0)

    yield {"lr": lr, "u+": u, "rho": rho, "s": s, "y": y}


def _bfgs(it, forward_run, m, params, u, u_, dJ_, B):
    """
    Take one BFGS iteration.

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
    :arg B: the previous Hessian approximation
    """
    J, u = forward_run(m, u, **params.model_options)
    dJ = fd_adj.compute_gradient(J, fd_adj.Control(u))
    B = B or Matrix(u.function_space())
    yield {"J": J, "u": u.copy(deepcopy=True), "dJ": dJ.copy(deepcopy=True)}

    # Compute the search direction and step length and take a step
    P = B.copy().scale(-1).solve(dJ)
    lr = line_search(forward_run, m, u, P, J, dJ, params)
    u += lr * P

    # Update the approximated inverted Hessian matrix
    if u_ is not None and dJ_ is not None:
        dJ_ = params.transfer_fn(dJ_, dJ.function_space())
        u_ = params.transfer_fn(u_, u.function_space())

        s = u.copy(deepcopy=True)
        s -= u_
        y = dJ.copy(deepcopy=True)
        y -= dJ_

        y_star_s = dotproduct(y, s)
        y_y_star = OuterProductMatrix(y, y)
        second_term = y_y_star.scale(1 / y_star_s)

        Bs = B.multiply(s)
        sBs = dotproduct(s, Bs)
        sB = B.multiply(s, side="left")
        BssB = OuterProductMatrix(Bs, sB)
        third_term = BssB.scale(1 / sBs)

        B.add(second_term)
        B.subtract(third_term)
    yield {"lr": lr, "u+": u, "B": B}


def _newton(it, forward_run, m, params, u):
    """
    Take one full Newton iteration.

    :arg it: the current iteration number
    :arg forward_run: a Python function that
        implements the forward model and
        computes the objective functional
    :arg m: the current mesh
    :kwarg params: :class:`OptAdaptParameters` instance
        containing parameters for the optimisation and
        adaptation routines
    :arg u: the current control value
    """

    # Annotate the tape and compute the gradient
    J, u = forward_run(m, u, **params.model_options)
    dJ = fd_adj.compute_gradient(J, fd_adj.Control(u))
    ddJ = compute_full_hessian(J, fd_adj.Control(u))
    yield {"J": J, "u": u.copy(deepcopy=True), "dJ": dJ.copy(deepcopy=True)}

    try:
        P = ddJ.copy().scale(-1).solve(dJ)
    except np.linalg.LinAlgError:
        raise Exception("Hessian is singular, please try the other methods")

    # Take a step downhill
    lr = line_search(forward_run, m, u, P, J, dJ, params)
    u += lr * P
    yield {"lr": lr, "u+": u, "ddJ": ddJ}


_implemented_methods = {
    "gradient_descent": {
        "func": _gradient_descent,
        "order": 1,
        "type": "gradient_based",
    },
    "bfgs": {"func": _bfgs, "order": 2, "type": "quasi_newton"},
    "lbfgs": {"func": _bfgs, "order": 2, "type": "quasi_newton"},
    "newton": {"func": _newton, "order": 2, "type": "newton"},
    "adam": {"func": _adam, "order": 1, "type": "gradient_based"},
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
    **kwargs,
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
    tape = fd_adj.get_working_tape()
    tape.clear_tape()
    u_plus = initial_control_fn(mesh)
    Rspace = u_plus.ufl_element().family() == "Real"
    method = kwargs.get("method", "newton" if Rspace else "gradient_descent")
    method = method.lower()
    try:
        step, order, _ = _implemented_methods[method].values()
    except KeyError:
        raise ValueError(f"Method '{method}' not recognised")
    params = kwargs.get("params", OptAdaptParameters(method))
    op = kwargs.get("op", OptimisationProgress())
    dJ_init = None
    target = params.target_base
    B = None
    mesh_adaptation = adapt_fn != identity_mesh

    # Enter the optimisation loop
    nc = mesh.num_cells()
    adaptor = adapt_fn
    for it in range(1, params.maxiter + 1):
        term_msg = f"Terminated after {it} iterations due to "
        u_ = None if it == 1 else op.m_progress[-1]
        dJ_ = None if it == 1 else op.dJ_progress[-1]

        if step == _gradient_descent:
            args = [u_plus, u_, dJ_]
        elif step == _adam:
            if it == 1:
                a_ = None
                b_ = None
            args = [u_plus, a_, b_]
        elif step == _bfgs:
            if it == 1:
                B = None
            args = [u_plus, u_, dJ_, B]
        elif step == _lbfgs:
            if it == 1:
                rho = None
                s = None
                y = None
            args = [u_plus, rho, s, y]
        elif step == _newton:
            args = [u_plus]
        else:
            raise NotImplementedError(f"Method {method} unavailable")

        # Take a step
        cpu_timestamp = perf_counter()
        out = {}
        for o in step(it, forward_run, mesh, params, *args):
            out.update(o)
        J, u, dJ = out["J"], out["u"], out["dJ"]
        lr, u_plus = out["lr"], out["u+"]
        if step == _adam:
            a_, b_ = out["a-"], out["b-"]
        elif step == _bfgs:
            B = out["B"]
        elif step == _lbfgs:
            rho, s, y = out["rho"], out["s"], out["y"]
        elif step == _newton:
            B = out["ddJ"]

        # Print to screen, if requested
        t = perf_counter() - cpu_timestamp
        if params.disp > 0:
            g = dJ.dat.data[0] if Rspace else fd.norm(dJ)
            msgs = [f"{it:3d}:  J = {J:9.4e}"]
            if Rspace:
                msgs.append(f"m = {u_plus.dat.data[0]:9.4e}")
            if Rspace:
                msgs.append(f"dJdm = {g:11.4e}")
            else:
                msgs.append(f"||dJdm|| = {g:9.4e}")
            msgs.append(f"step length = {lr:9.4e}")
            msgs.append(f"#elements = {nc:5d}")
            msgs.append(f"time = {t:.2f}s")
            pprint(",  ".join(msgs))

        # Stash progress
        op.t_progress.append(t)
        op.J_progress.append(J)
        op.m_progress.append(u)
        op.dJ_progress.append(dJ)
        if B is not None:
            op.ddJ_progress.append(B)
        op.nc_progress.append(nc)
        op.mesh_progress.append(fd.Mesh(mesh.coordinates.copy(deepcopy=True)))

        # If lr is too small, the difference u-u_ will be 0, and it may cause error
        if params.check_lr:
            if lr < params.lr_lowerbound:
                raise fd.ConvergenceError(
                    term_msg + "fail, because control variable didn't move"
                )

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
        # For some situation, convergence should be true, but don't satisfy the above condition
        elif np.abs(J - op.J_progress[-2]) < params.qoi_rtol * op.J_progress[-2]:
            if fd.norm(dJ) / dJ_init < params.gtol_loose:
                if params.disp > 0:
                    pprint(term_msg + "gtol convergence (second situation)")
                break

        # Check for reaching maximum iteration count
        if it == params.maxiter:
            raise fd.ConvergenceError(term_msg + "reaching maxiter")

        if it > 2 and mesh_adaptation:
            J_ = op.J_progress[-2]
            if nc < nc_:
                mesh_adaptation = False
                adaptor = identity_mesh
                if params.disp > 1:
                    pprint("NOTE: turning adaptation off due to mesh converged")
                continue
            elif np.abs(J - J_) < params.qoi_rtol * np.abs(J_):
                mesh_adaptation = False
                adaptor = identity_mesh
                if params.disp > 1:
                    pprint("NOTE: turning adaptation off due to qoi_rtol convergence")
                continue
            elif np.abs(nc - nc_) < params.element_rtol * nc_:
                mesh_adaptation = False
                adaptor = identity_mesh
                if params.disp > 1:
                    pprint(
                        "NOTE: turning adaptation off due to element_rtol convergence"
                    )
                continue
            else:
                adaptor = adapt_fn

        if mesh_adaptation:
            # Ramp up the target complexity
            target = min(target + params.target_inc, params.target_max)
            # Adapt the mesh
            mesh = adaptor(mesh, target=target, control=u_plus)

            nc_ = nc
            nc = mesh.num_cells()

        # Clean up
        tape.clear_tape()
    return u_plus
