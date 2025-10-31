from typing import Callable, List, Tuple, Union

import numpy as np
from casadi import *
from do_mpc.model import Model


def add_implicit_layer(super_model, n_measurements: int, initial_guess: Union[MX, np.ndarray], output_expressions: Union[MX, List[MX]]) -> Union[MX, List[MX]]:
    """Adds the optimization layer to the output expressions of the neural network to enforce physicality constraints."""
    if isinstance(output_expressions, MX):
        output_expressions = [output_expressions]
    A, b = super_model.get_physicality_constraints(num_stacks=n_measurements)
    x_in = super_model.get_bc_for_all_measurements(n_measurements=n_measurements)
    scales = super_model.get_scaling_factors(n_measurements=n_measurements)
    opt_layer = make_implicit_layer(A=A, z_ref=x_in, scales=scales)
    expressions_with_opt = []
    for expression in output_expressions:

        expressions_with_opt.append(opt_layer(initial_guess / scales, expression))
    return expressions_with_opt if len(expressions_with_opt) > 1 else expressions_with_opt[0]


def make_explicit_layer(A: np.ndarray, boundary_cond: np.ndarray, b: np.ndarray = None) -> Function:
    r"""Creates an explicit optimization layer that is the analytic solution of the following optimization task.
      .. math::
        \begin{aligned}
        & \underset{z_s}{\text{minimize}}
        & & \| z_s - x_s \|_2^2 \\
        & \text{subject to}
        & & A (z_s - z_{ref,s}) = 0
        \end{aligned}

    where:
    - `x_s = x / scales` is the scaled input vector.
    - `z_{ref,s} = boundary_cond / scales` is the scaled reference vector
    - The final result is rescaled: `z* = z_s * scales`.
    """
    if boundary_cond.ndim == 2:
        boundary_cond = boundary_cond.squeeze()

    if b is None:
        b = np.zeros(A.shape[0])
    U = np.concatenate([2 * np.eye(A.shape[1]), A.T], axis=1)
    L = np.concatenate([A, np.zeros((A.shape[0], A.shape[0]))], axis=1)
    K_inv = np.linalg.inv(np.concatenate([U, L], axis=0))
    x = MX.sym("x", A.shape[1])
    rhs_vector = vertcat(2 * x, A @ boundary_cond + b)
    sol = K_inv @ rhs_vector
    sol = sol[: A.shape[1]]  # cutting off the dual variables
    return Function("z", [x], [sol])


def make_implicit_layer(A: np.ndarray, z_ref: np.ndarray, scales: np.ndarray = None, tol: float = 1e-5, error_on_fail: bool = False, max_iter: int = 50) -> Function:
    r"""This function constructs a differentiable "optimization layer" that can be used, for example, at the end of a neural network. It takes an input vector `x`
    (e.g., the network's output) and finds the vector `z*` that is closest to `x` while satisfying a set of linear constraints.
    The layer solves the following quadratic program (QP) for the scaled state `z_s`:
    
    .. math::
        \begin{aligned}
        & \underset{z_s}{\text{minimize}}
        & & \| z_s - x_s \|_2^2 \\
        & \text{subject to}
        & & A (z_s - z_{ref,s}) = 0
        \end{aligned}

    where:
    - `x_s = x / scales` is the scaled input vector.
    - `z_{ref,s} = z_ref / scales` is the scaled reference vector.
    - The final result is rescaled: `z* = z_s * scales`.

    The problem is solved by forming the Karush-Kuhn-Tucker (KKT) conditions
    and using a Newton-Raphson root-finding algorithm.
    Args:
        A (np.ndarray): The matrix A that defines the linear constraints.
        b (np.ndarray): The vector b that defines the offset of the linear constraints. It set to zero for default.
        z_ref (casadi.MX): The inlet values of the states to calculate the concentration difference.
        scales (np.ndarray): A reference state vector to scale the reference states.
        tol (float): The tolerance of the newton root finding algorithm
    Returns:
        Function: Casadi function that takes the output of the neural network and the initial starting point for the rootfinding situation
        and returns the optimal solution z* that is close to the network output and satisfies the constraint A @ (z - z_ref) = 0.
        [x_initial, x_network] -> [x*]
    """
    assert z_ref.shape[0] == A.shape[1], f"The number of states in the inlet vector {z_ref.shape} must be equal to the state coefficients of the matrix A {A.shape}"
    z = MX.sym("z", A.shape[1])
    x = MX.sym("x", A.shape[1])
    if scales is None:
        scales = np.ones(A.shape[1])
    f = norm_2(z - x / scales)
    h = A @ (z - z_ref / scales)
    nu = MX.sym("nu", h.shape)
    G = horzcat(jacobian(f, z) + nu.T @ jacobian(h, z), h.T).T  # KKT matrix G
    z_initial = MX.sym("z0", A.shape[1], 1)
    newton = rootfinder("newton", "newton", {"x": vertcat(z, nu), "g": G, "p": x}, {"error_on_fail": error_on_fail, "abstol": tol, "max_iter": max_iter})
    nu0 = np.random.rand(nu.shape[0], 1)
    sol = newton(vertcat(z_initial, nu0), x)
    z_star = sol[: z.shape[0]] * scales  # Cut off the dual variables and rescale to original scale
    return Function("z", [z_initial, x], [z_star])


def get_narx_input_shift_rhs(
    stack_function: Callable,
    state_keys: List[str],
    n_measurements: int,
    input_keys: List[str],
    time_horizon: int,
    model: Model,
    tvp_keys: List[str] = [],
) -> Tuple[MX, Model]:
    """Creates the input vector for an autoregressive network model with the current input on top to by pass the encoder.
    It sets the states and inputs accordingly to create the full input vector.
    It also automatically sets the trivial rhs`s for the past states and inputs.
    Returns the fully stacked input vector as well as the vector of the current states.
    """
    # set the current state vector
    n_total_states = len(state_keys) * n_measurements
    current_states = []
    for state_key in state_keys:
        current_states.append(model.set_variable("_x", state_key, shape=(n_measurements, 1)))
    current_states = vertcat(*current_states)

    n_inputs = len(input_keys)
    current_inputs = []
    for input_key in input_keys:
        current_inputs.append(model.set_variable("_u", input_key))
    current_inputs = vertcat(*current_inputs)

    past_states = model.set_variable("_x", "past_states", shape=(n_total_states * (time_horizon - 1), 1))
    past_inputs = model.set_variable("_x", "past_inputs", shape=(n_inputs * (time_horizon - 1), 1))
    if len(tvp_keys) > 0:
        current_tvps = []
        for tvp_key in tvp_keys:
            current_tvps.append(model.set_variable("_tvp", tvp_key, shape=(len(tvp_keys), 1)))
            past_tvps = model.set_variable("_x", "past_tvps", shape=(len(tvp_keys) * (time_horizon - 1), 1))
        current_tvps = vertcat(*current_tvps)
    else:
        current_tvps = DM([])
        past_tvps = DM([])

    # set the rhs for all past states and inputs
    old_state_shift = vertcat(current_states, *vertsplit(past_states)[:-n_total_states])  # drop all the states at the last time point
    old_input_shift = vertcat(current_inputs, *vertsplit(past_inputs)[:-n_inputs])
    old_tvp_shift = vertcat(current_tvps, *vertsplit(past_tvps)[: -len(tvp_keys)])
    model.set_rhs(f"past_states", old_state_shift)
    model.set_rhs(f"past_inputs", old_input_shift)
    model.set_rhs(f"past_tvps", old_tvp_shift)

    # stack current states and inputs and all past states and inputs on top of each other
    X = stack_function(current_states, past_states, current_inputs, past_inputs, current_tvps, past_tvps)
    return X, model
