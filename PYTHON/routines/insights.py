import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from casadi import MX, Function, jacobian, vertcat
from do_mpc.controller import MPC
from do_mpc.model import Model
from do_mpc.simulator import Simulator

"""Implements useful helper functions to calculate and visualize internal mpc parmeters, intermediate variables such as hessian and jacobian matrices, conditioning numbers etc.."""


def calculcate_surrogate_jacobian(surrogate: torch.nn.Module, input: np.ndarray) -> np.ndarray:
    """Calculates the jacobian of the neural network with respect to its input features. And evaluates it with the given input. It supports batching."""
    input = torch.tensor(input, requires_grad=True, dtype=torch.float32)
    jac = torch.stack([torch.autograd.functional.jacobian(surrogate, input[i]) for i in range(input.shape[0])])
    jac = np.array(jac)
    return jac


def plot_rhs_jac(
    model: Model,
    states: np.ndarray,
    inputs: np.ndarray,
    tvps: np.ndarray,
    params: np.ndarray,
    h: Optional[float] = 1e-6,
    save_path: Optional[str] = None,
):
    x = vertcat(model.x)
    rhs_fun = Function("rhs", [x, vertcat(model.u), vertcat(model.tvp), vertcat(model.p)], [model._rhs])
    jac_rhs = jacobian(model._rhs, x)
    jac_rhs_fun = Function("jac_rhs", [x, vertcat(model.u), vertcat(model.tvp), vertcat(model.p)], [jac_rhs])
    jac_rhs_fun = jac_rhs_fun(states, inputs, tvps, params)

    jac_rhs_approx = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        e = np.zeros(x.shape[0])
        e[i] = h
        diff_i = (rhs_fun(states + e, inputs, tvps, params) - rhs_fun(states, inputs, tvps, params)) / h
        jac_rhs_approx[:, i] = diff_i.full().flatten()

    print(jac_rhs_fun)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    map_jac = ax[0].imshow(jac_rhs_fun)
    map_jac_approx = ax[1].imshow(jac_rhs_approx)
    cbar = fig.colorbar(map_jac, ax=ax[1])
    cbar.ax.set_ylabel("partial derivative value", rotation=-90)
    ax[0].set_title(f"Analytic Jacobian")  #  (cond {cond:.2e})")
    ax[1].set_title(f"Finite Difference Jacobian")  #  (cond {cond_approx:.2e})")

    ax[0].set_xlabel(r"$\partial x_i$")
    ax[1].set_xlabel(r"$\partial x_i$")
    ax[0].set_ylabel(r"$\partial f_j$")
    plt.tight_layout()

    if save_path is not None:
        final_save_path = os.path.join(save_path, "rhs_jacobian.pdf")
        plt.savefig(final_save_path)

    exit()


def plot_mpc_jacobi(mpc: MPC, finite_diff_step: float = 1e-8):
    """
    Calculate and visualize the Jacobian of the MPC model constraints rolled out
    over the prediction horizon. Compares the analytic Jacobian against a finite
    difference approximation and plots both as heatmaps.
    Args:
        mpc (MPC): The MPC controller object.
        finite_diff_step (float): Step size for finite difference approximation.
    Returns:
        fig (matplotlib.figure.Figure): Figure object of the plots.
        ax (numpy.ndarray): Array of Axes objects containing the subplots.
    """

    # Collect optimization variables and parameters
    x_opt = vertcat(mpc._opt_x)
    p_opt = vertcat(mpc._opt_p)
    # Initial guesses
    x0_opt_num = vertcat(mpc.opt_x_num)
    p0_opt_num = vertcat(mpc.opt_p_num)
    print("Total number of optimization variables:", x_opt.shape)
    print("Initial guess for the optimization variables:", x0_opt_num)
    print("Optimization parameters:", p0_opt_num)
    # Constraints of the rolled-out MPC
    f_rolled_out = mpc.nlp_cons
    print("Total number of equality constraints:", f_rolled_out.shape)
    # Constraint function
    f_rolled_out_f = Function("rhs_rolled_out", [x_opt, p_opt], [f_rolled_out])
    # Finite difference Jacobian approximation
    jac_approx = np.zeros((f_rolled_out.shape[0], x_opt.shape[0]))
    for i in range(x_opt.shape[0]):
        e = np.zeros(x_opt.shape[0])
        e[i] = finite_diff_step
        diff_i = (f_rolled_out_f(x0_opt_num + e, p0_opt_num) - f_rolled_out_f(x0_opt_num, p0_opt_num)) / finite_diff_step
        jac_approx[:, i] = diff_i.full().flatten()
    # Analytic Jacobian
    jac = jacobian(f_rolled_out, x_opt)
    jac_f = Function("Jacobi_Matrix", [x_opt, p_opt], [jac])
    jac_f0 = jac_f(x0_opt_num, p0_opt_num)
    jac_f0 = np.array(jac_f0)
    # Conditioning numbers
    # cond = np.linalg.cond(jac_f0)
    # cond_approx = np.linalg.cond(jac_approx)
    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    map_jac = ax[0].imshow(jac_f0)
    map_jac_approx = ax[1].imshow(jac_approx)
    cbar = fig.colorbar(map_jac, ax=ax[0])
    cbar.ax.set_ylabel("gradient value", rotation=-90)
    ax[0].set_title(f"Analytic Jacobian")  #  (cond {cond:.2e})")
    ax[1].set_title(f"Finite Difference Jacobian")  #  (cond {cond_approx:.2e})")
    plt.show()


def plot_objective_gradient(mpc: MPC):
    """Plots the gradient of the objective function."""
    import matplotlib.pyplot as plt

    x_opt = vertcat(mpc._opt_x)
    p_opt = vertcat(mpc._opt_p)
    x0_opt_num = vertcat(mpc.opt_x_num)
    p0_opt_num = vertcat(mpc.opt_p_num)
    f_objective = mpc._nlp_obj
    grad_f = jacobian(f_objective, x_opt)
    grad_f = Function("gradient_f", [x_opt, p_opt], [grad_f])
    grad_f = grad_f(x0_opt_num, p0_opt_num).full().flatten()
    for i, df_i in enumerate(grad_f):
        if np.abs(df_i) >= 1e-8:
            print("gradient with respect to ", i, df_i)

    max_grad = np.linalg.norm(grad_f, ord=np.inf).round(2)
    # fig, ax = plt.subplots(1, figsize=(6, 6))
    # heatmap = ax.imshow(grad_f.reshape(1, -1))
    # cbar = fig.colorbar(heatmap, ax=ax)
    # cbar.ax.set_ylabel("gradient value", rotation=-90)
    # ax.set_title(r"objective function gradient $||\nabla_{x,u} f(x,u)||_\infty$ = " + str(max_grad))
    # plt.show()


# def plot_objective_function():
#     from mpl_toolkits.mplot3d import Axes3D

#     current_states, past_states = mpc_model.x["x"], mpc_model.x["past_states"]
#     current_inputs, past_inputs = mpc_model.u["u"], mpc_model.x["past_inputs"]
#     current_tvps, past_tvps = mpc_model.tvp["tvp"], mpc_model.x["past_tvps"]
#     do_mpc_vector = vertcat(current_states, past_states, past_inputs, past_tvps)
#     current_inputs_tvps = vertcat(current_inputs, current_tvps)
#     objective_function = Function("objective", [do_mpc_vector, current_inputs_tvps], [selectivity])
#     non_doe_vector = structurizer.to_dompc_vector(data)
#     T_in_range = np.linspace(550, 650, 100)
#     T_c4_range = np.linspace(550, 650, 100)
#     Tin_mesh, Tc4_mesh = np.meshgrid(T_in_range, T_c4_range)
#     Tin = MX.sym("Tin", 1)
#     Tc4 = MX.sym("Tc4", 1)
#     new_input = vertcat(Tin, DM([600, 610, 590]), Tc4, DM(0.3))
#     objective_eval = objective_function(non_doe_vector, new_input)
#     objective_function = Function("objective_tind", [Tin, Tc4], [objective_eval])
#     result = np.zeros((non_doe_vector.shape[-1], 100, 100))
#     for i in range(100):
#         for j in range(100):
#             objective_result = objective_function(Tin_mesh[i, j], Tc4_mesh[i, j])
#             result[:, i, j] = objective_result

#     colors = ["#a32cff", "#FDFCBB"]
#     bg_color = "#121212"

#     def set_fig_preferences(fig, ax):
#         fig.set_facecolor(bg_color)
#         ax.set_facecolor(bg_color)
#         ax.grid(False)
#         ax.tick_params(
#             axis="both",  # Betrifft beide Achsen
#             which="both",  # Betrifft sowohl Haupt- als auch Neben-Ticks
#             bottom=False,
#             top=False,
#             left=False,
#             right=False,
#             labelbottom=False,
#             labelleft=False,
#         )
#         ax.spines["top"].set_color(bg_color)
#         ax.spines["right"].set_color(bg_color)
#         ax.spines["left"].set_color("white")
#         ax.spines["bottom"].set_color("white")
#         ax.spines["left"].set_linewidth(4)  # Linke Achse dicker machen
#         ax.spines["bottom"].set_linewidth(4)  # Untere Achse dicker machen
#         return fig, ax

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_surface(Tin_mesh, Tc4_mesh, result[0], cmap="magma")

#     set_fig_preferences(fig, ax)

#     plt.show()


# KONDITIONSZAHL
# input_vector = structurizer.stack_data(data)
# jac = calculcate_surrogate_jacobian(torch_models["nominal"], input_vector)
# cond = np.linalg.cond(jac)
