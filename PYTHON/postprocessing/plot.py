import json
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import l4casadi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from do_mpc.data import Data, MPCData
from do_mpc.graphics import Graphics
from matplotlib.animation import FuncAnimation
from routines.setup_routines import SurrogateTypes
from routines.utils import _load_single_json, load_json_results_for_all

from .performance_metrics import calculate_state_mse
from .plotting_helpers import *
from .Visualizer import Visualizer

set_mpt_settings()


# ------- Plots for Neural Network Training --------


def plot_val_loss(
    training_history_dict: Dict,
    hidden_units_list: List[List[int]],
    save_dir: str,
    plot_cfg: Optional[Dict] = None,
    save_cfg: Optional[Dict] = None,
):
    save_cfg = save_cfg or {}
    plot_cfg = plot_cfg or {}

    fig, ax = plt.subplots(1)
    n_data_rows = len(hidden_units_list)
    n_types = len(training_history_dict)
    colors = make_colors(n_colors=len(hidden_units_list))
    line_cycler = make_line_cycler(n_types)
    color_cycler = make_color_cycler(n_colors=n_data_rows)
    ax.set_prop_cycle(line_cycler * color_cycler)

    for surrogate_type, history_list in training_history_dict.items():
        for hidden_units, color in zip(hidden_units_list, colors):
            for history_dict in history_list:
                if hidden_units == history_dict.get("units"):
                    ax.plot(
                        history_dict["epoch"],
                        np.log10(history_dict["val_loss"]),
                        label=f"{latex_notation_map[surrogate_type]["general"]}" + r" $n_\mathrm{hidden}$ = " + f"{hidden_units}",
                        color=color,
                    )
                    break

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, plot_cfg.get("legend_y_pos", 1.2)), ncol=n_data_rows)
    ax.set_xlabel("epoch / -")
    ax.set_ylabel(r"$\mathrm{log}_{10} \; \mathcal{L}_\mathrm{MSE, val}$ / -")

    plt.subplots_adjust(right=0.95, left=0.1, top=0.88, bottom=0.15)
    if save_cfg.get("show_fig", False):
        plt.show()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "val_loss.pdf"))
    plt.close()


# ------- Plots for System Trajectories ----------


def plot_open_loop_state(
    sim_cfg: Dict[str, Any],
    surrogate_results: Dict[str, Dict],
    test_data: np.ndarray,
    state: str,
    surrogate_type: str,
    positions: List[int],
    meta_data: Optional[Dict[str, Any]] = None,
    plot_cfg: Optional[Dict[str, Any]] = None,
    save_cfg: Optional[Dict[str, Any]] = None,
):
    """
    Plots open-loop simulation results against test data for a single state.

    This function visualizes surrogate model trajectories and ground truth test
    trajectories, calculates the Mean Squared Error (MSE), and annotates the
    plot with simulation metadata.

    Args:
        sim_cfg: Simulation configuration dictionary.
        surrogate_results: A dictionary where keys are model identifiers (nominal, upper, lower) and values
                           are dictionaries containing simulation data arrays
                           ('_x', '_aux', '_time').
        test_data: Array of test measurement data (ground truth).
        state: The name of the state variable to plot (e.g., 'x', 'phi').
        surrogate_type: Type of surrogate model ('rom', 'narx', etc.).
        positions: A list of indices for the state vector to plot.
        meta_data: Metadata from the simulation for annotations.
        plot_cfg: Dictionary with plotting configurations, such as 't_steps',
                  'annotations', 'legend_y_pos', 'ylabel_size', etc.
        save_cfg: Dictionary with saving configurations, such as 'save_dir',
                  'export_name', 'save_meta', 'show_fig'.

    Returns:
        The matplotlib Figure object.
    """
    # Unpack configurations with sensible defaults
    plot_cfg = plot_cfg or {}
    save_cfg = save_cfg or {}
    plt.close("all")

    t_steps = plot_cfg.get("t_steps", -1)
    annotations = plot_cfg.get("annotations", [])
    test_data_color = plot_cfg.get("test_data_color", make_colors(4, alpha=0.5)[3])
    plot_kwargs = plot_cfg.get("plot_kwargs", {"data": {}, "surrogate": {}})

    show_fig = save_cfg.get("show_fig", True)
    save_dir = save_cfg.get("save_dir")

    # --- Data Preparation ---
    structurizer = DataStructurizer(
        n_measurements=sim_cfg["narx"]["n_measurements"],
        n_initial_measurements=sim_cfg["simulation"]["N_finite_diff"],
        time_horizon=sim_cfg["narx"]["time_horizon"],
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        tvp_keys=sim_cfg["tvps"]["keys"],
        aux_keys=sim_cfg["aux"]["keys"],
    )
    test_data = structurizer.reduce_measurements(test_data, sim_cfg["simulation"]["N_finite_diff"])
    test_data_for_state = structurizer.get_states_from_data(test_data, n_measurements=sim_cfg["narx"]["n_measurements"], state=state)

    warm_up_steps = meta_data.get("warm_up_steps", structurizer.time_horizon) if meta_data else structurizer.time_horizon
    var_key = "_y" if surrogate_type == SurrogateTypes.Rom.value else "_x"

    # --- MSE Calculation ---
    mse = None
    if surrogate_results:
        # Use the first result to determine the trajectory for MSE calculation
        first_result = next(iter(surrogate_results.values()))
        surrogate_trajectory = structurizer.get_states_from_data(data=first_result[var_key], state=state)[:t_steps]
        test_trajectory = test_data_for_state[..., warm_up_steps : t_steps + warm_up_steps, :]
        if test_trajectory.ndim == 3:
            # average over all trajectories to calculate the mse
            test_trajectory = test_trajectory.mean(axis=0)
        state_idx = sim_cfg["states"].get("keys").index(state)
        mse = calculate_state_mse(test_trajectory, surrogate_trajectory, state_scale=sim_cfg["states"]["scales"][state_idx])

    # --- Plotting ---
    vis = Visualizer(sim_cfg, cmap="viridis")
    fig, axes = vis.make_axes_for_all_vars(n_plots=len(positions), cbar=False)

    time_vector = np.arange(-warm_up_steps, t_steps if t_steps != -1 else test_data_for_state.shape[1] - warm_up_steps) * sim_cfg["simulation"]["t_step"]

    if test_data_for_state.ndim < 3:
        test_data_for_state = np.expand_dims(test_data_for_state, axis=0)  # add batch dimension
    test_data_for_state = np.swapaxes(test_data_for_state, 0, 1)  # swap time and batch dimension for matplotlib to plot all batches at once

    for i, (ax, pos) in enumerate(zip(axes, positions)):
        # Plot test data
        test_data_to_plot = test_data_for_state[: len(time_vector), :, pos]  # plot all needed time steps, for all batches, at given position
        ax.plot(time_vector, test_data_to_plot, color=test_data_color, linestyle="-", label="Test Data", **plot_kwargs.get("data", {}))

        # Plot surrogate results
        if surrogate_results:
            ax.set_prop_cycle(make_line_cycler(n_styles=len(surrogate_results)) + make_color_cycler(n_colors=len(surrogate_results)))
            for name, data in surrogate_results.items():
                state_data = structurizer.get_states_from_data(data=data[var_key], state=state)
                label = latex_notation_map[surrogate_type]["general"]
                ax.plot(data["_time"][:t_steps], state_data[:t_steps, pos], label=label, **plot_kwargs.get("surrogate", {}))

        ax.set_ylabel(sim_cfg["plotting"]["zlabels"][state][pos], size=plot_cfg.get("ylabel_size", 20))
        ax.set_ylim(lower_ylims[state[0]], upper_ylims[state[0]])

    axes[-1].set_xlabel(latex_notation_map["time"])

    # --- Annotations and Legend ---
    if annotations and meta_data:
        complete_meta_data = {**meta_data, "mse": f"{mse.mean():.6f}"} if mse is not None else meta_data
        axes[-1].annotate(complete_meta_data, xy=(0.05, 0.85), xycoords="axes fraction")

    if surrogate_results:
        format_legend(ax=axes[0], plot_cfg=plot_cfg)

    plt.subplots_adjust(right=0.92, left=0.12, top=0.88, bottom=0.15)

    # --- Saving and Display ---
    if save_dir:
        default_name = f"plot_{surrogate_type}_{state}"
        export_name = save_cfg.get("export_name", default_name)
        plt.savefig(os.path.join(save_dir, f"{export_name}.pdf"))
        if save_cfg.get("save_meta", False) and "complete_meta_data" in locals():
            save_meta_data(save_dir, export_name, complete_meta_data)

    if show_fig:
        plt.show()
    plt.close(fig=fig)


def plot_loop(
    sim_cfg: Dict,
    data: Union[Data, MPCData],
    surrogate_type: Type[SurrogateTypes],
    n_measurements: int,
    annotations: Optional[List[str]] = None,
    animate: Optional[bool] = False,
    show_fig: Optional[bool] = True,
):
    """
    Plot and animate closed-loop simulation results from an MPC (Model Predictive Control) run.
    This function loads simulation results, prepares a multi-panel plot for states, inputs,
    time-varying parameters, and auxiliary variables, and visualizes both predictions and
    actual results. Constraint lines and proper labels are added automatically. The output
    can be viewed as a static plot (last time step) or as an animated time series.
    Parameters
    ----------
    file_path : str
        Path (relative to "results/") of the stored simulation results file (.pkl format).
    surrogate_type : str
        Type of surrogate model used. If "rom", states are read from auxiliary variables
        ("_aux"); otherwise from standard state variables ("_x").
    n_measurements : int
        Number of measurements used for creating distinct color cyclers for state plots.
    Workflow
    --------
    1. Load results and extract simulator data.
    2. Initialize plotting axes for all variables defined in the simulation configuration.
    3. Add lines for:
       - State trajectories (from "_x" or "_y" depending on surrogate type).
       - Input trajectories ("_u").
       - Time-varying parameters ("_tvp").
       - Auxiliary variables (e.g., S_mpc, X_mpc).
    4. Add visual constraint lines (e.g., Tmax, Xmin).
    5. Synchronize axis labels and limits across subplots.
    6. Define an update function to refresh plots at each simulation time step.
    7. Display final results and create an animated visualization.
    """
    vis = Visualizer(sim_cfg, cmap="viridis")
    time = data["_time"]
    # Change the Data Object from the NARX Simulation to separate the dummy states "x" into the original states
    g = Graphics(data)
    plot_keys = sim_cfg["plotting"]["ylabels"].keys()
    fig, axes = vis.make_axes_for_all_vars(len(plot_keys))
    input_index = len(sim_cfg["states"]["keys"])
    tvp_index = input_index + 1
    aux_index = tvp_index + 1
    # adding custom color cyclers seperately for states, inputs, tvps and aux
    input_color_cycler = make_color_cycler(n_colors=len(sim_cfg["inputs"]["all_keys"]))
    # adding all line objects
    state_color_cycler = make_color_cycler(n_colors=n_measurements)
    add_color_cyclers(axes, state_color_cycler, input_color_cycler, input_index)
    for i, state_key in enumerate(sim_cfg["states"]["keys"]):
        vartype = "_y" if surrogate_type == SurrogateTypes.Rom else "_x"
        g.add_line(vartype, state_key, axis=axes[i])
    for input_key in data.model["_u"].keys():
        g.add_line("_u", input_key, axis=axes[input_index])
    g.add_line("_tvp", "u", axis=axes[tvp_index])

    g.add_line("_aux", "S", axis=axes[aux_index], label=r"$S_{mpc}$")
    g.add_line("_aux", "X", axis=axes[aux_index], label=r"$X_{mpc}$")
    add_constraint_line(axes=axes, axes_index=input_index - 1, value=630 / sim_cfg["scales"].get("T"), kwargs={"label": r"$T_{max}$"})
    add_constraint_line(axes, axes_index=aux_index, value=0.6, kwargs={"label": r"$X_{min}$"})
    # axes[-1].legend()
    set_labels(sim_cfg=sim_cfg, latex_notation_map=latex_notation_map, axes=axes)
    vis.sync_ylims(axes, plot_keys=plot_keys)

    # annotation = make_meta_data_annotation(data=data, include=annotations)
    # axes[-1].annotate(annotation, xy=(0, -1.5), xycoords="axes fraction", size=12)
    # fig.text(0.5, -0.05, annotation, ha="center", va="center", size="12")

    def update(frame):
        if isinstance(data, MPCData):
            g.plot_predictions(frame)
        g.plot_results(frame)
        g.reset_axes()

    # plt.savefig(file_path.replace(".pkl", ".pdf"))
    fig.align_ylabels()
    if animate:
        animation = FuncAnimation(fig, update, frames=range(time.shape[0]), repeat=False)
    # animation.save("rom_mpc.gif", writer="imagemagick", fps=2, dpi=160)
    update(-1)
    if show_fig:
        plt.show()
    return fig


def plot_random_trajectories(
    sim_cfg: Dict,
    n_trajectories: int,
    result_dir: str,
    save_to_dir: str,
    test_data: np.ndarray,
    filter_test_trajectories: bool = True,
    states: List[str] = ["T", "chi_E"],
    plot_cfg: Optional[Dict] = None,
    save_cfg: Optional[Dict] = None,
):

    if plot_cfg is None:
        plot_cfg = {}
    if save_cfg is None:
        save_cfg = {}

    for entry in os.scandir(result_dir):
        sub_dir_name = entry.name
        sub_dir_path = entry.path
        output_sub_dir = os.path.join(save_to_dir, sub_dir_name)

        if entry.is_dir():
            if list(os.scandir(entry.path))[0].is_dir():  # dir contains sub dirs
                plot_random_trajectories(
                    sim_cfg=sim_cfg,
                    n_trajectories=n_trajectories,
                    result_dir=sub_dir_path,
                    save_to_dir=output_sub_dir,
                    test_data=test_data,
                    filter_test_trajectories=filter_test_trajectories,
                    states=states,
                    plot_cfg=plot_cfg,
                    save_cfg=save_cfg,
                )

            else:
                if not os.path.exists(output_sub_dir):
                    os.makedirs(output_sub_dir, exist_ok=True)
                    json_files = [f for f in os.listdir(sub_dir_path) if f.endswith(".json") and "meta" not in f]

                    if not json_files:
                        print(f"No .json files in {sub_dir_path} found. Skipping.")
                        continue

                    n_to_select = min(n_trajectories, len(json_files))
                    files_to_plot = np.random.choice(json_files, size=n_to_select, replace=False)

                    try:
                        meta_data = _load_single_json(os.path.join(sub_dir_path, "meta_data.json"))
                    except FileNotFoundError:
                        print(f"Warning: No meta_data.json in {sub_dir_path} found.")
                        meta_data = {}
                    surrogate_type = meta_data.get("surrogate_type")

                    for filename in files_to_plot:
                        file_path = os.path.join(sub_dir_path, filename)
                        surrogate_result = _load_single_json(file_path)

                        if filter_test_trajectories:
                            test_data_index = surrogate_result.get("meta_data", {}).get("index")
                            if test_data_index is None:
                                print(f"Warning: Metadata {filename} has no index key. Cannot plot test data.")
                                continue
                            test_data_to_plot = test_data[test_data_index]
                        else:
                            test_data_to_plot = test_data

                        for state in states:
                            # Kopie erstellen, um Seiteneffekte zu vermeiden
                            current_save_cfg = save_cfg.copy()
                            export_name = filename.replace(".json", f"_{state}")
                            current_save_cfg.update({"save_dir": output_sub_dir, "export_name": export_name})
                            plot_open_loop_state(
                                sim_cfg=sim_cfg,
                                surrogate_results={f"{sub_dir_name}": surrogate_result},
                                test_data=test_data_to_plot,
                                state=state,
                                surrogate_type=surrogate_type,
                                positions=[0, 3],  # Beispielpositionen
                                meta_data=meta_data,
                                plot_cfg=plot_cfg,
                                save_cfg=current_save_cfg,
                            )


# ------- Plots for performance Analyis over time ---------


# ---- Mean squared error of the nominal surrogates
def plot_mses_vs_time(
    mse_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    time: np.ndarray,
    plot_dir: str,
    save_cfg: Optional[Dict] = None,
    plot_cfg: Optional[Dict] = None,
):
    """Plots the Mean Squared Error (MSE) of surrogates over time.

    This function generates a 2D line plot showing the mean MSE for one or
    more surrogates as a function of time. A shaded region around each line
    represents the standard deviation of the MSE.

    Args:
        mse_data_dict: A dictionary mapping surrogate keys (str) to data tuples.
            Each tuple should contain two 1D NumPy arrays: (mean_mse_over_time,
            std_dev_over_time).
        time: A 1D NumPy array representing the time values for the x-axis.
        plot_dir: The directory path where the plot will be saved.
        save_cfg: An optional dictionary for saving behavior.
            Keys: "export_name" (str), "show_fig" (bool).
        plot_cfg: An optional dictionary for plot aesthetics.
            Keys: "legend" (Dict).

    Returns:
        The matplotlib Figure object.
    """
    # Initialize configuration dictionaries to avoid errors with None
    plot_cfg = plot_cfg or {}
    save_cfg = save_cfg or {}
    final_export_path = os.path.join(plot_dir, f"{save_cfg.get("export_name", "mse_vs_time")}.pdf")
    if not os.path.exists(final_export_path):
        # --- Plot Initialization ---
        fig, ax = plt.subplots()

        # --- Plot Styling ---
        # Create and set a property cycler for distinct line styles and colors,
        # ensuring visual clarity for multiple surrogates.
        num_surrogates = len(mse_data_dict)
        prop_cycler = make_line_cycler(n_styles=num_surrogates) + make_color_cycler(n_colors=num_surrogates)
        ax.set_prop_cycle(prop_cycler)
        # --- Plotting Loop ---
        # Iterate through each surrogate's data to plot its MSE evolution.
        for surr_key, (mean_mse, std_mse) in mse_data_dict.items():
            # Use a graceful fallback for labels if a key is not in the map
            label = latex_notation_map.get(surr_key, {}).get("general", surr_key)
            # Plot the mean MSE as a solid line
            ax.plot(time, mean_mse, label=label)
            # Add a shaded region to represent the standard deviation (±1 sigma)
            ax.fill_between(time, mean_mse, mean_mse + std_mse, alpha=0.25)
        # --- Axes Formatting and Legend ---
        ax.set_xlabel(latex_notation_map.get("time", "$t$ / s"))
        ax.set_ylabel("MSE($t$)")
        ax.set_ylim(*plot_cfg.get("ylims"))

        # Generate and apply custom formatting to the legend
        ax.legend()
        ax = format_legend(ax=ax, plot_cfg=plot_cfg)

        plt.tight_layout()
        plt.savefig(final_export_path, bbox_inches="tight")

        # Display the figure if configured to do so
        if save_cfg.get("show_fig", False):
            plt.show()


# ---- Physical Correctness
def plot_pc_violation_vs_time(
    time: np.ndarray,
    pc_violation_dict: Dict[str, np.ndarray],
    save_dir: str = None,
    plot_cfg: Optional[Dict] = None,
    save_cfg: Optional[Dict] = None,
) -> plt.figure:
    """
    Plots the Physics Constraint (PC) violation over time, comparing results
    from different model types (e.g., surrogates).

    The function plots the logarithm (base 10) of the L2-norm of the residual
    (violation) across the time dimension. It shows both the mean violation
    (solid line) across all trajectories and individual trajectory violations
    (faint lines) for visualization.

    Args:
        time: 1D array of time steps corresponding to the violation data.
        pc_violation_dict: Dictionary where keys are model identifiers (e.g.,
                           surrogate names) and values are 2D numpy arrays
                           containing the constraint violation, typically
                           of shape (n_trajectories, n_time_steps).
        save_dir: Directory path where the plot will be saved. If None, the
                  plot is only shown (if configured). Defaults to None.
        plot_cfg: Optional dictionary for plot customization, including
                  "alpha_all_traj" (opacity for individual trajectories),
                  "legend_y_pos", and "annotations_y" for text placement.
        save_cfg: Optional dictionary for save/show customization, including
                  "export_name" and "show_fig".

    Returns:
        plt.figure: The matplotlib Figure object generated.
    """

    plot_cfg = plot_cfg or {}
    save_cfg = save_cfg or {}
    final_save_path = os.path.join(save_dir, save_cfg.get("export_name", "pc_violation_vs_time") + ".pdf")
    if not os.path.exists(final_save_path):
        print("---- Plotting physics violation vs time. ----")
        n_data_rows = len(pc_violation_dict)
        fig, ax = plt.subplots()
        prop_cylcer = make_line_cycler(n_data_rows) + make_color_cycler(n_data_rows)
        ax.set_prop_cycle(prop_cylcer)
        colors = make_colors(n_data_rows, alpha=plot_cfg.get("alpha_all_traj", 0.2))

        label = r"{}".format(latex_notation_map["b_residual"])
        ax.axhline(y=np.log10(2.2 * 10**-16), color="gray", ls="--", label=latex_notation_map["machine_epsilon"])
        [ax.plot(time, np.log10(pc_violation_dict[key].mean(axis=0)), label=latex_notation_map[key]["general"]) for key in pc_violation_dict.keys()]
        [ax.plot(time, np.log10(pc_violation_trajetories.T), color=color, ls="-") for pc_violation_trajetories, color in zip(pc_violation_dict.values(), colors)]

        [
            ax.annotate(
                latex_notation_map[key]["general"] + r": $\langle ||\bm{b}||_2 \rangle_t$ = " + f"{pc_violation_dict[key].mean():.2e}",
                xy=(0.95, plot_cfg.get("annotations_y", 0.5 - i * 0.1)),
                xycoords="axes fraction",
                ha="right",
                va="center",
            )
            for i, key in enumerate(pc_violation_dict.keys())
        ]
        ax.set_ylim(-17, -1)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, plot_cfg.get("legend_y_pos", 1.2)), ncol=n_data_rows + 1)
        ax.set_xlabel(latex_notation_map["time"])
        ax.set_ylabel(label)
        # plt.tight_layout()
        plt.subplots_adjust(right=0.95, left=0.1, top=0.88, bottom=0.15)

        plt.savefig(final_save_path)
        if save_cfg.get("show_fig", False):
            plt.show()


# ---- Uncertainty Quantification performance
def plot_intervall_coverages(
    coverages_dict: Dict[str, np.ndarray],
    time: np.ndarray,
    save_dir: str,
    plot_cfg: Optional[Dict] = None,
    save_cfg: Optional[Dict] = None,
):
    plot_cfg = plot_cfg or {}
    save_cfg = save_cfg or {}
    final_save_path = os.path.join(save_dir, f"{save_cfg.get("export_name", "coverages")}.pdf")

    # if not os.path.exists(final_save_path):
    print("---- Plotting intervall coverages. ----")
    fig, ax = plt.subplots(1)
    n_data_rows = len(coverages_dict.keys())
    prop_cycler = make_color_cycler(n_data_rows) + make_line_cycler(n_data_rows)
    ax.set_prop_cycle(prop_cycler)

    for surrogate_key, coverage_arr in coverages_dict.items():
        ax.plot(time, coverage_arr, label=latex_notation_map[surrogate_key]["general"])

    ax.axhline(0.8, label="ideal coverage", color="gray", ls="dashed")
    ax.set_xlabel(latex_notation_map.get("time", "t"))
    ax.set_ylabel(r"$\mathrm{coverage}$ / -")
    ax.set_ylim(0, 1)
    ax.legend()
    ax = format_legend(ax, plot_cfg=plot_cfg)

    plt.tight_layout()
    plt.savefig(final_save_path)
    if save_cfg.get("show_fig", False):
        plt.show()


def plot_intervall_widths(
    intervall_widths_dict: Dict[str, np.ndarray],
    time: np.ndarray,
    save_dir: str,
    plot_cfg: Optional[Dict] = None,
    save_cfg: Optional[Dict] = None,
):
    plot_cfg = plot_cfg or {}
    ylabels = plot_cfg.get("ylabels", ["z1", "z2", "z3", "z4"])
    save_cfg = save_cfg or {}
    final_save_path = os.path.join(save_dir, f"{save_cfg.get("export_name", "coverages")}.pdf")

    # incoming shape is {"surrogate": np.ndaarray (trajects, n_t_steps, n_measurements)}

    # if not os.path.exists(final_save_path):
    n_measurements = list(intervall_widths_dict.values())[0].shape[-1]
    fig, axes = plt.subplots(n_measurements, sharex=True, sharey=True)
    n_data_rows = len(intervall_widths_dict.keys())
    # colors = make_colors(n_data_rows, alpha=1)
    prop_cylcer = make_color_cycler(n_data_rows) + make_line_cycler(n_data_rows)

    for i, ax in enumerate(axes):
        ax.set_prop_cycle(prop_cylcer)
        for surrogate_key, intervall_width_arr in intervall_widths_dict.items():
            mean = intervall_width_arr.mean(axis=0)
            std = intervall_width_arr.std(axis=0)

            surrogate_label = latex_notation_map[surrogate_key]["general"]
            ax.plot(time, mean[:, i], label=surrogate_label)
            ax.fill_between(time, mean[:, i] + std[:, i], mean[:, i] - std[:, i], label=rf"$\pm \sigma(${surrogate_label})", alpha=0.3)
            ax.set_ylabel(ylabels[i])

    axes[-1].set_xlabel(latex_notation_map.get("time", "t"))
    # ax.set_ylim(0, 1)
    axes[0].legend()
    axes[0] = format_legend(axes[0], plot_cfg=plot_cfg)
    fig.text(0.02, 0.5, "rel. intervall width / -", va="center", rotation="vertical")

    plt.subplots_adjust(right=0.95, left=0.15, top=0.88, bottom=0.15)
    # plt.tight_layout()
    print("---- Saving intervall widths plot. ----")
    plt.savefig(final_save_path)
    if save_cfg.get("show_fig", False):
        plt.show()


# -------- Summarization plots -----------


def plot_general_metric_summary(
    metric_keys: List[str],
    xaxis_key: str,
    metric_summary: Dict[str, Dict[str, np.ndarray]],
    save_dir: str,
    plot_cfg: Optional[Dict] = None,
    save_cfg: Optional[Dict] = None,
):
    """
    Generates a figure with multiple subplots to summarize and compare various
    performance metrics across different surrogate model types.

    It plots one metric key per subplot against a common x-axis variable.

    Args:
        metric_keys: A list of metric names (strings) to be plotted on separate subplots.
        xaxis_key: The key corresponding to the data to be used for the x-axis (e.g.,
                   number of training samples, model size).
        metric_summary: A nested dictionary containing the data to be plotted.
                        Structure: {surrogate_type: {metric_key: np.ndarray, ...}}.
                        Metric values are expected to be arrays where the first column
                        is the mean and the second column (optional/commented out) is the standard deviation.
        save_dir: The directory path where the resulting plot PDF will be saved.
        plot_cfg: Optional dictionary for plot customization, potentially including
                  "xlabel", "ylabels", "titles", and "ylims".
        save_cfg: Optional dictionary for save/show customization, potentially including
                  "show_fig" and "export_name".
    """
    plot_cfg = plot_cfg or {}
    save_cfg = save_cfg or {}

    n_data_rows = len(metric_summary.keys())
    prop_cycler = make_color_cycler(n_data_rows) + make_marker_cyler(n_data_rows)
    fig, axes = plt.subplots(1, len(metric_keys), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ylabels = plot_cfg.get("ylabels", None)
    titles = plot_cfg.get("titles", None)
    for i, (ax, metric_key) in enumerate(zip(axes, metric_keys)):
        ax.set_prop_cycle(prop_cycler)
        for surrogate_key, metric_dict in metric_summary.items():
            ax.errorbar(
                metric_dict[xaxis_key],
                np.log10(metric_dict[metric_key][:, 0]),
                # yerr=metric_dict[metric_key][:, 1],
                label=latex_notation_map[surrogate_key]["general"],
                alpha=0.5,
                linestyle="none",
            )  # mean  # std
        ax.set_xlabel(plot_cfg.get("xlabel"))
        ax.set_ylabel(ylabels[i]) if ylabels is not None else 0
        ax.set_title(titles[i])
        ax.set_ylim(*plot_cfg.get("ylims", (None, None)))
    axes[-1].legend()

    plt.tight_layout()
    if save_cfg.get("show_fig", False):
        plt.show()

    os.makedirs(save_dir, exist_ok=True)
    final_export_path = os.path.join(save_dir, f"{save_cfg.get("export_name", "metric_summary")}.pdf")
    plt.savefig(final_export_path)
    plt.close(fig)


def plot_exec_time_distribution(
    surrogate_result_dict: Dict[str, Dict],
    plot_dir: str,
    plot_cfg: Optional[Dict] = None,
    save_cfg: Optional[Dict] = None,
):
    """
    Plots the distribution of total wall clock simulation times for different surrogate models.

    The plot is a box plot showing the distribution of log-transformed simulation times.
    Mean simulation time is annotated for each model.

    Args:
        surrogate_result_dict (Dict[str, Dict]): A dictionary where keys are model
            names (str) and values are dictionaries containing simulation results.
            Each result dict must contain the key 't_wall_total', which holds a
            numpy array of wall clock times (in seconds).
        plot_dir (str): Directory where the plot will be saved.
        plot_cfg (Optional[Dict]): Configuration dictionary for plot aesthetics
            (e.g., 'figsize', 'x_annotations'). Defaults to {}.
        save_cfg (Optional[Dict]): Configuration dictionary for saving/showing the
            plot (e.g., 'export_name', 'show_fig'). Defaults to {}.
    """

    plot_cfg = plot_cfg or {}
    save_cfg = save_cfg or {}
    final_export_path = os.path.join(plot_dir, f"{save_cfg.get('export_name', "exe_time_dist")}.pdf")
    if not os.path.exists(final_export_path):
        print("---- Plotting execution time distribution. ----")

        fig, ax = plt.subplots(figsize=plot_cfg.get("figsize", (10, 6)))
        colors = make_colors(len(surrogate_result_dict))
        x_vals = np.arange(1, len(surrogate_result_dict) + 1)
        boxplot = ax.boxplot(
            [np.log10(result["t_wall_total"]) for result in surrogate_result_dict.values()],
            positions=x_vals,
            patch_artist=True,
            widths=0.6,
            showfliers=False,
            medianprops=dict(color="black", linewidth=2),
        )
        for i, result in enumerate(surrogate_result_dict.values()):
            mean = result["t_wall_total"].mean()
            ax.annotate(
                r"$\langle \Delta t_\mathrm{sim}\rangle$ = " + f"{mean:.3f}" + " s",
                xy=(plot_cfg.get("x_annotations", [0.18, 0.5, 0.82])[i], 0.2),
                xycoords="axes fraction",
                ha="center",
                va="center",
            )

        ax.set_xticks(x_vals)
        ax.set_xticklabels([latex_notation_map[key]["general"] for key in surrogate_result_dict.keys()])
        ax.set_ylabel(r"$\log_{10}\!\left(\frac{\Delta t_{\mathrm{sim}}}{\mathrm{s}}\right)$")

        for box, color in zip(boxplot["boxes"], colors):
            box.set_facecolor(color)
            box.set_edgecolor(color)
            box.set_alpha(0.7)

        plt.tight_layout()

        plt.savefig(final_export_path)
        if save_cfg.get("show_fig", False):
            plt.show()


def plot_mse_distribution(
    mse_data_dict_list: List[Dict[str, np.ndarray]],
    plot_dir: str,
    state_keys: List[str],
    save_cfg: Optional[Dict] = None,
    plot_cfg: Optional[Dict] = None,
):
    """Plots violin distributions of Mean Squared Error (MSE) for different states.

    This function generates a figure with one or more subplots. Each subplot
    displays violin plots for the MSE distribution of various surrogates
    corresponding to a specific system state. The mean MSE is overlaid as a marker.

    Args:
        mse_data_dict_list: A list of dictionaries. Each dictionary maps a
            surrogate's key to its MSE data array. Each dictionary corresponds
            to a state in `state_keys`.
        plot_dir: The directory path where the plot will be saved.
        state_keys: A list of keys for the states being plotted, used for titles
            and labels.
        save_cfg: An optional dictionary for saving behavior.
            Keys: "export_name" (str), "show_fig" (bool).
        plot_cfg: An optional dictionary for plot aesthetics.
            Keys: "annotations_x" (List[float]), "annotations_y" (List[float]),
            "y_lower" (List[float]), "y_upper" (List[float]).

    Returns:
        The matplotlib Figure object.
    """
    # Initialize configuration dictionaries to avoid errors with None
    save_cfg = save_cfg or {}
    plot_cfg = plot_cfg or {}

    final_export_path = os.path.join(plot_dir, f"{save_cfg.get("export_name", "mse_dist")}.pdf")

    if not os.path.exists(final_export_path):
        # Standardize input to always be a list for consistent processing
        if not isinstance(mse_data_dict_list, list):
            mse_data_dict_list = [mse_data_dict_list]

        # --- Plot Initialization ---
        num_subplots = len(mse_data_dict_list)
        fig, axes = plt.subplots(ncols=num_subplots, sharex=True, sharey=True)

        # Ensure 'axes' is always iterable, even for a single subplot
        if num_subplots == 1:
            axes = [axes]

        # --- Plot-wide Configurations ---
        # Extract surrogate keys and colors from the first data dictionary
        # (assumes all dictionaries have the same surrogate keys)
        surrogate_keys = list(mse_data_dict_list[0].keys())
        surrogate_labels = [latex_notation_map[key]["general"] for key in surrogate_keys]
        colors = make_colors(len(surrogate_keys))

        # Get custom plot limits and annotation positions from config
        annotation_x = plot_cfg.get("annotations_x", [0.1] * len(surrogate_keys))
        annotation_y = plot_cfg.get("annotations_y", [0.8] * num_subplots)
        y_lims_lower = plot_cfg.get("y_lower", [None] * num_subplots)
        y_lims_upper = plot_cfg.get("y_upper", [None] * num_subplots)

        # --- Subplot Generation Loop ---
        # Iterate through each state's data to create a dedicated subplot
        for i, (ax, state_key, mse_data_dict) in enumerate(zip(axes, state_keys, mse_data_dict_list)):

            # Unpack data for the current subplot
            mse_data = list(mse_data_dict.values())
            means = [mse.mean() for mse in mse_data]
            stdevs = [mse.std() for mse in mse_data]
            x_vals = np.arange(1, len(surrogate_keys) + 1)
            # Use log10 for better visualization of error distribution
            log_mse_data = [np.log10(mse) for mse in mse_data]
            # Create violin plot
            boxplot = ax.boxplot(
                log_mse_data,
                positions=x_vals,
                patch_artist=True,
                widths=0.6,
                showfliers=False,
                medianprops=dict(color="black", linewidth=2),
            )
            # Customize boxes with specific colors
            for box, color in zip(boxplot["boxes"], colors):
                box.set_facecolor(color)
                box.set_edgecolor(color)
                box.set_alpha(0.7)
            # Overlay mean MSE values as markers and add text annotations
            for j, (mean, std) in enumerate(zip(means, stdevs)):
                # ax.scatter(x_vals[j], np.log10(mean), marker="o", s=30, color=colors[j], ec="black", zorder=3)
                text = rf"$\langle$MSE$\rangle_t = {mean:.2e}$" + "\n " + rf"$\pm {std:.2e}$"
                ax.annotate(text=text, xy=(annotation_x[j], annotation_y[i]), xycoords="axes fraction", va="center", ha="right")
            # --- Axes Formatting ---
            ax.set_xticks(x_vals)
            ax.set_xticklabels(surrogate_labels)
            ax.grid(True, linestyle="--", alpha=0.6)

            # Set y-axis label only for the first subplot
            if i == 0:
                ax.set_ylabel(f"log$_{{10}}$ MSE")
            ax.set_ylim(y_lims_lower[i], y_lims_upper[i])

        # --- Final Figure Adjustments and Export ---
        plt.tight_layout()
        plt.savefig(final_export_path, bbox_inches="tight")

        # Display the figure if configured to do so
        if save_cfg.get("show_fig", False):
            plt.show()
