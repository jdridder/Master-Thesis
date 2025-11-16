import os
import sys
from typing import Dict, Optional

import numpy as np
import yaml

CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
from configs.uncertainty_quantification import data_cfg, test_cfg_list, training_cfgs
from models import EtOxModel
from postprocessing.performance_metrics import *
from postprocessing.plot import *
from postprocessing.plotting_helpers import make_colors
from routines.data_structurizer import DataStructurizer
from routines.utils import apply_to_double_dict, get_directory_for_today, load_json_results_for_all
from simulation.data_generation import generate_data_for_specs
from simulation.open_loop import run_open_loop_for_specs
from training.run import run_training


def run_proof_of_concept(
    sim_cfg: Dict,
    experiments_directory: str,
    meta_model: EtOxModel,
    data_structurizer: DataStructurizer,
    n_test_trajectories: int = -1,
):
    """The experiment POC covers the vaildation of the physics consistency activation
    and the single trajectory simulations."""

    # make directory
    exp_name = "001_poc"
    experiment_dir = os.path.join(experiments_directory, exp_name)
    current_experiment_working_dir = os.path.join(experiment_dir, get_directory_for_today(experiment_dir))
    os.makedirs(current_experiment_working_dir, exist_ok=True)

    # generate uncertain training data
    for data_purpose, data_specs in data_cfg.items():
        data_target_dir = os.path.join(experiment_dir, "..", "" "data", data_purpose)
        if not os.path.exists(data_target_dir):
            print(f"----- Generating data for {data_purpose}. -----")
            os.makedirs(data_target_dir, exist_ok=True)
            generate_data_for_specs(meta_model=meta_model, sim_cfg=sim_cfg, specs=data_specs, data_dir=data_target_dir)

    # train uncertainty models
    # train the narx and rom models
    trained_model_dir = os.path.join(current_experiment_working_dir, "trained_models")
    training_data_dir = os.path.join(experiment_dir, "..", "data", "train")
    os.makedirs(trained_model_dir, exist_ok=True)
    boundary_cond = meta_model.get_bc_for_all_measurements(n_measurements=data_structurizer.n_measurements)[:, :20]  # only chi states
    weight_distances = {}

    for training_cfg in training_cfgs:
        final_model_dir = os.path.join(trained_model_dir, training_cfg.get("save_dir"))
        os.makedirs(final_model_dir, exist_ok=True)
        run_training(
            model_parameter_dir=final_model_dir,
            training_data_dir=training_data_dir,
            data_structurizer=data_structurizer,
            training_cfg=training_cfg,
            constraint_matrix=meta_model.get_balance_constraint_matrix(num_stacks=data_structurizer.n_measurements, include_temp_as_zero=False),
            boundary_cond=boundary_cond,
        )
        if training_cfg.get("save_distances", False):
            surrogate_key = training_cfg.get("save_dir")
            for file in os.scandir(final_model_dir):
                if file.name.startswith("distances") and file.name.endswith(".npy"):
                    quantile_key = f"{file.name.split(".")[1]}0"
                    distances = np.load(file)
                    if surrogate_key not in weight_distances.keys():
                        weight_distances[surrogate_key] = {}
                    weight_distances[surrogate_key][quantile_key] = distances

    # # simulate open loop with uncertainty models
    result_directory = os.path.join(current_experiment_working_dir, "results")
    os.makedirs(result_directory, exist_ok=True)
    test_data_dir = os.path.join(experiment_dir, "..", "data", "test")
    test_data = data_structurizer.load_data(data_dir=test_data_dir, num_trajectories=n_test_trajectories)

    for specs in test_cfg_list:
        # run simulation batches
        specs["n_trajectories"] = test_data.shape[0]
        final_model__parameter_dir = os.path.join(trained_model_dir, specs.get("state_dict_folder"))
        run_open_loop_for_specs(
            specs=specs,
            meta_model=meta_model,
            data_structurizer=data_structurizer,
            sim_cfg=sim_cfg,
            model_parameter_dir=final_model__parameter_dir,
            save_dir=result_directory,
            initialization_data=test_data[0],
        )

    # --------------- Plot Training Statistics ---------------
    plot_dir = os.path.join(current_experiment_working_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    light_colors = make_colors(4, alpha=0.1)
    full_colors = make_colors(4, alpha=1)
    plot_weight_distances(
        distance_dict=weight_distances,
        save_dir=plot_dir,
        sigma=0.002,
        plot_cfg={
            "colors": {"weight_function": full_colors[2], "90": light_colors[3], "10": light_colors[0]},
            "labels": {"weight_function": r"$\mathcal{N}(d) \; \mathrm{with} \; \sigma = 0.002$", "10": r"$d_{0.1}$", "90": r"$d_{0.9}$"},
            "xlabel": r"$d_\tau$ / -",
            "xlims": (0, 0.02),
            "ylabels": {"weight_function": "$w(d)$ / -", "distance": r"$N_\mathrm{points}$ / -"},
            "legend_y_pos": 1.15,
        },
    )
    # ------ Temperature Distances for Weight calculation

    # is the PC NARX better in predicting the state uncertainty ?
    # load upper and lower bound
    # target dictionary: {"narx": {"nominal": np.ndarray, "upper": np.ndarray, "lower": np.ndarray}}
    test_data = data_structurizer.reduce_measurements(test_data)
    t_step = sim_cfg["simulation"].get("t_step")
    warm_up_steps = test_cfg_list[0].get("warm_up_steps")
    result_dict = {}
    # ----- Load result data -----
    for surrogate_entry in os.scandir(result_directory):
        if os.path.isdir(surrogate_entry):
            results_for_surrogate = load_json_results_for_all(result_dir=surrogate_entry.path)
            only_states = {}
            for key, result in results_for_surrogate.items():
                only_states[key] = data_structurizer.get_states_from_data(result["_x"])
            result_dict[surrogate_entry.name] = only_states

    # --------------- Plot Recursive Simulation with Uncertainty Confidence ---------------

    # ------ State trajectories
    positions = [0, -1]
    state_indices = [-1, 2]
    state_labels = ["$T'$ / - ", r"$\chi_\mathrm{EO}$ / -"]

    for surrogate_key in result_dict.keys():
        for state_label, state_idx in zip(state_labels, state_indices):
            result_for_surrogate = {surrogate_key: result_dict[surrogate_key]}
            filtered_for_state_and_position = data_structurizer.filter_dict_data_for_state_and_position(result_for_surrogate, positions=positions, state_indices=state_idx, n_positions=4)
            plot_test_data_with_simulation(
                time_step=t_step,
                warm_up=warm_up_steps,
                test_data=data_structurizer.filter_arr_for_state_and_position(data_structurizer.get_states_from_data(test_data), positions, state_idx, n_positions=4),
                surrogate_dict=filtered_for_state_and_position,
                save_path=plot_dir,
                plot_cfg={
                    "colors": {"test": light_colors[1], "nominal": full_colors[2], "upper": full_colors[3], "lower": full_colors[0]},
                    "labels": {"test": "first principle", "nominal": r"$\mathbb{E}$", "lower": "$Q_{0.1}$", "upper": "$Q_{0.9}$"},
                    "linestyles": {"vanilla": "dashed", "pc": "solid", "naive_pc": "dashdot"},
                    "ylabels": [rf"{state_label}\\$z = 0.25L$", rf"{state_label}\\$z = L$"],
                    "legend_y_pos": 1.25,
                    "legend_cols": 4,
                },
                save_cfg={"save_name": f"test_with_{surrogate_key}_{state_idx}"},
            )

    # --- Input Trajectory
    plot_test_data_with_simulation(
        test_data=data_structurizer.get_inputs_from_data(test_data),
        time_step=t_step,
        warm_up=warm_up_steps,
        save_path=plot_dir,
        plot_cfg={
            "figsize": (10, 8),
            "ylabels": [r"$T_\mathrm{w,1}$ / K", r"$T_\mathrm{w,2}$ / K", r"$T_\mathrm{w,3}$ / K", r"$T_\mathrm{w,4}$ / K"],
            "colors": {"nominal": full_colors[2], "upper": full_colors[3], "lower": full_colors[0]},
        },
        save_cfg={
            "save_name": "input_trajectories",
        },
    )

    # --------------- Plot Recursive Physics consistency as ||b||_2 = f(t) for all surrogates ---------------
    physics_violations = apply_to_double_dict(double_dict=result_dict, fn=calculate_state_physics_vio, meta_model=meta_model, sim_cfg=sim_cfg, norm_measurements=True)
    time = np.arange(0, test_cfg_list[0].get("t_steps"))
    for surrogate_key in physics_violations.keys():
        plot_pc_violation_vs_time(
            time=time,
            pc_violation_dict={surrogate_key: physics_violations[surrogate_key]},
            save_dir=plot_dir,
            plot_cfg={
                "legend_y_pos": 1.25,
                "legend_cols": 5,
                "colors": {"nominal": full_colors[2], "upper": full_colors[3], "lower": full_colors[0]},
                "linestyles": {"vanilla": "dashed", "pc": "solid", "naive_pc": "dashdot"},
            },
            save_cfg={
                "show_fig": False,
                "export_name": f"pc_violation_{surrogate_key}",
            },
        )

    print(f"---- {exp_name} finished. -----")


if __name__ == "__main__":
    sim_cfg_name = "etox_control_task.yaml"
    config_directory = os.path.abspath(os.path.join(ROOT_DIR, "configs"))
    with open(os.path.join(config_directory, sim_cfg_name), "r") as f:
        sim_cfg = yaml.safe_load(f)
    model_cfg_directory = os.path.abspath(os.path.join(ROOT_DIR, "models", sim_cfg["model_name"]))
    with open(os.path.join(model_cfg_directory, "EtOxModel.yaml"), "r") as f:
        model_cfg = yaml.safe_load(f)
    experiments_directory = os.path.abspath(os.path.join(ROOT_DIR, "..", "experiments"))

    meta_model = EtOxModel(
        model_cfg=model_cfg,
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        N_finite_diff=sim_cfg["simulation"]["N_finite_diff"],
    )
    structurizer = DataStructurizer(
        n_initial_measurements=sim_cfg["simulation"]["N_finite_diff"],
        n_measurements=sim_cfg["narx"]["n_measurements"],
        time_horizon=sim_cfg["narx"]["time_horizon"],
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        tvp_keys=sim_cfg["tvps"]["keys"],
    )

    run_proof_of_concept(
        sim_cfg=sim_cfg,
        experiments_directory=experiments_directory,
        meta_model=meta_model,
        data_structurizer=structurizer,
    )
