import datetime
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import yaml

CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
from configs.uncertainty_quantification import data_cfg, test_cfg_list, uq_test_cfg
from models import EtOxModel
from postprocessing.performance_metrics import *
from postprocessing.plot import *
from postprocessing.plotting_helpers import format_legend, make_colors
from routines.data_structurizer import DataStructurizer
from routines.utils import NumpyEncoder, apply_to_double_dict, df_from_double_dict, get_directory_for_today, load_json_results
from simulation.data_generation import generate_data_for_specs
from simulation.open_loop import run_open_loop
from simulation.simulation import generate_random_ramp_signal
from simulation.simulation_process import run_parallel_simulations


def run_coverage_width(
    sim_cfg: Dict,
    experiments_directory: str,
    meta_model: EtOxModel,
    data_structurizer: DataStructurizer,
):
    """The experiment coverage_width treats the calculation of
    intervall coverages and intervall widths for the temperature quantile models.
    It is based on the data and the trained models of the poc experiment."""

    # make directory
    exp_name = "002_coverage_intervall_width"
    experiment_dir = os.path.join(experiments_directory, exp_name)
    current_experiment_working_dir = os.path.join(experiment_dir, get_directory_for_today(experiment_dir))
    os.makedirs(current_experiment_working_dir, exist_ok=True)
    test_data_cfg = data_cfg.get("test")

    trained_model_dir = os.path.join(current_experiment_working_dir, "trained_models")
    results_dir = os.path.join(current_experiment_working_dir, "results")
    plot_dir = os.path.join(current_experiment_working_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    initial_state = meta_model.get_initial_state()

    # --- Perform the simulations ----
    for run_i in range(uq_test_cfg.get("n_experiments", 2)):
        save_path_for_run = os.path.join(results_dir, f"run_{run_i}.json")
        if not os.path.exists(save_path_for_run):
            t_start = time.perf_counter()
            print(f"--- Running coverage and intervall width calc for run {run_i} ---.")

            input_trajectory = generate_random_ramp_signal(
                feature_bounds=test_data_cfg.get("input_bounds"),
                num_steps=test_data_cfg.get("t_steps"),
                time_step=sim_cfg["simulation"].get("t_step", 1),
                tau=uq_test_cfg.get("input_signal_tau"),
            )
            tvp_trajectory = generate_random_ramp_signal(
                feature_bounds=test_data_cfg.get("tvp_bounds"),
                num_steps=test_data_cfg.get("t_steps"),
                time_step=sim_cfg["simulation"].get("t_step", 1),
                tau=test_data_cfg.get("tvp_signal_tau"),
            )

            # sample parameters
            sampled_parameters = meta_model.sample_parameters(
                n_batches=uq_test_cfg.get("N_fp_trajects", None),
                covariance_gain=test_data_cfg.get("covariance_gain", None),
                lam_bed_std=test_data_cfg.get("lam_bed_std", None),
            )

            # run first principle model N times
            first_principle_results = run_parallel_simulations(
                simulation_cfg=sim_cfg,
                meta_model=meta_model,
                t_steps=uq_test_cfg.get("t_steps"),
                model_type=SurrogateTypes.Rigorous.value,
                initial_states=np.repeat(initial_state, axis=0, repeats=uq_test_cfg.get("N_fp_trajects", None)),
                tvp_signals=np.repeat(tvp_trajectory, axis=0, repeats=uq_test_cfg.get("N_fp_trajects", None)),
                input_signals=np.repeat(input_trajectory, axis=0, repeats=uq_test_cfg.get("N_fp_trajects", None)),
                model_params=sampled_parameters,
                run_cfg={"save_as": "return", "save_variable_types": ["_y", "_u", "_tvp"]},
                n_workers=uq_test_cfg.get("n_fp_workers", 1),
            )

            # run surrogates once
            # run for vanilla, naive_pc, pc, only upper and lower quantile models
            narx_result_dict = {}
            init_data = first_principle_results.mean(axis=0, keepdims=True)
            warm_up_steps = uq_test_cfg.get("warm_up_steps")
            for surrogate_key, state_dict_folder in zip(uq_test_cfg.get("surrogate_types"), uq_test_cfg.get("state_dict_folders")):
                final_model_parameter_dir = os.path.join(trained_model_dir, state_dict_folder)
                if surrogate_key not in narx_result_dict.keys():
                    narx_result_dict[surrogate_key] = {}

                for scenario_key in ["upper", "lower"]:
                    narx_result = run_open_loop(
                        sim_cfg=sim_cfg,
                        meta_model=meta_model,
                        data_structurizer=data_structurizer,
                        t_steps=uq_test_cfg.get("t_steps") - warm_up_steps,
                        warm_up_steps=warm_up_steps,
                        surrogate_type=surrogate_key,
                        scenario=scenario_key,
                        model_parameter_dir=final_model_parameter_dir,
                        initialization_data=init_data,
                        run_cfg={"save_as": "return", "save_variable_types": ["_y"]},
                        n_workers=uq_test_cfg.get("n_narx_workers", 1),
                    )
                    narx_result_dict[surrogate_key][scenario_key] = data_structurizer.get_states_from_data(data=narx_result, state="T")

            # ----- Calculate intervall widths and coverages for the temperature quantiles -----
            expected_value_traj = data_structurizer.get_states_from_data(init_data[:, warm_up_steps:], state="T")
            intervall_dict = calculate_intervall_width(narx_result_dict, true_expected_val_traj=expected_value_traj)
            fp_temperature_trajects = data_structurizer.get_states_from_data(data=first_principle_results, state="T")
            coverage_dict = calculate_coverage(surrogate_dict=narx_result_dict, test_data=fp_temperature_trajects[:, warm_up_steps:])

            # save as
            for surrogate_key in intervall_dict:
                intervall_dict[surrogate_key].update(**coverage_dict[surrogate_key])

            with open(save_path_for_run, "w") as f:
                f.write(json.dumps(intervall_dict, cls=NumpyEncoder, indent=4))

            duration = time.perf_counter() - t_start
            print(f"--- Experimental run took {duration:.3f} s. ---")

    with open(os.path.join(results_dir, "uq_meta.json"), "w") as f:
        f.write(json.dumps(uq_test_cfg, indent=4))

    # --- Analysis ---
    width_coverage_dict = load_json_results(result_dir=results_dir, n_trajectories=10)
    # ---- Plotting functions
    deep_colors = make_colors(4)
    for surrogate_key, surrogate_dict in width_coverage_dict.items():
        plot_coverage_width_vs_z(
            z_coords=np.arange(0.25, 1.25, step=0.25),
            coverage_width_dict={surrogate_key: surrogate_dict},
            save_dir=plot_dir,
            plot_cfg={
                "colors": {"coverage": deep_colors[0], "intervall_width": deep_colors[2], "ideal_coverage": "gray"},
                "ylabels": {"intervall_width": "rel. intervall width / -", "coverage": "coverage / -", "ideal_coverage": "ideal coverage"},
                "ylims": {"intervall_width": (0, 0.1), "coverage": (0, 1.05)},
                "xlims": (0, 1.25),
                "legend_y_pos": 1.15,
                "ideal_coverage": 0.8,
            },
            save_cfg={"export_name": f"{surrogate_key}_coverage_width"},
        )

    # ---- Dataframe Action ----
    width_coverage_dict = apply_to_double_dict(width_coverage_dict, fn=np.squeeze)
    width_coverage_dict_mean = apply_to_double_dict(width_coverage_dict, fn=np.mean, axis=(0, 1), keepdims=False)
    width_coverage_dict_std = apply_to_double_dict(width_coverage_dict, fn=np.std, axis=(0, 1), keepdims=False)

    positions = np.arange(start=0.25, stop=1.25, step=0.25)
    mean_df = df_from_double_dict(width_coverage_dict_mean, column_names=["framework", "metric", "position", "value"], arr_indices=positions)
    mean_df["param"] = "mean"
    std_df = df_from_double_dict(width_coverage_dict_std, column_names=["framework", "metric", "position", "value"], arr_indices=positions)
    std_df["param"] = "std"
    df = pd.concat([mean_df, std_df], axis=0).reset_index(drop=True)
    df = pd.pivot_table(df, index=["position", "framework"], columns=["metric", "param"], values="value")

    print(df.to_latex(float_format="${:.4f}$".format))

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

    run_coverage_width(
        sim_cfg=sim_cfg,
        experiments_directory=experiments_directory,
        meta_model=meta_model,
        data_structurizer=structurizer,
    )
