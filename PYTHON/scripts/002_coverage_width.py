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
from routines.utils import apply_to_double_dict, get_directory_for_today
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

    initial_state = meta_model.get_initial_state()

    for run_i in range(uq_test_cfg.get("n_experiments", 2)):
        t_start = time.perf_counter()
        print(f"--- Running coverage and intervall width calc for run {run_i} ---.")

        input_trajectory = generate_random_ramp_signal(
            feature_bounds=test_data_cfg.get("input_bounds"),
            num_steps=test_data_cfg.get("t_steps"),
            time_step=sim_cfg["simulation"].get("t_step", 1),
            tau=test_data_cfg.get("input_signal_tau"),
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
        for surrogate_key, state_dict_folder in zip(uq_test_cfg.get("surrogate_types"), uq_test_cfg.get("state_dict_folders")):
            final_model_parameter_dir = os.path.join(trained_model_dir, state_dict_folder)
            if surrogate_key not in narx_result_dict.keys():
                narx_result_dict[surrogate_key] = {}

            for scenario_key in ["upper", "lower"]:
                narx_result = run_open_loop(
                    sim_cfg=sim_cfg,
                    meta_model=meta_model,
                    data_structurizer=data_structurizer,
                    t_steps=uq_test_cfg.get("t_steps") - uq_test_cfg.get("warm_up_steps"),
                    warm_up_steps=uq_test_cfg.get("warm_up_steps"),
                    surrogate_type=surrogate_key,
                    scenario=scenario_key,
                    model_parameter_dir=final_model_parameter_dir,
                    initialization_data=init_data,
                    run_cfg={"save_as": "return", "save_variable_types": ["_y"]},
                    n_workers=uq_test_cfg.get("n_narx_workers", 1),
                )
                narx_result_dict[surrogate_key][scenario_key] = narx_result

        # calculate intervall width

        # save as .npy
        print(narx_result_dict)

        duration = time.perf_counter() - t_start
        print(f"--- Experimental run took {duration:.3f} s. ---")
        exit()

    # load npy files

    # summarize

    # make plots

    # # ----- Calculate intervall widths -----
    # intervall_widths = calculate_intervall_width(result_dict)
    # intervall_widths_by_state = separate_into_state_by_slice(
    #     intervall_widths,
    #     state_slices=[slice(None, 5), slice(5, None)],
    #     scaling_factors=[sim_cfg["scales"].get("c"), sim_cfg["scales"].get("T")],
    #     apply=[np.mean, None],
    #     axes=[-1, None],
    # )

    # # coverages = {}
    # # reduced_test_data = data_structurizer.reduce_measurements(test_data)
    # # test_data_states = data_structurizer.get_states_at_measurements(data_structurizer.get_states_from_data(reduced_test_data)[:, :n_time_steps_test])

    # # states_upper = data_structurizer.get_states_at_measurements(data_structurizer.get_states_from_data(results_for_surrogate["upper"]["_x"]))
    # # states_lower = data_structurizer.get_states_at_measurements(data_structurizer.get_states_from_data(results_for_surrogate["lower"]["_x"]))

    # # states_lower.shape == states_upper.shape == test_data_states.shape
    # # # ----- Calculation of coverages. -------
    # # states_inside = np.logical_and(absolute_upper >= test_data_states, test_data_states >= absolute_lower)
    # # coverage = np.count_nonzero(states_inside, axis=0) / states_inside.shape[0]
    # # coverages[surrogate_entry.name] = coverage

    # # ---- Plotting functions
    # for key, intervall_state_slice in zip(["states", "temp"], intervall_widths_by_state):
    #     plot_intervall_widths(
    #         time=time,
    #         intervall_widths_dict=intervall_state_slice,
    #         save_dir=plot_dir,
    #         plot_cfg={
    #             "ylabels": [r"$z = 0.25 L$", r"$z = L$"],
    #             "legend_y_pos": 1.3,
    #             "legend_cols": 4,
    #         },
    #         save_cfg={
    #             "export_name": f"intervall_widths_{key}",
    #         },
    #     )

    # coverage_for_state_slice = {surr_key: cov[:, state_slice].reshape((n_time_steps_test, -1)).mean(axis=-1) for surr_key, cov in coverages.items()}
    #     plot_intervall_coverages(
    #         time=time,
    #         coverages_dict=coverage_for_state_slice,
    #         save_dir=plot_dir,
    #         plot_cfg={
    #             "legend_y_pos": 1.15,
    #         },
    #         save_cfg={
    #             "export_name": f"coverages_{key}",
    #         },
    #     )

    # calculate and plot the KPIs
    # 1. coverage as f(t) over the complete horizon
    #  -> does it become worse later in the horizon?
    # pc narx better coverage (close to 90) ?

    # 2. intervall with
    # -> increases over the horizon?
    # pc narx smaller intervalls than narx?

    # traj_validation_dir = os.path.join(plot_dir, "surrogate_trajectories")
    # if not os.path.exists(traj_validation_dir):
    #     plot_random_trajectories(
    #         sim_cfg=sim_cfg,
    #         n_trajectories=3,
    #         result_dir=result_directory,
    #         save_to_dir=traj_validation_dir,
    #         test_data=test_data,
    #         filter_test_trajectories=False,
    #         plot_cfg={
    #             "t_steps": 480,
    #             "legend_y_pos": 1.35,
    #             "ylabel_size": 20,
    #             "test_data_color": make_colors(4, alpha=0.1)[1],
    #             "annotations": ["mse"],
    #         },
    #         save_cfg={
    #             "export_name": None,
    #             "save_meta": True,
    #             "show_fig": False,
    #         },
    #     )

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
