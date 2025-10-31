import datetime
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import yaml

CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
from configs.experiment_uncertain_open_loop import data_cfg, test_cfg_list, training_cfg
from models import EtOxModel
from postprocessing.performance_metrics import calculate_intervall_width, separate_into_state_by_slice
from postprocessing.plot import plot_intervall_coverages, plot_intervall_widths, plot_random_trajectories
from postprocessing.plotting_helpers import make_colors
from routines.data_structurizer import DataStructurizer
from routines.utils import filter_test_data_for_surrogates, get_directory_for_today, load_json_results_for_all
from simulation.data_generation import generate_data_for_specs
from simulation.open_loop import run_open_loop_for_specs
from training.run import run_training


def run_uncertain_open_loop_experiment(
    sim_cfg: Dict,
    experiments_directory: str,
    meta_model: EtOxModel,
    data_structurizer: DataStructurizer,
    measurements_to_plot: Optional[List[int]] = [0, 3],
    n_time_steps_test: int = 480,
    n_test_trajectories: int = -1,
):
    # make directory
    exp_name = "002_uncertain_open_loop_kpis"
    experiment_dir = os.path.join(experiments_directory, exp_name)
    current_experiment_working_dir = os.path.join(experiment_dir, get_directory_for_today(experiment_dir))
    os.makedirs(current_experiment_working_dir, exist_ok=True)

    # generate uncertain training data
    for data_purpose, data_specs in data_cfg.items():
        data_target_dir = os.path.join(current_experiment_working_dir, "data", data_purpose)
        if not os.path.exists(data_target_dir):
            print(f"----- Generating data for {data_purpose}. -----")
            os.makedirs(data_target_dir, exist_ok=True)
            generate_data_for_specs(meta_model=meta_model, sim_cfg=sim_cfg, specs=data_specs, data_dir=data_target_dir)

    # train uncertainty models
    # train the narx and rom models
    trained_model_dir = os.path.join(current_experiment_working_dir, "trained_models")
    training_data_dir = os.path.join(current_experiment_working_dir, "data", "train")
    os.makedirs(trained_model_dir, exist_ok=True)
    run_training(
        model_parameter_dir=trained_model_dir,
        training_data_dir=training_data_dir,
        data_structurizer=data_structurizer,
        training_cfg=training_cfg,
    )

    # simulate open loop with uncertainty models
    result_directory = os.path.join(current_experiment_working_dir, "results")
    test_data_dir = os.path.join(current_experiment_working_dir, "data", "test")
    test_data = data_structurizer.load_data(data_dir=test_data_dir, num_trajectories=n_test_trajectories)

    if not os.path.exists(result_directory):
        os.makedirs(result_directory, exist_ok=True)
        for specs in test_cfg_list:
            # run simulation batches
            specs["n_trajectories"] = test_data.shape[0]
            run_open_loop_for_specs(
                specs=specs,
                meta_model=meta_model,
                data_structurizer=data_structurizer,
                sim_cfg=sim_cfg,
                model_parameter_dir=trained_model_dir,
                save_dir=result_directory,
                initialization_data=test_data,
            )

    # is the PC NARX better in predicting the state uncertainty ?
    # load upper and lower bound
    # target dictionary: {"narx": {"nominal": np.ndarray, "upper": np.ndarray, "lower": np.ndarray}}
    result_dict = {}
    # ----- Load result data -----
    for surrogate_entry in os.scandir(result_directory):
        results_for_surrogate = load_json_results_for_all(result_dir=surrogate_entry.path)
        # test_data_for_surrogate = filter_test_data_for_surrogates(test_data=test_data, surrogate_results=results_for_surrogate)
        # only keep the states and reduce them to the 24 measurements
        only_states = {}
        for key, result in results_for_surrogate.items():
            states = data_structurizer.get_states_from_data(result["_x"])
            # get the states only at the respective measurments and filter them
            only_states[key] = data_structurizer.get_states_at_measurements(states)[..., measurements_to_plot, :]
            # shape is (n_trajects, t_steps, n_states, n_measurements)
        result_dict[surrogate_entry.name] = only_states

    # ----- Calculate intervall widths -----
    intervall_widths = calculate_intervall_width(result_dict)
    intervall_widths_by_state = separate_into_state_by_slice(
        intervall_widths,
        state_slices=[slice(None, 5), slice(5, None)],
        scaling_factors=[sim_cfg["scales"].get("c"), sim_cfg["scales"].get("T")],
        apply=[np.mean, None],
        axes=[-1, None],
    )

    # coverages = {}
    # reduced_test_data = data_structurizer.reduce_measurements(test_data)
    # test_data_states = data_structurizer.get_states_at_measurements(data_structurizer.get_states_from_data(reduced_test_data)[:, :n_time_steps_test])

    # states_upper = data_structurizer.get_states_at_measurements(data_structurizer.get_states_from_data(results_for_surrogate["upper"]["_x"]))
    # states_lower = data_structurizer.get_states_at_measurements(data_structurizer.get_states_from_data(results_for_surrogate["lower"]["_x"]))

    # states_lower.shape == states_upper.shape == test_data_states.shape
    # # ----- Calculation of coverages. -------
    # states_inside = np.logical_and(absolute_upper >= test_data_states, test_data_states >= absolute_lower)
    # coverage = np.count_nonzero(states_inside, axis=0) / states_inside.shape[0]
    # coverages[surrogate_entry.name] = coverage

    # ---- Plotting functions
    time = np.arange(0, n_time_steps_test, sim_cfg["simulation"]["t_step"])
    plot_dir = os.path.join(current_experiment_working_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    for key, intervall_state_slice in zip(["states", "temp"], intervall_widths_by_state):
        plot_intervall_widths(
            time=time,
            intervall_widths_dict=intervall_state_slice,
            save_dir=plot_dir,
            plot_cfg={
                "ylabels": [r"$z = 0.25 L$", r"$z = L$"],
                "legend_y_pos": 1.3,
                "legend_cols": 4,
            },
            save_cfg={
                "export_name": f"intervall_widths_{key}",
            },
        )

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

    traj_validation_dir = os.path.join(plot_dir, "surrogate_trajectories")
    if not os.path.exists(traj_validation_dir):
        plot_random_trajectories(
            sim_cfg=sim_cfg,
            n_trajectories=3,
            result_dir=result_directory,
            save_to_dir=traj_validation_dir,
            test_data=test_data,
            filter_test_trajectories=True,
            plot_cfg={
                "t_steps": 480,
                "legend_y_pos": 1.35,
                "ylabel_size": 20,
                "test_data_color": make_colors(4, alpha=0.2)[1],
                "annotations": ["mse"],
            },
            save_cfg={
                "export_name": None,
                "save_meta": True,
                "show_fig": False,
            },
        )

    print("---- experiment 002 uncertain open loop finished. -----")


if __name__ == "__main__":
    sim_cfg_name = "etox_control_task.yaml"
    config_directory = os.path.abspath(os.path.join(ROOT_DIR, "configs"))
    with open(os.path.join(config_directory, sim_cfg_name), "r") as f:
        sim_cfg = yaml.safe_load(f)
    model_cfg_directory = os.path.abspath(os.path.join(ROOT_DIR, "models", sim_cfg["model_name"]))
    with open(os.path.join(model_cfg_directory, "EtOxModel.yaml"), "r") as f:
        model_cfg = yaml.safe_load(f)
    experiments_directory = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "experiments"))

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

    run_uncertain_open_loop_experiment(
        sim_cfg=sim_cfg,
        experiments_directory=experiments_directory,
        meta_model=meta_model,
        data_structurizer=structurizer,
    )
