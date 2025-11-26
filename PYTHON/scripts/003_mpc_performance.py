import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
sys.path.insert(0, ROOT_DIR)
from configs.uncertainty_quantification import mpc_perf_cfg
from models import EtOxModel
from postprocessing.performance_metrics import *
from postprocessing.plot import *
from postprocessing.plotting_helpers import make_colors
from routines.data_structurizer import DataStructurizer
from routines.utils import apply_to_double_dict, df_from_double_dict, get_directory_for_today, load_json_results_for_all
from simulation.closed_loop import run_narx_mpc_loop
from simulation.closed_loop_process import run_parallel_mpc_loop
from simulation.data_generation import generate_random_ramp_signal, run_parallel_simulations


def run_mpc_performance(
    sim_cfg: Dict,
    experiments_directory: str,
    meta_model: EtOxModel,
    path_to_init_data: str,
    data_structurizer: DataStructurizer,
    n_test_trajectories: int = -1,
):
    """The experiment mpc-performance calculates the mean performance and constraint violation
    of the NARX frame works vanilla, naive and pc for 100 independent parameter combinations."""

    # make directory
    exp_name = "003_mpc_performance"
    experiment_dir = os.path.join(experiments_directory, exp_name)
    current_experiment_working_dir = os.path.join(experiment_dir, get_directory_for_today(experiment_dir))
    os.makedirs(current_experiment_working_dir, exist_ok=True)
    trained_model_dir = os.path.join(current_experiment_working_dir, "trained_models")
    plot_dir = os.path.join(current_experiment_working_dir, "plots")
    assert os.path.exists(trained_model_dir), "The folder of the trained models does not exist."

    kinetic_parameters = meta_model.sample_parameters(
        n_batches=mpc_perf_cfg.get("n_experiments", 10),
        covariance_gain=mpc_perf_cfg.get("covariance_gain", 1),
        lam_bed_std=mpc_perf_cfg.get("lam_bed_std", 0.01),
    )
    tvp_signals = generate_random_ramp_signal(
        feature_bounds=[sim_cfg["tvps"]["level_bounds"]],
        num_steps=mpc_perf_cfg.get("t_steps") + mpc_perf_cfg["mpc_cfg"].get("n_horizon"),
        tau=mpc_perf_cfg.get("tvp_tau"),
        batch_size=mpc_perf_cfg.get("n_experiments"),
        time_step=sim_cfg["simulation"]["t_step"],
    )

    # load initialization data
    init_data = np.load(path_to_init_data)
    init_data = np.expand_dims(init_data, axis=0)
    init_data = np.repeat(init_data, repeats=mpc_perf_cfg.get("n_experiments"), axis=0)
    sim_initial_states = data_structurizer.get_states_from_data(init_data[:, -1], n_measurements=sim_cfg["simulation"]["N_finite_diff"])
    narx_initial_states = data_structurizer.reduce_measurements(init_data)
    narx_initial_states = data_structurizer.to_dompc_vector(narx_initial_states)[:, -1]

    results_dir = os.path.join(current_experiment_working_dir, "results")

    # loop over surrogate types
    for surrogate_key in mpc_perf_cfg.get("surrogate_types"):
        state_dict_path = os.path.join(trained_model_dir, mpc_perf_cfg["state_dict_folder"][surrogate_key])
        final_results_dir = os.path.join(results_dir, surrogate_key)
        if not os.path.exists(final_results_dir):
            os.makedirs(final_results_dir, exist_ok=True)
            run_parallel_mpc_loop(
                n_workers=mpc_perf_cfg.get("n_workers", 1),
                t_steps=mpc_perf_cfg.get("t_steps"),
                data_structurizer=data_structurizer,
                meta_model=meta_model,
                mpc_initial_states=narx_initial_states,
                simulator_initial_states=sim_initial_states,
                state_dict_dir=state_dict_path,
                narx_type=surrogate_key,
                scenarios=mpc_perf_cfg["mpc_cfg"].get("scenarios"),
                physical_params=kinetic_parameters,
                tvp_signals=tvp_signals,
                sim_cfg=sim_cfg,
                mpc_cfg=mpc_perf_cfg.get("mpc_cfg"),
                run_cfg={
                    "save_dir": final_results_dir,
                    "save_as": "json",
                    "result_name": "narx_mpc",
                    "save_variable_types": ["_y", "_u", "_tvp", "_aux", "t_wall_total"],
                },
            )

    # load json results
    complete_result_dict = load_json_results_for_all(results_dir)

    # calculate mean performance -> mean selectivity
    kpi_dict = {}
    for surrogate_key, surrogate_dict in complete_result_dict.items():
        mean_sel = surrogate_dict["_aux"][..., 1].mean()  # selectivity has index 1
        temp_arr = data_structurizer.get_states_from_data(surrogate_dict["_y"], state="T")
        conv_arr = surrogate_dict["_aux"][..., 2]  # conversion has index 2

        temp_violated = 0 < temp_arr - sim_cfg["states"]["upper_bounds"]["T"] / sim_cfg["scales"]["T"]
        conv_violated = 0 > conv_arr - sim_cfg["aux"]["lower_bounds"]["X"]
        temp_violated = temp_violated.mean()
        conv_violated = conv_violated.mean()
        t_mean = surrogate_dict["t_wall_total"].mean()

        if surrogate_key not in kpi_dict.keys():
            kpi_dict[surrogate_key] = {"selectivity": mean_sel, "wall_time": t_mean, "temp_vio": temp_violated, "conv_vio": conv_violated}

    kpi_df = pd.DataFrame(kpi_dict).T

    print(kpi_df.to_latex(column_format="ccccc", float_format="${:.4f}$".format))


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

    run_mpc_performance(
        data_structurizer=structurizer,
        sim_cfg=sim_cfg,
        meta_model=meta_model,
        experiments_directory=experiments_directory,
        path_to_init_data="/Users/jandavidridder/Desktop/Masterarbeit/Master-Thesis/PYTHON/models/EtOxModel/initialization_data.npy",
    )
