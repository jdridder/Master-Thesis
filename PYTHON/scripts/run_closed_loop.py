import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import yaml
from casadi import vertcat
from do_mpc.controller import MPC
from do_mpc.data import save_results
from do_mpc.model import Model
from do_mpc.simulator import Simulator

CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
CONFIG_NAME = "etox_control_task.yaml"
CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIR, "configs", CONFIG_NAME))
INITIAL_STATE_PATH = os.path.join(ROOT_DIR, "models", "EtOxModel", "initial_state.npy")
PHYS_CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIR, "models", "EtOxModel", "EtOxModel.yaml"))
RESULTS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "results"))
sys.path.insert(0, ROOT_DIR)

import l4casadi
from models import EtOxModel
from routines.data_structurizer import DataStructurizer
from routines.insights import plot_mpc_jacobi
from routines.setup_routines import *
from simulation.data_generation import generate_random_ramp_signal
from tqdm import tqdm


def main():
    lam_dudt = 1000
    lam_conversion = 1000
    surrogate_type = "narx"
    time_steps_to_simulate = 128
    T_penalty = None
    time_start = 10
    save_data = True
    r = 10
    with_opt_layer = True

    with open(CONFIG_PATH, "r") as f:
        sim_cfg = yaml.safe_load(f)
    with open(PHYS_CONFIG_PATH, "r") as f:
        meta_model_cfg = yaml.safe_load(f)
    structurizer = DataStructurizer(
        n_measurements=sim_cfg["narx"].get("n_measurements"),
        n_initial_measurements=sim_cfg["simulation"]["N_finite_diff"],
        time_horizon=sim_cfg["narx"].get("time_horizon"),
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        tvp_keys=sim_cfg["tvps"]["keys"],
    )

    N_trajectories = 32
    trajectory_idx = 6
    data_path = os.path.abspath("/Users/jandavidridder/Desktop/Masterarbeit/Master-Thesis/experiments/data/test")
    state_dict_dir = os.path.abspath("/Users/jandavidridder/Desktop/Masterarbeit/Master-Thesis/experiments/002_coverage_intervall_width/2025-11-22/trained_models/vanilla")

    full_data = structurizer.load_data(data_dir=data_path, num_trajectories=N_trajectories, num_time_steps=-1)[trajectory_idx]
    data = structurizer.reduce_measurements(full_data)
    initial_state_data = {"full_system": full_data[time_start], "mpc": data[: time_start + 1]}  # mpc data has to include the data at the time instant time_start

    meta_model = EtOxModel(model_cfg=meta_model_cfg, state_keys=sim_cfg["states"]["keys"], input_keys=sim_cfg["inputs"]["all_keys"], N_finite_diff=sim_cfg["simulation"]["N_finite_diff"])
    if surrogate_type == "narx":
        surrogate_expressions, mpc_surrogate = get_narx_expressions(
            data_structurizer=structurizer,
            super_model=meta_model,
            simulation_cfg=sim_cfg,
            with_opt_layer=with_opt_layer,
            model_parameter_dir=state_dict_dir,
            scenarios=["nominal", "upper", "lower"],
        )
        mpc_surrogate = configure_narx_surrogate(data_structurizer=structurizer, surrogate=mpc_surrogate, surrogate_expressions=surrogate_expressions, simulation_cfg=sim_cfg)
    # elif surrogate_type == "rom":
    # snapshots = structurizer.get_states_from_data(full_data, n_measurements=128).T
    #     mpc_surrogate = configure_rom_surrogate(data_structurizer=structurizer, super_model=meta_model, rank=r, simulation_cfg=sim_cfg, snapshots=snapshots, model_parameter_dir=MODEL_DIR)
    else:
        raise ValueError(f"Wrong surrogate type {surrogate_type}.")

    E_in, E_out = mpc_surrogate.x["chi_E"][0], mpc_surrogate.x["chi_E"][-1]
    EO_in, EO_out = mpc_surrogate.x["chi_EO"][0], mpc_surrogate.x["chi_EO"][-1]

    mpc_surrogate.set_expression("X", (E_in - E_out) / E_in)
    mpc_surrogate.set_expression("S", (EO_out - EO_in) / (E_in - E_out))
    mpc_surrogate.setup()

    full_model = meta_model.create_physical_model()
    full_model.setup()
    simulator = configure_simulator(simulator_model=full_model, simulation_cfg=sim_cfg)

    # This must be looped for the Bayesian Optimization loop
    mpc = configure_mpc(
        mpc_surrogate=mpc_surrogate,
        surrogate_type=surrogate_type,
        data_structurizer=structurizer,
        meta_model=meta_model,
        simulation_cfg=sim_cfg,
        lam_dudt=lam_dudt,
        lam_conversion=lam_conversion,
    )
    mpc.data.set_meta(surrogate_type=surrogate_type)
    mpc.data.set_meta(pc_layer=with_opt_layer)
    results = run_closed_loop(
        simulation_cfg=sim_cfg,
        simulator=simulator,
        mpc=mpc,
        surrogate_type=surrogate_type,
        meta_model=meta_model,
        time_steps=time_steps_to_simulate,
        data_structurizer=structurizer,
        initial_state_data=initial_state_data,
        save_data=save_data,
    )


if __name__ == "__main__":
    main()
