import datetime
import inspect
import json
import os
import sys
from typing import Dict, List, Type

import numpy as np
import yaml
from routines.setup_routines import SurrogateTypes

CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
MODEL_DIR = os.path.join(ROOT_DIR, "models", "EtOxModel")
sys.path.insert(0, ROOT_DIR)
from models import EtOxModel
from routines.utils import get_kwargs_from_specs
from simulation import generate_initial_state, generate_random_ramp_signal
from simulation.simulation_process import run_parallel_simulations

data_set_name = "train_1M"
sim_cfg_name = "etox_control_task.yaml"


def generate_data(
    specs: Dict,
    sim_cfg: Dict,
    meta_model: EtOxModel,
    t_steps: int,
    n_workers: int,
    n_trajectories: int,
    data_dir: str,
    vary_input: bool = True,
    vary_tvp: bool = True,
    use_true_model_params: bool = False,
    save_as: str = "npy",
    result_name: str = "traj",
):

    input_signals = generate_random_ramp_signal(
        feature_bounds=specs.get("input_bounds"),
        num_steps=t_steps,
        tau=specs.get("input_signal_tau", sim_cfg["simulation"].get("tau_system", 10)),
        time_step=sim_cfg["simulation"]["t_step"],
        batch_size=n_trajectories,
        randomize=vary_input,
    )
    tvp_signals = generate_random_ramp_signal(
        feature_bounds=specs.get("tvp_bounds"),
        num_steps=t_steps,
        time_step=sim_cfg["simulation"]["t_step"],
        tau=specs.get("tvp_signal_tau", sim_cfg["simulation"].get("tau_system", 10)),
        batch_size=n_trajectories,
        randomize=vary_tvp,
    )

    if use_true_model_params:
        kinetic_parametes = meta_model.get_true_parameters(n_batches=n_trajectories)
    else:
        kinetic_parametes = meta_model.sample_parameters(n_batches=n_trajectories, covariance_gain=specs.get("covariance_gain", 1), lam_bed_std=specs.get("lam_bed_std", 0.05))

    x0 = meta_model.get_initial_state(n_batches=n_trajectories)

    no_data_points = t_steps * n_trajectories
    print(f"Generating data for {no_data_points} points.")

    run_parallel_simulations(
        data_structurizer=None,
        initial_states=x0,
        input_signals=input_signals,
        tvp_signals=tvp_signals,
        model_params=kinetic_parametes,
        meta_model=meta_model,
        model_type=SurrogateTypes.Rigorous.value,
        n_workers=n_workers,
        model_parameter_dir=None,
        simulation_cfg=sim_cfg,
        t_steps=t_steps,
        scenario="nominal",
        save_kwargs={"result_name": result_name, "save_dir": data_dir, "save_as": save_as},
    )
    print(f"Training data save to {data_dir}")
    meta_data = {
        **specs,
    }
    with open(os.path.join(data_dir, "meta_data.json"), "w") as f:
        f.write(json.dumps(meta_data))


def generate_data_for_specs(meta_model: EtOxModel, sim_cfg: Dict, specs: Dict, data_dir: str):
    # extract valid keyword arguments
    kwargs = get_kwargs_from_specs(function=generate_data, specs=specs)
    generate_data(
        specs=specs,
        sim_cfg=sim_cfg,
        meta_model=meta_model,
        data_dir=data_dir,
        **kwargs,
    )


if __name__ == "__main__":
    data_path = "/Users/jandavidridder/Desktop/Masterarbeit/src/data"
    with open(os.path.join(CONFIG_DIR, sim_cfg_name), "r") as f:
        sim_cfg = yaml.safe_load(f)
    with open(os.path.join(MODEL_DIR, "EtOxModel.yaml"), "r") as f:
        model_cfg = yaml.safe_load(f)

    meta_model = EtOxModel(
        model_cfg=model_cfg,
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        N_finite_diff=sim_cfg["simulation"]["N_finite_diff"],
    )

    data_gen_specs_list = (
        {
            "desc": "training data for uncertainty models",
            "id": "unc_train",
            "physical_model": sim_cfg["model_name"],
            "use_true_model_params": False,
            "vary_input": True,
            "vary_tvp": True,
            "t_steps": 1024,
            "n_trajectories": 4,
            "n_workers": 2,
            "covariance_gain": 2,
            "lam_bed_std": 0.08,
        },
    )

    for spec in data_gen_specs_list:
        generate_data_for_specs(
            meta_model=meta_model,
            sim_cfg=sim_cfg,
            specs=spec,
            data_dir="test",
        )
