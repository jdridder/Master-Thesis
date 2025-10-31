import json
import os
import sys

import l4casadi
import numpy as np
import yaml
from models import EtOxModel
from routines.setup_routines import *
from routines.utils import get_kwargs_from_specs
from simulation import simulate
from simulation.simulation_process import run_parallel_simulations


def run_open_loop(
    specs: Dict,
    sim_cfg: Dict,
    meta_model: EtOxModel,
    data_structurizer: DataStructurizer,
    t_steps: int,
    n_workers: int,
    surrogate_type: Type[SurrogateTypes],
    scenario: str,
    model_parameter_dir: str,  # TODO: Refactor such the parameters are passed in a dictionary with the fields {"nominal": params, "upper": params, "lower": params}
    initialization_data: np.ndarray,
    kinetic_parameter_type: str = "true",
    warm_up_steps: int = None,
    save_dir: str = None,
    save_as: str = "json",
    integration_opts: Optional[Dict] = None,
):
    if warm_up_steps is None:
        warm_up_steps = data_structurizer.time_horizon

    reduced_data = data_structurizer.reduce_measurements(initialization_data)
    input_signals = data_structurizer.get_inputs_from_data(reduced_data)[:, warm_up_steps:]
    tvp_signals = data_structurizer.get_tvps_from_data(reduced_data)[:, warm_up_steps:]

    if surrogate_type == SurrogateTypes.Rom:
        with open(os.path.join(model_parameter_dir, "rom_params.json"), "r") as f:
            rom_params = json.load(f)
            data_structurizer.import_rom_parameters(rom_params)
        snapshots = data_structurizer.get_states_from_data(initialization_data, n_measurements=128)
        x0_full = snapshots[..., warm_up_steps, :]
        x0 = data_structurizer.full_to_rom(x0_full)
    elif surrogate_type == SurrogateTypes.Rigorous:
        raise NotImplementedError("Rigorous model not implemented for open loop.")
    else:
        x0 = data_structurizer.to_dompc_vector(reduced_data[:, warm_up_steps - data_structurizer.time_horizon + 1 : (warm_up_steps + 1)])
        # TODO: Outsource the creation of x0 into the datastructurizer

    with open(os.path.join(save_dir, "meta_data.json"), "w") as f:
        f.write(json.dumps(specs, indent=4))

    if kinetic_parameter_type == "true":
        kinetic_params = meta_model.get_true_parameters(n_batches=reduced_data.shape[0])
    elif kinetic_parameter_type == "nominal":
        kinetic_params = meta_model.get_parameter_scenario("nominal")
    elif kinetic_parameter_type == "default":
        kinetic_params = np.empty(1)
    else:
        raise ValueError("Provide either true or nominal as valid choices for the model parameters.")

    print(f"----- Running open loop simulation for {surrogate_type}. -----")

    run_parallel_simulations(
        simulation_cfg=sim_cfg,
        model_parameter_dir=model_parameter_dir,
        meta_model=meta_model,
        data_structurizer=data_structurizer,
        model_type=surrogate_type,
        t_steps=t_steps,
        scenario=scenario,
        initial_states=x0,  # supports batching
        input_signals=input_signals,  # supports batching
        tvp_signals=tvp_signals,
        model_params=kinetic_params,
        index=np.arange(input_signals.shape[0]),
        save_kwargs={"result_name": "open_loop_traj", "save_dir": save_dir, "save_as": save_as},
        n_workers=n_workers,
        integration_opts=integration_opts,
    )


def run_open_loop_for_specs(
    meta_model: EtOxModel,
    data_structurizer: DataStructurizer,
    sim_cfg: Dict,
    specs: Dict,
    save_dir: str,
    model_parameter_dir: str,
    initialization_data: np.ndarray,
    integration_opts: Optional[Dict] = None,
):
    """
    Runs an open-loop simulation for a given set of specifications and saves the results.

    Skips execution if the target directory already exists.
    """
    kwargs = get_kwargs_from_specs(function=run_open_loop, specs=specs)
    final_directory = os.path.join(save_dir, specs["sub_directory"])
    if os.path.exists(final_directory):
        return
    os.makedirs(final_directory, exist_ok=True)
    run_open_loop(
        specs=specs,
        sim_cfg=sim_cfg,
        data_structurizer=data_structurizer,
        meta_model=meta_model,
        save_dir=final_directory,
        model_parameter_dir=model_parameter_dir,
        initialization_data=initialization_data,
        integration_opts=integration_opts,
        **kwargs,
    )


# def main():
#     CURR_DIR = os.path.dirname(__file__)
#     ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
#     CONFIG_NAME = "etox_control_task.yaml"
#     CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIR, "configs", CONFIG_NAME))
#     PHYS_CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIR, "models", "EtOxModel", "EtOxModel.yaml"))
#     RESULTS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "results"))
#     MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, "models", "EtOxModel", "surrogates"))
#     sys.path.insert(0, ROOT_DIR)

#     with open(CONFIG_PATH, "r") as f:
#         sim_cfg = yaml.safe_load(f)
#     MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, "models", sim_cfg["model_name"], "surrogates"))
#     with open(PHYS_CONFIG_PATH, "r") as f:
#         meta_model_cfg = yaml.safe_load(f)
#     with open(os.path.join(MODEL_DIR, "training_meta_data.json"), "r") as f:
#         training_meta_data = json.load(f)

#     sim_result_dir = os.path.join(RESULTS_DIR, "simulation")

#     n_back = training_meta_data["back_horizon"]
#     r = training_meta_data["pca_rank"]
#     n_measurements = 4
#     structurizer = DataStructurizer(
#         n_measurements=n_measurements,
#         time_horizon=n_back,
#         state_keys=sim_cfg["states"]["keys"],
#         input_keys=sim_cfg["inputs"]["all_keys"],
#         tvp_keys=sim_cfg["tvps"]["keys"],
#     )
#     # # -------------- Load data of physical system --------------
#     data_set_name = "test_100T"
#     data = structurizer.load_test_data("EtOxModel", data_set_name)
#     surrogate_kwargs = {
#         "rom": {"rank": 24},
#         "narx": {"with_opt_layer": False},
#         "pc_narx": {"with_opt_layer": True},
#     }

#     run_open_loop(
#         meta_model_cfg=meta_model_cfg,
#         data_structurizer=structurizer,
#         model_parameter_dir=MODEL_DIR,
#         test_data=data,
#         sim_cfg=sim_cfg,
#         surrogate_type="narx",
#         surrogate_kwargs=surrogate_kwargs,
#         scenario="nominal",
#         t_steps=500,
#         save_dir=sim_result_dir,
#     )


# if __name__ == "__main__":
#     main()
