import json
import os
import sys

import l4casadi
import numpy as np
import yaml
from models import EtOxModel
from routines.setup_routines import *
from simulation.simulation_process import run_parallel_simulations


def run_open_loop(
    sim_cfg: Dict,
    meta_model: EtOxModel,
    data_structurizer: DataStructurizer,
    t_steps: int,
    n_workers: int,
    surrogate_type: Type[SurrogateTypes],
    scenario: str,
    model_parameter_dir: str,  # TODO: Refactor such the parameters are passed in a dictionary with the fields {"nominal": params, "upper": params, "lower": params}
    initialization_data: np.ndarray,
    kinetic_parameter_type: Optional[str] = "true",
    warm_up_steps: Optional[int] = None,
    run_cfg: Optional[Dict] = None,
):
    if warm_up_steps is None:
        warm_up_steps = data_structurizer.time_horizon
    run_cfg = run_cfg or {}

    if initialization_data.ndim == 2:
        initialization_data = np.expand_dims(initialization_data, axis=0)

    input_signals = data_structurizer.get_inputs_from_data(initialization_data)[:, warm_up_steps:]
    tvp_signals = data_structurizer.get_tvps_from_data(initialization_data)[:, warm_up_steps:]

    if surrogate_type == SurrogateTypes.Rom:
        with open(os.path.join(model_parameter_dir, "rom_params.json"), "r") as f:
            rom_params = json.load(f)
            data_structurizer.import_rom_parameters(rom_params)
        snapshots = data_structurizer.get_states_from_data(initialization_data, n_measurements=data_structurizer.n_initial_measurements)
        x0_full = snapshots[..., warm_up_steps, :]
        x0 = data_structurizer.full_to_rom(x0_full)
    elif surrogate_type == SurrogateTypes.Rigorous:
        raise NotImplementedError("Rigorous model not implemented for open loop.")
    else:
        x0 = data_structurizer.to_dompc_vector(initialization_data[:, warm_up_steps - data_structurizer.time_horizon + 1 : (warm_up_steps + 1)])
        # TODO: Outsource the creation of x0 into the datastructurizer

    if kinetic_parameter_type == "true":
        kinetic_params = meta_model.get_true_parameters(n_batches=initialization_data.shape[0])
    elif kinetic_parameter_type == "nominal":
        kinetic_params = meta_model.get_parameter_scenario("nominal")
    elif kinetic_parameter_type == "default":
        kinetic_params = np.empty(1)
    else:
        raise ValueError("Provide either true or nominal as valid choices for the model parameters.")

    print(f"----- Running open loop simulation for {surrogate_type}. -----")

    result = run_parallel_simulations(
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
        run_cfg=run_cfg,
        n_workers=n_workers,
    )
    return result
