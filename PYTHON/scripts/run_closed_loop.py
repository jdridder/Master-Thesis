import json
import os
import sys
from typing import Dict

import numpy as np
import yaml
from casadi import vertcat
from do_mpc.controller import MPC
from do_mpc.data import Data, save_results
from do_mpc.model import Model
from do_mpc.model._pod_model import ProperOrthogonalDecomposition
from do_mpc.simulator import Simulator
from routines.setup_routines import configure_narx_surrogate, configure_rom_surrogate, configure_simulator, get_narx_expressions

CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
CONFIG_NAME = "etox_control_task.yaml"
CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIR, "configs", CONFIG_NAME))
INITIAL_STATE_PATH = os.path.join(ROOT_DIR, "models", "EtOxModel", "initial_state.npy")
PHYS_CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIR, "models", "EtOxModel", "EtOxModel.yaml"))
MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, "models", "EtOxModel", "surrogates"))
RESULTS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "results"))
sys.path.insert(0, ROOT_DIR)

import l4casadi
from models import EtOxModel
from neurals.casadi import get_narx_input_shift_rhs, make_explicit_layer
from neurals.torch_training import load_state_predictor
from routines.insights import plot_mpc_jacobi
from tqdm import tqdm

from PYTHON.MYCODE.simulation.data_generation import DataStructurizer, generate_random_ramp_signal, load_data


def configure_mpc(
    mpc_surrogate: Model,
    surrogate_type: str,
    data_structurizer: DataStructurizer,
    meta_model: EtOxModel,
    simulation_cfg: Dict,
    lam_dudt: float = 0,
    lam_conversion: float = 0,
    lam_dTdz: float = 0,
    lam_Tmax: float = 0,
    mpc_solver_opts: Dict = None,
    surpress_ipopt_output: bool = False,
) -> MPC:
    # -------------- Create the MPC Controller --------------
    mpc = MPC(mpc_surrogate)
    solver_opts = {
        "ipopt": {
            "max_iter": 1000,
            "tol": 1e-4,
            "acceptable_tol": 5e-4,
            "print_level": 5,
            "warm_start_init_point": "yes",
            "linear_solver": "ma57",
            "hessian_approximation": "limited-memory",
        }
    }

    # ------ For NARX surrogate model -----
    if surrogate_type == "narx":
        lam_conversion = 0  # 10000
        lam_dTdz = 0.01
        lam_Tmax = 0.1
        lam_dudt = {"T_in": 5, "T_c0": 2, "T_c1": 0.1, "T_c2": 0.1, "T_c3": 0.1}

        mpc.set_nl_cons("T_max", mpc_surrogate.x["T"], ub=630, soft_constraint=True, penalty_term_cons=lam_Tmax)
        for i, state_key in enumerate(simulation_cfg["states"]["keys"]):
            mpc.scaling["_x", state_key] = simulation_cfg["states"]["scales"][i]
            mpc.bounds["lower", "_x", state_key] = 0
        mpc.bounds["lower", "_x", "past_states"] = 0
        # mpc.bounds["upper", "_x", "T"] = 630

        state_scales = np.repeat(simulation_cfg["states"]["scales"], repeats=data_structurizer.n_measurements)
        past_state_scales = np.tile(state_scales, reps=data_structurizer.time_horizon - 1)
        mpc.scaling["_x", "past_states"] = past_state_scales
        mpc.scaling["_x", "past_inputs"] = 650
        mpc.scaling["_x", "past_tvps"] = 0.6
        # mpc.bounds["upper", "_x", "past_states"] = 630
        alpha_values = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # scenario 0: nominal, scenario 1: upper, scenario 2: lower
        mpc.set_uncertainty_values(alpha=alpha_values)

    # ------ For ROM surrogate model -----
    elif surrogate_type == "rom":
        lam_conversion = 0  # 10000
        lam_dTdz = 0.01
        lam_Tmax = 0.1
        lam_dudt = {"T_in": 5, "T_c0": 2, "T_c1": 0.1, "T_c2": 0.1, "T_c3": 0.1}
        mean_parameter_values = meta_model.get_mean_parameters()
        mpc.set_uncertainty_values(**mean_parameter_values)
        for state_key in simulation_cfg["states"]["keys"]:
            mpc.set_nl_cons(f"{state_key}_lb", -mpc_surrogate.aux[state_key], ub=0, soft_constraint=False)

    # ------ Common for both ROM and NARX surrogate models -----
    mterm = (0.6 - mpc_surrogate.aux["X"]) ** 2  # objective_function(selectivity)
    lterm = (0.6 - mpc_surrogate.aux["X"]) ** 2  # objective_function(selectivity)
    # constraints
    T_input = vertcat(mpc_surrogate.u)
    wall_temp_jumps = (T_input[:-1] - T_input[1:]) ** 2  # reduce size of jumps in the wall temperature in z-direction
    # mpc.set_nl_cons("temperature_jumps", wall_temp_jumps, ub=225, soft_constraint=True, penalty_term_cons=lam_dTdz)
    # mpc.set_nl_cons("conversion", -mpc_surrogate.aux["X"], ub=-0.3, soft_constraint=True, penalty_term_cons=lam_conversion)
    mpc.set_objective(lterm=lterm, mterm=mterm)

    # mpc.set_rterm(**lam_dudt)

    for input_key in simulation_cfg["inputs"]["all_keys"]:
        mpc.scaling["_u", input_key] = 650
        mpc.bounds["upper", "_u", input_key] = 650 if input_key == "T_in" else 630
        mpc.bounds["lower", "_u", input_key] = 580

    if mpc_solver_opts is not None:
        for key in mpc_solver_opts["ipopt"]:
            solver_opts["ipopt"][key] = mpc_solver_opts["ipopt"][key]
    mpc._settings.n_horizon = simulation_cfg["mpc"]["n_horizon"]
    mpc._settings.n_robust = simulation_cfg["mpc"]["n_robust"]
    mpc._settings.t_step = simulation_cfg["mpc"]["mpc_t_step"]
    mpc._settings.store_full_solution = simulation_cfg["mpc"]["store_full_solution"]
    mpc._settings.n_robust = simulation_cfg["mpc"]["n_robust"]
    mpc._settings.nlpsol_opts = solver_opts
    if surpress_ipopt_output:
        mpc._settings.supress_ipopt_output()

    meta_data = {
        "lam_conversion": lam_conversion,
        "lam_dTdz": lam_dTdz,
        "lam_Tmax": lam_Tmax,
        "lam_dudt": lam_dudt,
    }

    mpc.data.set_meta(**meta_data)
    return mpc


def main():
    lam_dudt = 1000
    lam_conversion = 1000
    surrogate_type = "rom"
    time_steps_to_simulate = 128
    T_penalty = None
    time_start = 10
    save_data = True
    n_measurements = 4
    r = 10
    n_back = 8
    with_opt_layer = True

    with open(CONFIG_PATH, "r") as f:
        sim_cfg = yaml.safe_load(f)
    with open(PHYS_CONFIG_PATH, "r") as f:
        meta_model_cfg = yaml.safe_load(f)
    structurizer = DataStructurizer(
        n_measurements=n_measurements, time_horizon=n_back, state_keys=sim_cfg["states"]["keys"], input_keys=sim_cfg["inputs"]["all_keys"], tvp_keys=sim_cfg["tvps"]["keys"]
    )

    N_trajectories = 32
    trajectory_idx = 6
    data_set_name = "test_100T"
    full_data = load_data(sim_cfg["model_name"], data_set_name, num_trajectories=N_trajectories, num_time_steps=-1)[trajectory_idx]
    data = structurizer.reduce_measurements(full_data, sim_cfg["simulation"]["N_finite_diff"])
    initial_state_data = {"full_system": full_data[time_start], "mpc": data[: time_start + 1]}  # mpc data has to include the data at the time instant time_start
    snapshots = structurizer.get_states_from_data(full_data, n_measurements=128).T

    meta_model = EtOxModel(model_cfg=meta_model_cfg, state_keys=sim_cfg["states"]["keys"], input_keys=sim_cfg["inputs"]["all_keys"], N_finite_diff=sim_cfg["simulation"]["N_finite_diff"])
    if surrogate_type == "narx":
        surrogate_expressions, mpc_surrogate = get_narx_expressions(
            data_structurizer=structurizer, super_model=meta_model, simulation_cfg=sim_cfg, reduced_rank=r, with_opt_layer=with_opt_layer, model_parameter_dir=MODEL_DIR
        )
        mpc_surrogate = configure_narx_surrogate(
            data_structurizer=structurizer, mpc_surrogate=mpc_surrogate, super_model=meta_model, surrogate_expressions=surrogate_expressions, simulation_cfg=sim_cfg
        )
    elif surrogate_type == "rom":
        mpc_surrogate = configure_rom_surrogate(data_structurizer=structurizer, super_model=meta_model, rank=r, simulation_cfg=sim_cfg, snapshots=snapshots, model_parameter_dir=MODEL_DIR)
    else:
        raise ValueError(f"Wrong surrogate type {surrogate_type}.")

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


def run_closed_loop(
    simulation_cfg: Dict,
    simulator: Simulator,
    mpc: MPC,
    surrogate_type: str,
    meta_model: EtOxModel,
    time_steps: int,
    data_structurizer: DataStructurizer,
    initial_state_data: Dict[np.ndarray, np.ndarray],
    save_data: bool = False,
) -> Dict:
    # This is equivalent to the initial run
    tvp_signal = generate_random_ramp_signal(
        levels=simulation_cfg["tvps"]["levels"], num_steps=time_steps + simulation_cfg["mpc"]["n_horizon"], time_step=simulation_cfg["simulation"]["t_step"], tau=20, seed=42
    )
    mpc_tvp_template = mpc.get_tvp_template()
    mpc_tvp_fun = meta_model.make_mpc_tvp_fun(simulation_cfg["simulation"]["t_step"], mpc_tvp_template, tvp_signal)
    mpc.set_tvp_fun(mpc_tvp_fun)

    x0_full = data_structurizer.get_states_from_data(initial_state_data["full_system"], n_measurements=simulation_cfg["simulation"]["N_finite_diff"]).reshape((-1, 1))
    if surrogate_type == "narx":
        x0_mpc = data_structurizer.to_dompc_vector(initial_state_data["mpc"], time_instant=-1)
    elif surrogate_type == "rom":
        x0_mpc = data_structurizer.full_to_rom(x0_full)
        mpc.scaling["_x", "x_tld"] = np.abs(x0_mpc)
    mpc.setup()
    mpc.x0 = x0_mpc
    mpc.set_initial_guess()

    if simulator.flags["setup"] is False:
        sim_tvp_template = simulator.get_tvp_template()
        tvp_fun = meta_model.make_simulator_tvp_fun(
            simulation_time_step=simulation_cfg["simulation"]["t_step"],
            tvp_template=sim_tvp_template,
            tvp_traj=tvp_signal[:time_steps],
            tvp_key="u",
        )
        simulator.set_tvp_fun(tvp_fun)
        kinetic_parameters = meta_model.get_true_parameters()
        meta_model.set_p_fun(simulator, params=kinetic_parameters)
        simulator.setup()

    simulator.x0 = x0_full
    simulator.reset_history()

    for i in tqdm(range(time_steps), desc="Running closed loop."):
        plot_mpc_jacobi(mpc)
        u_k = mpc.make_step(x0_mpc)
        # u_k = u_default[time_start + i].reshape((-1, 1))
        tvp_k = tvp_signal[i].reshape((-1, 1))
        x_sys = simulator.make_step(u_k)
        if surrogate_type == "narx":
            x0_mpc = data_structurizer.update_dompc_vector(x0_mpc, u_k, tvp_k, x_current_full=x_sys, n_full_measurements=simulation_cfg["simulation"]["N_finite_diff"])
        elif surrogate_type == "rom":
            x0_mpc = data_structurizer.full_to_rom(x_sys)

        # if i % safe_intervall == 0 and i >= 0:
        #     save_results([mpc, simulator], results_file_mame, overwrite=True)

    if save_data:
        save_results([mpc, simulator], result_path=RESULTS_DIR, result_name=f"/closed_loop_{surrogate_type}", overwrite=True)
    return {"mpc": mpc.data, "simulator": simulator.data}


if __name__ == "__main__":
    main()
