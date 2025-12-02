from time import perf_counter
from typing import Dict, List, Optional, Type

import numpy as np
from do_mpc.controller import MPC
from do_mpc.data import save_results
from do_mpc.model import Model
from do_mpc.simulator import Simulator
from routines.data_structurizer import DataStructurizer
from routines.setup_routines import *
from routines.utils import NumpyEncoder
from tqdm import tqdm

# create a parallel function for this to executed on multiple cores


def run_narx_mpc_loop(
    t_steps: int,
    sim_cfg: Dict,
    meta_model: EtOxModel,
    mpc_cfg: Dict,
    state_dict_dir: str,
    data_structurizer: DataStructurizer,
    physical_params: np.ndarray,
    tvp_signals: np.ndarray,
    mpc_initial_states: np.ndarray,
    simulator_initial_states: np.ndarray,
    narx_type: Type[SurrogateTypes],
    scenarios: List[str],
    run_cfg: Optional[Dict] = None,
    proc_name: Optional[str] = "",
):
    assert simulator_initial_states.shape[0] == mpc_initial_states.shape[0], "The batch size of simulator initial states and mpc initial states must be equal."

    # load torch models
    narx = configure_dompc_model(
        model_type=narx_type,
        data_structurizer=data_structurizer,
        meta_model=meta_model,
        model_parameter_dir=state_dict_dir,
        sim_cfg=sim_cfg,
        scenario=scenarios,
    )
    E_in, E_out = narx.x["chi_E"][0], narx.x["chi_E"][-1]
    EO_in, EO_out = narx.x["chi_EO"][0], narx.x["chi_EO"][-1]

    narx.set_expression("X", (E_in - E_out) / E_in)
    narx.set_expression("S", (EO_out - EO_in) / (E_in - E_out))
    narx.setup()

    fp_model = configure_dompc_model(model_type=SurrogateTypes.Rigorous.value, meta_model=meta_model)

    results = control(
        simulation_cfg=sim_cfg,
        n_time_steps=t_steps,
        data_structurizer=data_structurizer,
        mpc_cfg=mpc_cfg,
        simulator_model=fp_model,
        mpc_model=narx,
        simulator_initial_states=simulator_initial_states,
        mpc_initial_states=mpc_initial_states,
        tvp_signals=tvp_signals,
        physical_params=physical_params,
        run_cfg=run_cfg,
        process_name=proc_name,
    )
    return results


def control(
    simulation_cfg: Dict,
    n_time_steps: int,
    mpc_cfg: Dict,
    simulator_model: Model,
    mpc_model: Model,
    data_structurizer: DataStructurizer,
    physical_params: np.ndarray,  # batched
    tvp_signals: np.ndarray,  # batched
    simulator_initial_states: np.ndarray,  # batched
    mpc_initial_states: np.ndarray,  # batched
    process_name: Optional[str] = "",
    run_cfg: Optional[Dict] = None,
) -> Dict:
    run_cfg = run_cfg or {}
    n_trajects = tvp_signals.shape[0]
    if tvp_signals.ndim < 3:
        tvp_signals = np.expand_dims(tvp_signals, axis=0)
    if mpc_initial_states.ndim < 2:
        mpc_initial_states = np.expand_dims(mpc_initial_states, axis=0)
    if simulator_initial_states.ndim < 2:
        simulator_initial_states = np.expand_dims(simulator_initial_states, axis=0)
    if physical_params.ndim < 2:
        physical_params = np.expand_dims(physical_params, axis=0)  # add batch dimension
        physical_params = np.repeat(physical_params, axis=0, repeats=n_trajects)
    else:
        assert physical_params.shape[0] == n_trajects, f"The batch size of the physical parameters {physical_params.shape[0]} must equal the number of trajectories {n_trajects}."

    assert (
        n_trajects == mpc_initial_states.shape[0] == simulator_initial_states.shape[0]
    ), f"The number of trajectories to control {n_trajects} must equal the batch size of mpc initial states {mpc_initial_states.shape[0]} and the simulator initial states {simulator_initial_states.shape[0]}."
    iterable = zip(tvp_signals, physical_params, mpc_initial_states, simulator_initial_states)
    iterable = tqdm(iterable, desc="Running MPC control loops.", total=n_trajects) if process_name == "Proc 0" else iterable
    previous_parameter_combination = np.random.rand(*physical_params[0].shape) if physical_params is not None else None
    previous_tvp_signal = np.random.rand(*tvp_signals[0].shape)

    save_as = run_cfg.get("save_as", "json")
    save_dir = run_cfg.get("save_dir")

    for tvp_signal, parameter_combination, mpc_x0, simulator_x0 in iterable:
        if not np.allclose(previous_parameter_combination, parameter_combination) or not np.allclose(previous_tvp_signal, tvp_signal):
            simulator = configure_simulator(simulation_cfg=simulation_cfg, simulator_model=simulator_model, integration_opts=run_cfg.get("integration_opts", {}))
            if simulator_model.n_tvp > 0:
                tvp_template = simulator.get_tvp_template()
                tvp_fun = make_simulator_tvp_fun(
                    simulation_time_step=simulation_cfg["simulation"]["t_step"],
                    tvp_template=tvp_template,
                    tvp_traj=tvp_signal,
                    tvp_key=simulation_cfg["tvps"]["keys"][0],
                )
                simulator.set_tvp_fun(tvp_fun)

            if simulator_model.n_p > 0:
                set_p_fun(simulator, params=parameter_combination)
            simulator.setup()

            mpc = MPC(model=mpc_model)
            mpc = configure_mpc(mpc=mpc, mpc_cfg=mpc_cfg, surpress_ipopt=True if process_name != "Proc 0" else mpc_cfg.get("surpress_ipopt_output", False))
            if mpc_model.n_tvp > 0:
                tvp_template = mpc.get_tvp_template()
                tvp_fun = make_mpc_tvp_fun(simulation_time_step=simulation_cfg["simulation"]["t_step"], tvp_template=tvp_template, tvp_traj=tvp_signal)
                mpc.set_tvp_fun(tvp_fun)
            if mpc_model.n_p > 0:
                uncertainty_vals = mpc_cfg.get("uncertainty_values")
                mpc.set_uncertainty_values(**uncertainty_vals)
            mpc.setup()

            simulator_x0 = np.expand_dims(simulator_x0, axis=1)
            mpc_x0 = np.expand_dims(mpc_x0, axis=1)

        simulator.reset_history()
        simulator.x0 = simulator_x0
        simulator.set_initial_guess()

        mpc.reset_history()
        mpc.x0 = mpc_x0
        mpc.set_initial_guess()

        wall_times = np.zeros(n_time_steps)
        try:
            loop_iter = tqdm(range(n_time_steps), desc="Running control loop") if process_name == "Proc 0" else range(n_time_steps)
            for t in loop_iter:
                start = perf_counter()
                u_t = mpc.make_step(mpc_x0)
                wall_times[t] = perf_counter() - start
                y_next = simulator.make_step(u_t)
                tvp_t = tvp_signal[t].reshape((-1, 1))
                mpc_x0 = data_structurizer.update_dompc_vector(mpc_x0, u_t, tvp_t, y_next)

        except Exception as e:
            print(f"MPC control loop failed with error: {e}")
            continue
        del mpc

        var_types = run_cfg.get("save_variable_types")
        if save_as in ["npy", "json"]:
            ind = 1
            ext_result_name = run_cfg.get("result_name", "result")
            while os.path.isfile(f"{save_dir}/{ext_result_name}.{save_as}"):
                ext_result_name = f"{ind:03d}_{run_cfg.get("result_name", "result")}"
                ind += 1
            complete_file_name = os.path.join(save_dir, f"{ext_result_name}")

        if save_as == "npy":
            raise NotImplementedError("Npy not implemented as save type yet.")
        elif save_as == "json":
            with open(f"{complete_file_name}.json", "w") as f:
                json_result = simulator.data.export()
                json_result.update({"t_wall_total": wall_times})
                if var_types is not None:
                    json_result = {key: arr for key, arr in json_result.items() if key in var_types}
                f.write(json.dumps(json_result, indent=4, cls=NumpyEncoder))
        elif save_as == "pkl":
            save_results(save_list=[simulator], result_path=save_dir, result_name=f"/{run_cfg.get("result_name", "result")}")
        elif save_as == "return_simulator":
            return simulator.data
        elif save_as == "return_mpc":
            return mpc.data
        else:
            raise NotImplementedError(f"The save as type {save_as} is not implemented.")


def configure_mpc(mpc: MPC, mpc_cfg: Dict, surpress_ipopt: Optional[bool] = False) -> MPC:
    for state_key in mpc.model.x.keys():
        mpc.bounds["lower", "_x", state_key] = 0
    # mpc.bounds["upper", "_x", "T"] = 630

    mpc.scaling["_x", "past_inputs"] = mpc_cfg.get("input_scale", 1)
    mpc.scaling["_x", "past_tvps"] = mpc_cfg.get("tvp_scale", 1)

    # ------ Common for both ROM and NARX surrogate models -----
    mterm = -((mpc.model.aux["S"]) ** 2)  # objective_function(selectivity)
    lterm = -((mpc.model.aux["S"]) ** 2)  # objective_function(selectivity)
    mpc.set_objective(lterm=lterm, mterm=mterm)
    # constraints
    mpc.set_nl_cons("T_max", mpc.model.x["T"], ub=mpc_cfg["ub_T"], soft_constraint=True, penalty_term_cons=mpc_cfg["lam_Tmax"])
    mpc.set_nl_cons("conversion", -mpc.model.aux["X"], ub=-mpc_cfg["lb_X"], soft_constraint=True, penalty_term_cons=mpc_cfg["lam_X"])

    input_keys = mpc.model.u.keys()
    input_keys.remove("default")
    for pos, input_key in enumerate(input_keys):
        mpc.set_nl_cons(f"input temp {pos}", mpc.model.u[input_key] / mpc_cfg.get("input_scale") - mpc.model.x["T"], ub=0, soft_constraint=True, penalty_term_cons=mpc_cfg["lam_T_Tcool"])

    mpc.set_rterm(**mpc_cfg.get("lam_dudt"))

    for input_key in mpc.model.u.keys():
        mpc.scaling["_u", input_key] = mpc_cfg.get("input_scale")
        mpc.bounds["upper", "_u", input_key] = mpc_cfg["input_bounds"]["upper"]
        mpc.bounds["lower", "_u", input_key] = mpc_cfg["input_bounds"]["lower"]

    solver_opts = mpc_cfg.get("solver_opts", None)
    if solver_opts is not None:
        for key in solver_opts["ipopt"]:
            solver_opts["ipopt"][key] = solver_opts["ipopt"][key]

    mpc._settings.n_horizon = mpc_cfg["n_horizon"]
    mpc._settings.n_robust = mpc_cfg["n_robust"]
    mpc._settings.t_step = mpc_cfg["t_step"]
    mpc._settings.store_full_solution = mpc_cfg["store_full_solution"]

    mpc._settings.nlpsol_opts = solver_opts
    if surpress_ipopt:
        mpc._settings.supress_ipopt_output()

    return mpc

    # # ------ For ROM surrogate model -----
    # elif surrogate_type == "rom":
    #     lam_conversion = 0  # 10000
    #     lam_dTdz = 0.01
    #     lam_Tmax = 0.1
    #     lam_dudt = {"T_in": 5, "T_c0": 2, "T_c1": 0.1, "T_c2": 0.1, "T_c3": 0.1}
    #     mean_parameter_values = meta_model.get_mean_parameters()
    #     mpc.set_uncertainty_values(**mean_parameter_values)
    #     for state_key in simulation_cfg["states"]["keys"]:
    #         mpc.set_nl_cons(f"{state_key}_lb", -mpc_surrogate.aux[state_key], ub=0, soft_constraint=False)
