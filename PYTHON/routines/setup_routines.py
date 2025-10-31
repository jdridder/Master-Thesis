import json
import os
from enum import Enum
from typing import Callable, Dict, List, Optional, Type, Union

import l4casadi
import numpy as np
from casadi import DM, vertcat
from do_mpc.model import Model
from do_mpc.model._pod_model import ProperOrthogonalDecomposition
from do_mpc.simulator import Simulator
from models import EtOxModel
from neurals.casadi import get_narx_input_shift_rhs, make_explicit_layer
from neurals.torch_training import load_state_predictor
from neurals.TorchPredictors import PCStatePredictor, StatePredictor

from .data_structurizer import DataStructurizer


class SurrogateTypes(Enum):
    Rigorous = "rigorous"
    Rom = "rom"
    Vanilla = "vanilla"
    Naive = "naive_pc"
    Pc = "pc"


def configure_simulator(simulation_cfg: Dict, simulator_model: Model, integration_opts: Optional[Dict] = None) -> Simulator:
    simulator = Simulator(model=simulator_model)
    if simulator_model.model_type == "continuous":
        simulator.set_param(
            integration_tool="cvodes",
            abstol=float(simulation_cfg["simulation"]["abstol"]),
            reltol=float(simulation_cfg["simulation"]["reltol"]),
        )
    simulator.set_param(t_step=simulation_cfg["simulation"]["t_step"])
    simulator.settings.integration_opts = integration_opts if integration_opts is not None else {}

    return simulator


def get_narx_expressions(
    data_structurizer: DataStructurizer,
    simulation_cfg: Dict,
    super_model: EtOxModel,
    module_cls: Type[StatePredictor],
    scenarios: Union[str, List[str]],
    model_parameter_dir: str,
    with_opt_layer: bool = False,
):
    if isinstance(scenarios, str):
        scenarios = [scenarios]
    mpc_surrogate = Model("discrete", "MX")
    narx_input, mpc_surrogate = get_narx_input_shift_rhs(
        stack_function=data_structurizer.casadi_stack,
        model=mpc_surrogate,
        n_measurements=data_structurizer.n_measurements,
        state_keys=simulation_cfg["states"]["keys"],
        input_keys=simulation_cfg["inputs"]["all_keys"],
        tvp_keys=simulation_cfg["tvps"]["keys"],
        time_horizon=data_structurizer.time_horizon,
    )
    cfgs = {
        "nominal": {"state_model": {"file_name": "nominal_states.pth", "cls": module_cls}, "temperature_model": {"file_name": "nominal_temperature.pth", "cls": StatePredictor}},
        "upper": {"state_model": {"file_name": "upper_states.pth", "cls": module_cls}, "temperature_model": {"file_name": "upper_temperature.pth", "cls": StatePredictor}},
        "lower": {"state_model": {"file_name": "lower_states.pth", "cls": module_cls}, "temperature_model": {"file_name": "lower_temperature.pth", "cls": StatePredictor}},
    }
    torch_models = [load_state_predictor(cfgs[scenario_key], model_dir=model_parameter_dir) for scenario_key in scenarios]
    surrogate_expressions = {scenario_key: l4casadi.L4CasADi(torch_model, device="cpu") for scenario_key, torch_model in zip(scenarios, torch_models)}

    for key in surrogate_expressions.keys():
        surrogate_expressions[key] = surrogate_expressions[key](narx_input.T).T

    if with_opt_layer:
        A = super_model.get_balance_constraint_matrix(num_stacks=data_structurizer.n_measurements)
        bc = super_model.get_bc_for_all_measurements(n_measurements=data_structurizer.n_measurements)
        explicit_layer = make_explicit_layer(A=A, boundary_cond=bc)
        for key in surrogate_expressions.keys():
            surrogate_expressions[key] = explicit_layer(surrogate_expressions[key])
    return surrogate_expressions, mpc_surrogate


def configure_narx_surrogate(data_structurizer: DataStructurizer, surrogate: Model, super_model: EtOxModel, surrogate_expressions: Dict, simulation_cfg: Dict, scenario: str = None) -> Model:
    assert scenario in ["nominal", "upper", "lower"], f"The scenario case {scenario} must be one of nominal, upper or lower."
    if scenario is not None:
        rhs = surrogate_expressions[scenario]  # choose either "nominal", "lower", or "upper"
    else:
        alpha = surrogate.set_variable("_p", "alpha", shape=(3, 1))
        rhs = alpha[0] * surrogate_expressions["nominal"] + alpha[1] * surrogate_expressions["upper"] + alpha[2] * surrogate_expressions["lower"]
        # one-hot-encoding for the different models that account for different scenarios
    for i, state_key in enumerate(simulation_cfg["states"]["keys"]):
        start, stop = i * data_structurizer.n_measurements, (i + 1) * data_structurizer.n_measurements
        surrogate.set_rhs(state_key, rhs[start:stop])
    # stoic = {"E": -1, "EO": 1}
    # c_EO_in = super_model.bc["c_EO"]
    # c_E_in = super_model.bc["c_E"]
    # conversion = (c_E_in - surrogate.x["c_E"][-1]) / c_E_in
    # selectivity = (surrogate.x["c_EO"][-1] - c_EO_in) / (surrogate.x["c_E"][-1] - c_E_in) * stoic["E"] / stoic["EO"]
    # surrogate.set_expression("X", conversion)
    # surrogate.set_expression("S", selectivity)
    surrogate.setup()
    return surrogate


def configure_rom_surrogate(data_structurizer: DataStructurizer, super_model: EtOxModel, model_parameter_dir: str) -> Model:
    rom_surrogate = super_model.create_physical_model()
    rom_surrogate.setup()
    pod = ProperOrthogonalDecomposition()
    with open(os.path.join(model_parameter_dir, "rom_params.json"), "r") as f:
        rom_parameters = json.load(f)
    for key in rom_parameters.keys():
        rom_parameters[key] = np.array(rom_parameters[key])

    pod.import_parameters(rom_parameters)
    rom_surrogate = pod.reduce(rom_surrogate)
    data_structurizer.import_rom_parameters(rom_parameters)
    full_states = data_structurizer.rom_to_full(vertcat(rom_surrogate.x).reshape((1, -1)))  # shape must be (n_time_steps, n_features)
    reduced_states = data_structurizer.reduce_measurements(full_states, n_initial_measurements=data_structurizer.n_initial_measurements)
    for i, state_key in enumerate(data_structurizer.state_keys):
        start = i * data_structurizer.n_measurements
        stop = start + data_structurizer.n_measurements
        rom_surrogate.set_meas(state_key, expr=reduced_states[start:stop])
    rom_surrogate.setup()
    return rom_surrogate


def configure_rigorous_model(meta_model: EtOxModel) -> Model:
    model = meta_model.create_physical_model()
    model.setup()
    return model


def configure_dompc_model(model_type: Type[SurrogateTypes], sim_cfg: Dict, data_structurizer: DataStructurizer, meta_model: EtOxModel, model_parameter_dir: str, scenario: str) -> Model:
    if model_type == SurrogateTypes.Rom.value:
        dompc_model = configure_rom_surrogate(
            data_structurizer=data_structurizer,
            super_model=meta_model,
            model_parameter_dir=model_parameter_dir,
        )
    elif model_type == SurrogateTypes.Rigorous.value:
        dompc_model = configure_rigorous_model(meta_model=meta_model)
    elif model_type == SurrogateTypes.Vanilla.value or SurrogateTypes.Naive.value or SurrogateTypes.Pc.value:
        with_opt_layer = True if model_type == SurrogateTypes.Naive.value else False
        print(with_opt_layer)
        nn_module_cls = PCStatePredictor if model_type == SurrogateTypes.Pc.value else StatePredictor
        narx_expressions, dompc_model = get_narx_expressions(
            data_structurizer=data_structurizer,
            model_parameter_dir=model_parameter_dir,
            simulation_cfg=sim_cfg,
            module_cls=nn_module_cls,
            scenarios=scenario,
            super_model=meta_model,
            with_opt_layer=with_opt_layer,
        )
        dompc_model = configure_narx_surrogate(
            data_structurizer=data_structurizer,
            simulation_cfg=sim_cfg,
            super_model=meta_model,
            surrogate_expressions=narx_expressions,
            surrogate=dompc_model,
            scenario=scenario,
        )
    else:
        raise NotImplementedError(f"The provided surrogate type {model_type} is not implemented.")
    return dompc_model


def set_p_fun(simulator: Simulator, params: np.ndarray) -> Simulator:
    """Sets the do-mpc parameter function for the simulator. This function is called at the beginning of each simulation step to retrieve the nominal parameters.

    Args:
        simulator (Simulator): The do-mpc simulator object.

    Returns:
        Simulator: The simulator with the parameter function set.
    """
    p_template = simulator.get_p_template()
    assert params.shape[0] == len(p_template.keys()) - 1, f"The number of provided parameters {params.shape[0]} and parameter keys in the p template {p_template.keys()} do not match."
    for i, key in enumerate(p_template.keys()[1:]):  # skip the default key
        p_template[key] = params[i]

    def p_fun(t_now: float) -> Dict:
        return p_template

    simulator.set_p_fun(p_fun)
    return simulator


def make_simulator_tvp_fun(
    simulation_time_step: float,
    tvp_template: Dict,
    tvp_traj: np.ndarray,
    tvp_key: str,
) -> Callable:
    """Sets the do-mpc time varying parameter function for the simulator. This function is called at the beginning of each simulation step to retrieve the nominal parameters.

    Args:
        simulator (Simulator): The do-mpc simulator object.

    Returns:
        Simulator: The simulator with the parameter function set.
    """

    def tvp_fun(t_now: float) -> Dict:
        idx = int(t_now // simulation_time_step)
        tvp_template[tvp_key] = tvp_traj[idx]
        return tvp_template

    return tvp_fun


def make_mpc_tvp_fun(
    simulation_time_step: float,
    tvp_template: Dict,
    tvp_traj: np.ndarray,
) -> Callable:
    n_horizon = len(tvp_template["_tvp"])

    # TODO: This might be not necessary and introduces a bug
    def mpc_tvp_fun(t_now: float) -> Dict:
        idx = int(t_now // simulation_time_step)
        window = tvp_traj[idx : idx + n_horizon]
        # TODO: Instead of looping over the horizon of tvps, to vector arithmetics.
        list_window = []
        for t in window:
            list_window.append(DM(t))
        tvp_template["_tvp"] = list_window
        return tvp_template

    return mpc_tvp_fun
