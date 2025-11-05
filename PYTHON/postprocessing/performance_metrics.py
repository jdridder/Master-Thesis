import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import l4casadi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml
from do_mpc.data import Data, MPCData, load_results
from models.EtOxModel.EtOxModel import EtOxModel
from postprocessing.Visualizer import Visualizer
from routines.data_structurizer import DataStructurizer


def calculate_state_mses(
    sim_cfg: Dict,
    surrogate_test_data: Dict[str, np.ndarray],
    surrogate_pred_data: Dict[str, np.ndarray],
    states: List[str] = ["chi_E", "T"],
    n_measurements: int = 4,
    keep: str = "time",
) -> Dict[str, Dict[str, np.ndarray]]:
    state_mses = {}  # should have structure {"state": {"surrogate_key": (mses, stdevs)}}
    start = 0
    stop = n_measurements * 5
    # TODO: Make this more elegant do not hard code positions
    for state_key in states:
        state_wise_predictions = {surr_key: surrogate_pred_data[surr_key][..., :, start:stop] for surr_key in surrogate_pred_data.keys()}
        state_wise_test_data = {surr_key: surrogate_test_data[surr_key][..., :, start:stop] for surr_key in surrogate_test_data.keys()}
        state_scale = sim_cfg["states"]["scales"][sim_cfg["states"]["keys"].index(state_key)]
        surrogate_mses = calculate_state_mse_for_surrogates(
            state_prediction_data=state_wise_predictions,
            state_test_data=state_wise_test_data,
            state_scale=state_scale,
            keep=keep,
        )
        state_mses[state_key] = surrogate_mses
        start = stop
        stop = stop + n_measurements
    return state_mses


def calculate_state_mse_for_surrogates(
    state_prediction_data: Dict[str, np.ndarray],
    state_test_data: Dict[str, np.ndarray],
    state_scale: float = 1,
    keep: str = "time",
) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    # assert len(specs_list) == len(state_prediction_data), f"The number of specs {len(specs_list)} must equal the number of surrogate predictions {len(state_prediction_data)}."
    # state_prediction_data has the form {"surrogate": [{"state": np.ndarray with shape (n_time_steps, states)}]} with the list indices correspond to the specs
    # it returns a list of dictionaries with the mean and std MSE as a function of time for each state in the form [{"state": (np.ndarray mean(t), np.ndarray std(t))}]
    # the list indices correspond to the state prediction data of each surrogate
    surrogate_mses = {}
    for surr_key, state_trajectories in state_prediction_data.items():
        test_data = state_test_data[surr_key]
        mse_result = calculate_state_mse(
            state_test_data=test_data,
            state_prediction_data=state_trajectories,
            state_scale=state_scale,
            keep=keep,
        )
        surrogate_mses[surr_key] = mse_result
    return surrogate_mses


def calculate_state_mse(state_test_data: np.ndarray, state_prediction_data: np.ndarray, state_scale: float = 1, keep: str = "time") -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculates a time-dependent, L2 norm-based error between test and prediction.
    mse = 1/n_features * ||state(z)||_2^2
    The mse is a function of time.

    Args:
        state_test_data: Ground truth data. Shape: (batch, time, feats) or (time, feats).
        state_prediction_data: Predicted data. Shape: (batch, time, feats).

    Returns:
        The mean and std of the error if test data is batched, otherwise the error itself.
    """
    assert (
        state_test_data.shape == state_prediction_data.shape
    ), f"Both matrices must have same batch size and number of time steps and features. You have {state_test_data.shape} and {state_prediction_data.shape}"
    difference = (state_test_data - state_prediction_data) / state_scale  # has shape (n_trajectories, time_steps, n_states)
    mse = 1 / state_prediction_data.shape[-1] * np.linalg.norm(difference, ord=2, axis=-1) ** 2  # contains the mse over all time steps as shape (n_trajectories, n_tsteps)

    if mse.ndim == 2:
        # mse is batched for all trajectories, give average mean and stddev over all batches
        if keep == "time":
            return mse.mean(axis=0), mse.std(axis=0)
    return mse.flatten()  # is scaled


def calculate_aux_from_states(state_matrix: np.ndarray) -> np.ndarray:
    "Calculates selectivity and conversion directly from the concentrations."
    states_inlet = structurizer.get_states_at_measurement_from_data(reduced_data=state_matrix, measurement=0)
    states_outlet = structurizer.get_states_at_measurement_from_data(reduced_data=state_matrix, measurement=3)
    # 2 == EO, 0 == E
    c_E_in = 29.099769529825323
    delta_EO = states_outlet[:, 2]
    delta_E = -states_outlet[:, 0] + c_E_in
    conversion = delta_E / c_E_in
    selectivity = delta_EO / delta_E
    aux = np.array([selectivity, conversion]).T
    return aux


def calculate_mean_selectivity(results: Union[List[Dict], Dict]) -> np.ndarray:
    """Averages the selectivity across the time of one control trajectory.
    It uses the selectivities of the real model, because this is the variable that counts in reality."""
    if isinstance(results, Dict):
        results = [results]
    mean_selectivities = np.zeros(len(results))
    for i, results in enumerate(results):
        mean_selectivities[i] = results["simulator"]["_aux", "S"].mean(axis=0)[0]
    return mean_selectivities


def calculate_mean_constraint_vio(results: Union[List[Dict], Dict], X_min: float, T_max: float) -> np.ndarray:
    """Calculates the contraint violation of the simulator system. This is the mean distance of points, that violate the constraints.
    Points, that satisfy the constraint are counted with zero."""
    if isinstance(results, Dict):
        results = [results]
    mean_violations = {"X": np.zeros(len(results)), "T": np.zeros(len(results))}
    n_discretization = results[0]["simulator"]["_x", "T"].shape[1]
    for i, result in enumerate(results):
        conversion_vio = X_min - result["simulator"]["_aux", "X"]
        # negative werte erfüllen die constraints (auf 0 setzten)
        no_vio_mask = conversion_vio < 0
        conversion_vio[no_vio_mask] = 0
        mean_conversion_vio = conversion_vio.mean(axis=0) / X_min  # relative violation with respect to the boundary value
        temp_vio = result["simulator"]["_x", "T", -n_discretization:] - T_max
        no_vio_mask = temp_vio < 0
        temp_vio[no_vio_mask] = 0
        # for the temperature norm over all discretization points
        temp_vio = np.linalg.norm(temp_vio, ord=2, axis=-1) / T_max
        mean_temp_vio = temp_vio.mean(axis=0)  # average over time
        mean_violations["X"][i], mean_violations["T"][i] = mean_conversion_vio[0], mean_temp_vio
    return mean_violations


def calculate_control_effort(results: Union[List[Dict], Dict], T_max: float) -> np.ndarray:
    if isinstance(results, Dict):
        results = [results]
    control_effort = np.zeros(len(results))
    for i, result in enumerate(results):
        control_signal = result["simulator"]["_u"]
        n_time_steps = control_signal.shape[0]
        delta_u = np.linalg.norm(control_signal[1:] - control_signal[0:-1], ord=2, axis=0) ** 2
        effort = np.linalg.norm(delta_u)
        control_effort[i] = effort / (n_time_steps * T_max)
    return control_effort


def calculate_mean_cputime(results_list: List[Dict]) -> np.ndarray:
    mean_cpu_times = np.zeros(len(results_list))
    for i, result in enumerate(results_list):
        mean_cpu_times[i] = result["mpc"]["t_wall_total"].mean()
    return mean_cpu_times


def calculate_physics_violation(X_pred: np.ndarray, A: np.ndarray, n_measurements: int, x_in: np.ndarray) -> np.ndarray:
    """Calculates the violation of the physics consistency constraints

    Args:
        X_pred (np.ndarray): The prediction vector of a surrogate model. Has shape (t_steps, n_states * n_measurements).
        A (np.ndarray): The matrix of the linear equality constraints that satisfy the balance laws. It has shape (n_laws, n_states)
        n_measurements: int: The number of the measurements.
        x_in (np.ndarrary): The states at the inlet for every time step. It has shape (t_steps, n_states).

    Returns:
        np.ndarray: The average across the states and time at each position in the reactor of the residual vector b of the constraint equation Az - b = 0 (it must equal to zero).
    """
    assert X_pred.shape[0] == x_in.shape[0], f"X and x_in must have the same number of time steps. You have {X_pred.shape} and {x_in.shape}"
    # calculate the absolute change of the states with respect to their inlet values for all time steps

    X_pred = X_pred.reshape((X_pred.shape[0], -1, n_measurements))
    x_in = x_in.reshape((X_pred.shape[0], -1, 1))
    dx = X_pred - x_in
    violations = np.zeros((X_pred.shape[0], n_measurements))
    for i, dx_i in enumerate(dx):
        delta_b = A @ dx_i
        # average over the constrains, positions are conserverd
        err = np.linalg.norm(delta_b, ord=2)
        violations[i] = err
    # only for concentrations violation is in mol/m3
    return violations


def calculate_state_physics_vio(
    meta_model: EtOxModel,
    sim_cfg: Dict,
    state_result_list: Union[np.ndarray, List[np.ndarray]],
    norm_measurements: bool = False,
) -> List[np.ndarray]:
    """
    Calculates the physics violation for a list of simulation results based on
    stoichiometric constraints.

    The L2-norm of the residual from the stoichiometric constraint is used as
    the physics violation metric.

    Args:
        model_cfg (Dict): Dictionary with model configuration, used to obtain the
            stoichiometric matrix 'A'.
        sim_cfg (Dict): Dictionary with simulation configuration, used to get
            state scaling factors (e.g., sim_cfg["states"]["scales"][0]).
        state_result_list (Union[np.ndarray, List[np.ndarray]]): List of arrays,
            where each array contains state simulation results. The shape of each
            item is assumed to be (n_trajectories, n_tsteps, n_states * n_measurements).
            It's expected that states are already reduced to measurement positions.
        norm_measurements (bool): If True, the violation is computed across all
            constraints and all measurement positions, resulting in a violation
            per time step and trajectory. If False, the violation is computed
            per measurement position. Defaults to False.

    Returns:
        List[np.ndarray]: A list of arrays, where each array contains the physics
            violations. The shape of each array is:
            - If norm_measurements=False: (n_trajectories, n_tsteps, n_measurements)
            - If norm_measurements=True: (n_trajectories, n_tsteps)
    """
    print("Calculating physics violations.")
    # Ensure state_result_list is a list of arrays
    if not isinstance(state_result_list, List):
        state_result_list = list(state_result_list)
    n_measurements = sim_cfg["narx"]["n_measurements"]
    boundary_cond = meta_model.get_bc_for_all_measurements(n_measurements=n_measurements)
    boundary_cond = boundary_cond.reshape((-1, n_measurements))
    element_species_matrix = meta_model.get_balance_constraint_matrix(include_temp_as_zero=True)  # Assumed function
    n_states = element_species_matrix.shape[1]
    scale_factor = sim_cfg["states"]["scales"][0]
    physics_violations = []
    for result in state_result_list:
        # result shape is (n_trajectories, n_tsteps, n_states * n_measurements)
        # Reshape to (n_trajectories, n_tsteps, n_states, n_measurements)
        # This assumes the last dimension was packed as [s1_m1, s2_m1, ..., sn_m1, s1_m2, ...]
        n_tsteps = result.shape[1]
        n_trajectories = result.shape[0]
        result_reshaped = result.reshape(n_trajectories, n_tsteps, n_states, n_measurements)
        # Calculate residual: A * ( (states - inlet) / scale )
        # A is (n_constraints, n_states). Operation is on the third axis.
        # violation shape: (n_trajectories, n_tsteps, n_constraints, n_measurements)
        difference = result_reshaped - boundary_cond
        residual = element_species_matrix @ ((difference) / scale_factor)
        # Reshape for norm calculation
        # Swap axes to get (n_trajectories, n_tsteps, n_measurements, n_constraints)
        violation = np.swapaxes(residual, -1, -2)
        if norm_measurements:
            # Flatten measurement and constraint dimensions for one violation per t-step
            # violation shape: (n_trajectories, n_tsteps, n_measurements * n_constraints)
            violation = violation.reshape(n_trajectories, n_tsteps, -1)
        # Calculate L2-norm over the last axis (constraints or (measurements * constraints))
        # physics_violation shape: (n_trajectories, n_tsteps, n_measurements) or (n_trajectories, n_tsteps)
        physics_violation = np.linalg.norm(violation, ord=2, axis=-1)
        physics_violations.append(physics_violation)
    return physics_violations


def calculate_prediction_physics_vio(model_cfg: Dict, results_list: List[Dict[str, MPCData]], keep: Optional[str] = None) -> np.ndarray:
    """Calculates the violation of stoichiometric constraints from MPC predictions.

    This function quantifies how well the predicted states from a Model
    Predictive Control (MPC) simulation adhere to the system's linear physical
    constraints. These constraints, representing principles like mass conservation,
    are derived from the null space of the stoichiometric matrix.

    The violation is calculated as the L2 norm (Euclidean norm) of the residual
    vector `b` in the equation `A @ x = b`. Here, `A` is the constraint matrix
    and `x` is the difference between the predicted state and the inlet conditions.
    Ideally, `b` should be a zero vector.

    The resulting metric is then averaged across different dimensions based on the
    `keep` parameter to allow for targeted analysis.

    Args:
        results_list (List[Dict[str, Any]]):
            A list containing the simulation results for each trajectory. Each
            element is a dictionary that must hold an 'mpc' object. This object
            is expected to have a `prediction` method to retrieve state data.
        keep (Optional[str], optional):
            Specifies how to average the final metric. Defaults to `None`.
            - "position": Averages over the prediction horizon and returns the
              violation for each measurement position.
            - "horizon": Averages over the measurement positions and returns the
              violation for each time step of the prediction horizon.
            - `None`: Averages over both positions and horizon, which returns a
              single scalar.

    Returns:
        np.ndarray:
            The calculated physics violation in [mol/m³]. The shape of the
            array depends on the `keep` argument:
            - `(n_measurements,)` if `keep="position"`.
            - `(horizon_length,)` if `keep="horizon"`.
            - A scalar if `keep=None`.

    Notes:
        This function depends on the variables `model_cfg` and `n_measurements`
        being defined in the calling scope.
        - `model_cfg` (dict): Must contain the key "stoiciometric_matrix".
        - `n_measurements` (int): The number of spatial measurement positions.
    """
    # Calculate the constraint matrix from the stoichiometric space
    A = get_stoic_matrix(model_cfg)

    time = results_list[0]["mpc"]["_time"]
    n_trajectories = len(results_list)
    inlet_vector = get_inlet_vector()
    n_states, horizon_length, n_scenarios = results_list[0]["mpc"].prediction(("_x", "x")).shape

    # Pre-allocate array for all predictions
    # Shape: (trajectory, simulation_step, scenario, horizon_step, state)
    predictions = np.zeros((n_trajectories, time.shape[0], n_scenarios, horizon_length, n_states))
    for trajectory in range(n_trajectories):
        for t_ind in range(predictions.shape[1]):
            predictions_at_t_ind = np.zeros((n_states, horizon_length, n_scenarios))
            for i, state_key in enumerate(sim_cfg["states"]["keys"]):
                print(results_list[trajectory]["mpc"].prediction(("_x", state_key), t_ind))
                # predictions_at_t_ind[i] = results_list[trajectory]["mpc"].prediction(("_x", state_key), t_ind)
                # TODO: Fix the querying of the states they are not lumped into a giant state "x" they are seperated by their keys.

            # predictions[trajectory, t_ind] = predictions_at_t_ind.T  # Transpose to (scenario, horizon, state)

    # Reshape to separate spatial measurement positions
    predictions = predictions.reshape(*predictions.shape[:-1], -1, n_measurements)

    # Calculate the L2 norm of the residual vector A @ (x_pred - x_inlet)
    # The norm is taken over the components of the residual vector (axis=-2)
    physics_violation = np.linalg.norm(A @ (predictions - inlet_vector), ord=2, axis=-2)

    # Average over simulation time steps and scenarios
    # Shape becomes (trajectory, horizon_step, measurement_position)
    physics_violation = physics_violation.mean(axis=(1, 2))

    if keep == "position":  # Average over the prediction horizon
        physics_violation = physics_violation.mean(axis=1)
    elif keep == "horizon":  # Average over the measurement positions
        physics_violation = physics_violation.mean(axis=-1)
    else:  # Average over both position and prediction horizon
        physics_violation = physics_violation.mean(axis=(1, -1))

    return physics_violation  # Unit is mol/m³


def calculate_intervall_width(
    surrogate_trajectory_dict: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Compute absolute interval widths between upper and lower surrogate trajectories.

    Args:
        surrogate_trajectory_dict (Dict[str, Dict[str, np.ndarray]]):
            Dictionary mapping surrogate names to a dict containing
            "upper" and "lower" trajectory arrays of identical shape.

    Returns:
        Dict[str, np.ndarray]:
            Dictionary mapping each surrogate name to its interval width array
            (absolute difference between upper and lower bounds).

    Raises:
        AssertionError:
            If "upper" or "lower" keys are missing, or if their array shapes differ.
    """
    print("---- Calculating intervall widths. ----")
    intervall_widths = {}
    for surrogate_key, surrogate_scenario_dict in surrogate_trajectory_dict.items():
        if not "upper" in surrogate_scenario_dict.keys() or not "lower" in surrogate_scenario_dict.keys():
            raise AssertionError("Upper and lower keys must be in the surrogate dict.")
        upper_bound = surrogate_scenario_dict["upper"]
        lower_bound = surrogate_scenario_dict["lower"]
        assert upper_bound.shape == lower_bound.shape, f"Upper states {upper_bound.shape}, lower states {lower_bound.shape} must have same shape."
        difference = upper_bound - lower_bound
        intervall_widths[surrogate_key] = np.abs(difference)
    # returned shape is {surrogate: np.ndarray (..., n_states)}
    return intervall_widths


def separate_into_state_by_slice(
    dict_with_metric: Dict[str, np.ndarray],
    state_slices: List[slice] = [slice(0, 20), slice(20, None)],
    scaling_factors: List[float] = None,
    apply: Optional[List[Callable]] = None,
    axes: List[Tuple[int]] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Split surrogate metric arrays into state-specific slices with optional scaling and function application.

    Args:
        dict_with_metric (Dict[str, np.ndarray]):
            Dictionary mapping surrogate names to metric arrays of shape
            (n_trajectories, n_time_steps, n_measurements, n_states). The states must be in the last dimension.
        state_slices (List[slice], optional):
            List of slices defining state blocks to extract. Defaults to [slice(0, 20), slice(20, None)].
        scaling_factors (List[float], optional):
            Scaling factors applied to each state slice. Defaults to ones.
        apply (Optional[List[Callable]], optional):
            List of functions applied along specified axes for each slice. Defaults to None.
        axes (List[Tuple[int]], optional):
            Axes along which to apply the corresponding function for each slice. Defaults to None.

    Returns:
        List[Dict[str, np.ndarray]]:
            List of dictionaries, one per state slice, mapping each surrogate name
            to the processed (sliced, scaled, optionally transformed) array.

    Raises:
        AssertionError:
            If the lengths of `state_slices`, `scaling_factors`, `apply`, and `axes` differ.
    """

    scaling_factors = scaling_factors or np.ones(len(state_slices))
    apply = apply or [None] * len(state_slices)
    axes = axes or [None] * len(state_slices)
    # the functions supplied in the List apply are 1D -> 1D Functions that are applied to the last column of the state slice.
    assert len(apply) == len(state_slices) == len(axes) == len(scaling_factors), "The number of functions to apply must equal the number of blocks that are sliced from the states."
    # the incoming dict is of form {"surrogate_key": np.ndarray (n_trajects, n_time_steps, n_states)}
    # TODO: This could be part of the data structurizer
    separated = []
    for sl, fun, ax, scale in zip(state_slices, apply, axes, scaling_factors):
        separated.append(
            {surr_key: np.apply_over_axes(a=metric[..., sl], func=fun, axes=ax).squeeze() / scale if fun else metric[..., sl].squeeze() / scale for surr_key, metric in dict_with_metric.items()}
        )
        # This line is 100% self made :). My most beautiful dense line of code so far.
    return separated


def main():
    path_to_results = "/Users/jandavidridder/Desktop/Masterarbeit/PYTHON/MYCODE/results"
    loaded_results = []
    names = []
    for file_name in os.listdir(path_to_results):
        if file_name.endswith(".pkl"):
            names.append(file_name.replace(".pkl", ""))
            loaded_results.append(load_results(f"{path_to_results}/{file_name}"))

    constr_viol = calculate_mean_constraint_vio(results_list=loaded_results)
    performance_results = {
        "surrogate": names,
        "mean selectivity": calculate_mean_selectivity(results_list=loaded_results),
        "mean contr vio T": constr_viol["T"],
        "mean contr vio X": constr_viol["X"],
        # "phy vio": calculate_prediction_physics_vio(loaded_results, keep=None),
        "mean cpu time": calculate_mean_cputime(loaded_results),
    }

    as_data_frame = pd.DataFrame(performance_results)
    print(as_data_frame)
    # print(as_data_frame.to_latex())


if __name__ == "__main__":
    CURR_DIR = os.path.dirname(__file__)
    ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
    CONFIG_NAME = "etox_control_task.yaml"
    CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIR, "configs", CONFIG_NAME))
    MODEL_CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIR, "models", "EtOxModel", "EtOxModel.yaml"))
    sys.path.append(ROOT_DIR)

    n_measurements = 4
    n_discretization = 128
    with open(CONFIG_PATH, "r") as f:
        sim_cfg = yaml.safe_load(f)
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_cfg = yaml.safe_load(f)

    structurizer = DataStructurizer(
        n_measurements=n_measurements,
        time_horizon=8,
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        tvp_keys=sim_cfg["tvps"]["keys"],
    )
    vis = Visualizer(sim_cfg, cmap="magma")
    main()
