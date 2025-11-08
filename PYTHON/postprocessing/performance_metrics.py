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
) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Calculates the time-dependent MSE for a batch of surrogate model predictions.

    It iterates through a dictionary of prediction and test data pairs (keyed by
    surrogate name) and applies `calculate_state_mse` to each pair.

    Args:
        state_prediction_data: Dictionary where keys are surrogate names (str)
                               and values are predicted state trajectories (np.ndarray).
                               Shape of values: (batch, time, features) or (time, features).
        state_test_data: Dictionary of corresponding ground truth state trajectories.
                         Keys must match those in `state_prediction_data`.
        state_scale: Scaling factor passed to `calculate_state_mse`. Defaults to 1.
        keep: Passed to `calculate_state_mse`. Defaults to "time".

    Returns:
        Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]: A dictionary
        where keys are surrogate names and values are the resulting MSE(t).
        The value format depends on the shape of the input data (see `calculate_state_mse`):
        - Tuple[np.ndarray, np.ndarray] (mean, std) if batched input.
        - np.ndarray (MSE) if single trajectory input.
    """
    surrogate_mses = {}

    # Iterate over each surrogate's prediction data
    for surr_key, prediction_trajectories in state_prediction_data.items():
        # Retrieve the corresponding ground truth data
        if surr_key not in state_test_data:
            raise KeyError(f"Missing test data for surrogate key: '{surr_key}' in state_test_data.")

        test_trajectories = state_test_data[surr_key]

        # Calculate the MSE for this specific surrogate
        mse_result = calculate_state_mse(
            state_test_data=test_trajectories,
            state_prediction_data=prediction_trajectories,
            state_scale=state_scale,
            keep=keep,
        )
        surrogate_mses[surr_key] = mse_result

    return surrogate_mses


def calculate_state_mse(state_test_data: np.ndarray, state_prediction_data: np.ndarray, state_scale: float = 1, keep: str = "time") -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Calculates the time-dependent Mean Squared Error (MSE) between true and predicted
    state trajectories, normalized by the number of features.

    mse(t) = 1/N_features * ||(test(t) - pred(t)) / scale||_2^2

    Args:
        state_test_data: Ground truth data. Shape: (batch, time, features) or (time, features).
        state_prediction_data: Predicted data. Shape: (batch, time, features) or (time, features).
        state_scale: Factor to scale the difference by. Defaults to 1.
        keep: Specifies which dimension to retain (only 'time' is implemented for batched output).

    Returns:
        If batched (2D input): (mean_mse_over_batch, std_mse_over_batch) vs. time.
        If single trajectory (1D input): Array of MSE values vs. time.
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
    """
    Calculates the mean constraint violation for 'X' (conversion) and 'T' (temperature)
    across a list of simulation results.

    For points violating the constraint, the violation distance is measured.
    Points satisfying the constraint are counted as zero violation.

    The mean violation for 'X' is the time-averaged relative violation across the
    single final measurement point. The mean violation for 'T' is the time-averaged
    L2 norm of the relative violation across all discretization points.

    Args:
        results: A single dictionary or list of dictionaries containing simulation
                 results. Expected keys for violation calculation:
                 - 'simulator': {'_aux', 'X'} for conversion 'X'.
                 - 'simulator': {'_x', 'T'} for temperature 'T'.
        X_min: The minimum allowed value for the conversion variable 'X'.
        T_max: The maximum allowed value for the temperature variable 'T'.

    Returns:
        Dict[str, np.ndarray]: A dictionary with keys "X" and "T". Each value is
        a 1D array of mean constraint violations corresponding to the input
        list of simulation results.
    """
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
