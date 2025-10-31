import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from do_mpc.data import save_results
from do_mpc.model import Model
from routines.insights import plot_rhs_jac
from routines.setup_routines import configure_simulator, make_simulator_tvp_fun, set_p_fun
from routines.utils import NumpyEncoder
from tqdm import tqdm


def simulate(
    simulation_cfg: Dict,
    n_time_steps: int,
    do_mpc_model: Model,
    tvp_signals: np.ndarray,
    initial_states: np.ndarray,
    input_signals: np.ndarray,
    model_parameters: np.ndarray = None,
    index: Optional[np.ndarray] = None,
    process_name: str = "",
    save_dir: str = None,
    result_name: str = "result",
    save_as: str = "json",
    save_variable_types: List[str] = ["_x", "_u", "_tvp"],
    integration_opts: Optional[Dict] = None,
) -> Union[np.ndarray]:
    assert n_time_steps <= input_signals.shape[1], f"The maximum number of time steps to simulate is {input_signals.shape[1]} you have {n_time_steps}."

    if initial_states.ndim < 2:
        initial_states = np.expand_dims(initial_states, axis=0)  # add batch dimension
    if input_signals.ndim < 3:
        input_signals = np.expand_dims(input_signals, axis=0)  # add batch dimension
    if tvp_signals.ndim < 3:
        tvp_signals = np.expand_dims(tvp_signals, axis=0)
    if index is None:
        index = np.arange(initial_states.shape[0])
    assert (
        initial_states.shape[0] == input_signals.shape[0] == tvp_signals.shape[0] == index.shape[0]
    ), f"Number of initial states batches {initial_states.shape[0]} and indices {index.shape[0]} must match the number of input signals batches {input_signals.shape[0]} and tvp signal batches {tvp_signals.shape[0]}."
    n_trajectories = input_signals.shape[0]

    if model_parameters is None:
        model_parameters = np.zeros((1, 1))
        if do_mpc_model.n_p > 0:
            raise ValueError("You must provide parameters for the do mpc model.")
    else:
        if model_parameters.ndim < 2:
            model_parameters = np.expand_dims(axis=0)
            model_parameters = np.repeat(axis=0, repeats=n_trajectories)
            # add batch dimension and duplicate constant parameters for all batches
        else:
            assert (
                model_parameters.shape[0] == n_trajectories
            ), f"The number of kinetic parameter combinatons {model_parameters.shape[0]} and the number of input trajectories {n_trajectories} must match."

    if save_dir is None:
        print("Warning the simulation results will not be saved. Provide a directory for saving.")
    # simulate the given do-mpc model in a open loop for all given (initial_states, input_signals) and parameter combinations
    # the outer loop should be the parmeters as the simulator object needs to be recreated every time to run different parameters
    iterable = zip(index, initial_states, input_signals, tvp_signals, model_parameters)
    iterable = tqdm(iterable, desc="Simulating an open loop model.", total=n_trajectories) if process_name == "Proc 0" else iterable
    previous_parameter_combination = np.random.rand(*model_parameters[0].shape) if model_parameters is not None else None
    previous_tvp_signal = np.random.rand(*tvp_signals[0].shape)

    for i, x0, input_signal, tvp_signal, parameter_combination in iterable:
        if not np.allclose(previous_parameter_combination, parameter_combination) or not np.allclose(previous_tvp_signal, tvp_signal):

            # to set a new parameter combination, recreate the simulator object
            simulator = configure_simulator(simulation_cfg, do_mpc_model, integration_opts=integration_opts)
            # plot_rhs_jac(
            #     model=do_mpc_model,
            #     states=x0,
            #     inputs=input_signal[0],
            #     tvps=tvp_signal[0],
            #     params=parameter_combination,
            #     # save_path="/Users/jandavidridder/Desktop/Masterarbeit/src/experiments/001_certain_open_loop_kpis/2025-10-12/insights",
            # )

            if do_mpc_model.n_tvp > 0:
                tvp_template = simulator.get_tvp_template()
                tvp_fun = make_simulator_tvp_fun(
                    simulation_time_step=simulation_cfg["simulation"]["t_step"],
                    tvp_template=tvp_template,
                    tvp_traj=tvp_signal,
                    tvp_key=simulation_cfg["tvps"]["keys"][0],
                )
                simulator.set_tvp_fun(tvp_fun)
            if do_mpc_model.n_p > 0:
                set_p_fun(simulator, params=parameter_combination)
            simulator.setup()

        previous_tvp_signal = tvp_signal
        previous_parameter_combination = parameter_combination
        simulator.reset_history()
        simulator.x0 = x0
        simulator.set_initial_guess()

        try:
            start = time.perf_counter()
            for t in range(n_time_steps):
                x_next = simulator.make_step(u0=input_signal[t].reshape((-1, 1)))
            stop = time.perf_counter()
        except Exception as e:
            print(f"Simulation failed with error: {e}")
            continue
        if save_as == "pkl":
            save_results([simulator], result_path=f"{save_dir}/")
            continue

        ind = 1
        ext_result_name = result_name
        while os.path.isfile(f"{save_dir}/{ext_result_name}.{save_as}"):
            ext_result_name = f"{ind:03d}_{result_name}"
            ind += 1
        complete_file_name = os.path.join(save_dir, f"{ext_result_name}")
        meta_data_i = {"index": i, "t_wall_total": stop - start}
        simulator.data.set_meta(**meta_data_i)

        if save_as == "json":
            with open(f"{complete_file_name}.json", "w") as f:
                json_result = simulator.data.export()
                json_result.update({"meta_data": meta_data_i})
                f.write(json.dumps(json_result, indent=4, cls=NumpyEncoder))
        elif save_as == "npy":
            extracted_results = []
            for var_type in save_variable_types:
                extracted_results.append(simulator.data[var_type])
            result = np.concat(extracted_results, axis=-1)
            np.save(f"{complete_file_name}.npy", result)


def save_data(model_name: str, data: np.ndarray, run_id: str, data_dir: str = None):
    if data_dir is None:
        data_dir = "/Users/jandavidridder/Desktop/Masterarbeit/data"
    path_to_file = os.path.join(data_dir, model_name, run_id)
    os.makedirs(name=data_dir, exist_ok=True)
    np.save(path_to_file, data)
    print(f"Data saved to {path_to_file}")


def generate_random_ramp_signal(
    feature_bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
    num_steps: int,
    tau: int,
    randomize: bool = True,
    batch_size: int = 1,
    time_step: float = 0.01,
    hold_time_range: Tuple[float, float] = (1, 2),
    ramp_time_range: Tuple[float, float] = (0.5, 0.5),
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generates batched, random, piecewise ramp signals for multiple features (n_features)
    using NumPy vectorization.

    Args:
        levels (list of float OR list of lists of float):
            If list[float]: Possible stationary levels for ALL features.
            If list[list[float]]: Possible levels for each feature (len(levels) = n_features).
        num_steps (int): Total number of time steps to generate for each batch.
        tau (int): Time constant (seconds) used for scaling hold and ramp durations.
        batch_size (int): Number of signals (batch_size) to generate.
        time_step (float): Time step size (seconds).
        initial_levels (float or list of float or None): Starting level(s).
            If float/None, applies to all features. If list, must match n_features.
        hold_time_range (tuple): (min, max) scale factor for tau during hold.
        ramp_time_range (tuple): (min, max) scale factor for tau during ramp.
        seed (int or None): Random seed for reproducibility.

    Returns:
        np.ndarray: Generated signals of shape (batch_size, num_steps, n_features).
    """
    rng = np.random.default_rng(seed)
    if isinstance(feature_bounds, Tuple):
        feature_bounds = [feature_bounds]

    n_features = len(feature_bounds)
    min_hold_time_s = hold_time_range[0] * tau
    max_hold_time_s = hold_time_range[1] * tau
    min_ramp_time_s = ramp_time_range[0] * tau
    max_ramp_time_s = ramp_time_range[1] * tau

    min_hold_steps = int(min_hold_time_s / time_step)
    min_ramp_steps = int(min_ramp_time_s / time_step)
    min_event_steps = min_hold_steps + min_ramp_steps

    max_events = num_steps // min_event_steps + 2

    if randomize:
        n_samples = batch_size
        duplicates = 1
    else:
        n_samples = 1
        duplicates = batch_size

    signals = np.zeros((n_samples, num_steps, n_features))

    for f in range(n_features):
        current_feature_bounds = feature_bounds[f]
        start_levels = rng.uniform(high=current_feature_bounds[-1], low=current_feature_bounds[0], size=n_samples)
        all_levels = np.zeros((n_samples, max_events + 1))
        all_levels[:, 0] = start_levels
        for i in range(max_events):
            next_levels = np.zeros(n_samples)
            for b in range(n_samples):
                next_level = rng.uniform(high=current_feature_bounds[-1], low=current_feature_bounds[0])
                next_levels[b] = next_level

            all_levels[:, i + 1] = next_levels

        hold_times_s = rng.uniform(min_hold_time_s, max_hold_time_s, size=(n_samples, max_events))
        hold_steps = np.floor(hold_times_s / time_step).astype(int)

        ramp_times_s = rng.uniform(min_ramp_time_s, max_ramp_time_s, size=(n_samples, max_events))
        ramp_steps = np.floor(ramp_times_s / time_step).astype(int)

        total_steps_per_event = hold_steps + ramp_steps
        event_end_indices = np.cumsum(total_steps_per_event, axis=1)
        event_start_indices = np.hstack([np.zeros((n_samples, 1), dtype=int), event_end_indices[:, :-1]])

        for i in range(max_events):

            start_index = event_start_indices[:, i]
            end_index = event_end_indices[:, i]

            hold_len = hold_steps[:, i]
            ramp_len = ramp_steps[:, i]

            level_from = all_levels[:, i]
            level_to = all_levels[:, i + 1]
            for b in range(n_samples):

                if start_index[b] >= num_steps:
                    continue

                current_start = start_index[b]
                hold_end = min(current_start + hold_len[b], num_steps)
                signals[b, current_start:hold_end, f] = level_from[b]
                ramp_start = hold_end
                ramp_end = min(ramp_start + ramp_len[b], num_steps)

                if ramp_start < num_steps:
                    current_ramp = np.linspace(level_from[b], level_to[b], ramp_len[b], endpoint=False)
                    fill_len = ramp_end - ramp_start
                    signals[b, ramp_start:ramp_end, f] = current_ramp[:fill_len]

    signals = np.repeat(signals, axis=0, repeats=duplicates)
    return signals[:, :num_steps, :]


def generate_initial_state(n_batches: int = 1) -> np.ndarray:
    # TODO: Randomly generate feasible intial states.
    initial_state = np.load("/Users/jandavidridder/Desktop/Masterarbeit/src/PYTHON/MYCODE/models/EtOxModel/initial_state.npy")
    initial_state = np.expand_dims(initial_state, axis=0)
    initial_state = np.repeat(initial_state, axis=0, repeats=n_batches)
    return initial_state


# def generate_excitation_signal(
#     n_time_steps: int,
#     time_step: float,
#     levels: Union[List, np.ndarray],
#     system_time_constants: List[Dict],
#     min_overall_signal_dwell_time: float = None,
#     max_overall_signal_dwell_time: float = None,
#     base_value: float = None,
#     seed: int = None,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Generates a multi-level, randomly switched excitation signal,
#     adapting its dwell time based on the system's longest relevant time constant.

#     Args:
#         n_time_steps (int): The total number of points in the signal.
#         time_step (float): Time step (dt) for the signal, in time units.
#         levels (list or np.ndarray): List of possible amplitude levels for the signal.
#                                      If base_value is provided, these are treated as
#                                      deviations from base_value.
#         system_time_constants (list): A list of dictionaries, each defining a system time constant.
#                                       Each dict must have:
#                                       - 'type' (str): 'fixed' for constant time constants (e.g., reaction kinetics)
#                                                       'residence_time' for input-dependent (L/u type)
#                                       - 'value' (float): The constant time in seconds if 'type' is 'fixed'.
#                                                          This acts as the 'L_value' if 'type' is 'residence_time'.
#                                       - 'scaling_factor' (float): Multiplier for this specific time constant
#                                                                   to determine the signal's dwell time.
#                                                                   (e.g., how many of this tau should the signal wait).
#                                       Example: [{'type': 'fixed', 'value': 600, 'scaling_factor': 1.0},
#                                                 {'type': 'residence_time', 'value': 100, 'scaling_factor': 5.0}]
#         min_overall_signal_dwell_time (float, optional): An absolute minimum signal dwell time,
#                                                          never falling below this.
#         max_overall_signal_dwell_time (float, optional): An absolute maximum signal dwell time,
#                                                          never exceeding this.
#         base_value (float, optional): A base value around which levels fluctuate.
#                                       If None, levels are used absolutely.
#         seed (int, optional): Seed for the random number generator for reproducibility.

#     Returns:
#         tuple: (time_vector, signal_vector)
#                time_vector (np.array): Time points.
#                signal_vector (np.array): Generated input signal.
#     """

#     rng = np.random.default_rng(seed)

#     if not isinstance(levels, (list, np.ndarray)) or not levels:
#         raise ValueError("Levels must be a non-empty list or NumPy array.")

#     if not isinstance(system_time_constants, list) or not system_time_constants:
#         raise ValueError("system_time_constants must be a non-empty list of dictionaries.")

#     for tc_def in system_time_constants:
#         if not isinstance(tc_def, dict) or "type" not in tc_def or "value" not in tc_def or "scaling_factor" not in tc_def:
#             raise ValueError("Each item in system_time_constants must be a dict with 'type', 'value', and 'scaling_factor'.")
#         if tc_def["type"] not in ["fixed", "residence_time"]:
#             raise ValueError(f"Unknown time constant type: {tc_def['type']}. Must be 'fixed' or 'residence_time'.")

#     num_levels = len(levels)
#     if num_levels == 0:
#         raise ValueError("Levels list cannot be empty.")

#     total_duration = n_time_steps * time_step
#     time_vector = np.linspace(0, total_duration, n_time_steps, endpoint=False)
#     signal_vector = np.zeros((n_time_steps))

#     if base_value is not None:
#         actual_levels = [base_value + l for l in levels]
#     else:
#         actual_levels = list(levels)

#     current_time_idx = 0
#     current_level = rng.choice(actual_levels)

#     while current_time_idx < n_time_steps:
#         # Calculate all relevant system time constants for the current level
#         current_system_tau_values = []
#         for tc_def in system_time_constants:
#             tc_value = tc_def["value"]  # This is L for residence_time, or fixed tau for fixed
#             scaling_factor = tc_def["scaling_factor"]
#             current_tau = tc_value * scaling_factor
#             current_system_tau_values.append(current_tau)

#         # The signal's dwell time should be based on the longest (dominant) time constant
#         if not current_system_tau_values:  # Should not happen with validation, but for safety
#             dominant_tau = time_step  # Default to minimal if no time constants are defined
#         else:
#             dominant_tau = max(current_system_tau_values)

#         # Add some randomness to the signal's dwell time around the dominant tau
#         random_factor = rng.uniform(0.8, 1.2)  # +/- 20% variation for broadband excitation
#         actual_signal_dwell_duration = dominant_tau * random_factor

#         # Apply overall absolute min/max bounds for the signal's dwell time
#         if min_overall_signal_dwell_time is not None:
#             actual_signal_dwell_duration = max(actual_signal_dwell_duration, min_overall_signal_dwell_time)
#         if max_overall_signal_dwell_time is not None:
#             actual_signal_dwell_duration = min(actual_signal_dwell_duration, max_overall_signal_dwell_time)

#         # Ensure dwell duration is at least one time step
#         actual_signal_dwell_duration = max(actual_signal_dwell_duration, time_step)

#         num_dwell_points = int(round(actual_signal_dwell_duration / time_step))
#         num_dwell_points = max(1, num_dwell_points)  # Ensure at least 1 point

#         end_idx_for_level = min(current_time_idx + num_dwell_points, n_time_steps)
#         signal_vector[current_time_idx:end_idx_for_level] = current_level

#         current_time_idx = end_idx_for_level

#         # Choose the next level, ideally different from the current one
#         if num_levels > 1:
#             possible_next_levels = [l for l in actual_levels if l != current_level]
#             if not possible_next_levels:  # If all levels are the same or only one remains
#                 current_level = rng.choice(actual_levels)
#             else:
#                 current_level = rng.choice(possible_next_levels)
#         elif actual_levels:  # Only one level defined
#             current_level = actual_levels[0]

#     return time_vector, signal_vector


# def generate_excited_input_signals(
#     n_time_steps: int,
#     t_step: float,
#     t_constants: Dict,
#     input_levels: np.array,
#     scaling_factors: np.array = None,
#     seeds: np.array = None,
# ) -> np.array:
#     """Creates a excited signals for every input of the system.

#     Args:
#         n_time_steps (int): Number of time steps for which the signal is generated.
#         x0 (np.array): Initial state of the system to calculate possible time constants.
#         seed (int, optional): Random seed to sample the signal levels. Defaults to 42.

#     Returns:
#         np.array: Excited signal inputs for every time step and input. It is of shape (n_time_steps, n_inputs).
#     """

#     u = np.zeros((n_time_steps, len(input_levels)))
#     if scaling_factors is None:
#         scaling_factors = np.ones(len(t_constants.values()))

#     time_constants = [
#         {
#             "type": "fixed",
#             "value": t_constants["t_r1"],
#             "scaling_factor": scaling_factors[0],
#         },
#         {
#             "type": "fixed",
#             "value": t_constants["t_r2"],
#             "scaling_factor": scaling_factors[1],
#         },
#         {
#             "type": "fixed",
#             "value": t_constants["t_h"],
#             "scaling_factor": scaling_factors[2],
#         },
#         {
#             "type": "fixed",
#             "value": t_constants["t_u"],
#             "scaling_factor": scaling_factors[3],
#         },
#     ]
#     for i, levels in enumerate(input_levels):
#         _, u[:, i] = generate_excitation_signal(
#             n_time_steps=n_time_steps,
#             time_step=t_step,
#             levels=list(levels),
#             system_time_constants=time_constants,
#             seed=seeds[i] if seeds is not None else None,
#         )
#     return u
