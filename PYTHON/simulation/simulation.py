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
    physical_params: Optional[np.ndarray] = None,
    index: Optional[np.ndarray] = None,
    process_name: Optional[str] = "",
    save_dir: Optional[str] = None,
    result_name: Optional[str] = "result",
    save_as: Optional[str] = "json",
    save_variable_types: Optional[List[str]] = ["_x", "_u", "_tvp"],
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

    if physical_params is None:
        physical_params = np.zeros((1, 1))
        if do_mpc_model.n_p > 0:
            raise ValueError("You must provide parameters for the do mpc model.")
    else:
        if physical_params.ndim < 2:
            physical_params = np.expand_dims(axis=0)
            physical_params = np.repeat(axis=0, repeats=n_trajectories)
            # add batch dimension and duplicate constant parameters for all batches
        else:
            assert (
                physical_params.shape[0] == n_trajectories
            ), f"The number of kinetic parameter combinatons {physical_params.shape[0]} and the number of input trajectories {n_trajectories} must match."

    # simulate the given do-mpc model in a open loop for all given (initial_states, input_signals) and parameter combinations
    # the outer loop should be the parmeters as the simulator object needs to be recreated every time to run different parameters
    iterable = zip(index, initial_states, input_signals, tvp_signals, physical_params)
    iterable = tqdm(iterable, desc="Simulating an open loop model.", total=n_trajectories) if process_name == "Proc 0" else iterable
    previous_parameter_combination = np.random.rand(*physical_params[0].shape) if physical_params is not None else None
    previous_tvp_signal = np.random.rand(*tvp_signals[0].shape)
    results_concat = []

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
        meta_data_i = {"index": i, "t_wall_total": stop - start}
        simulator.data.set_meta(**meta_data_i)

        if save_as in ["npy", "json"]:
            ind = 1
            ext_result_name = result_name
            while os.path.isfile(f"{save_dir}/{ext_result_name}.{save_as}"):
                ext_result_name = f"{ind:03d}_{result_name}"
                ind += 1
            complete_file_name = os.path.join(save_dir, f"{ext_result_name}")

        if save_as == "pkl":
            save_results([simulator], result_path=f"{save_dir}/")
            continue
        elif save_as == "return" or save_as == "npy":
            extracted_results = []
            for var_type in save_variable_types:
                extracted_results.append(simulator.data[var_type])
            extracted_results = np.concat(extracted_results, axis=-1)
            if save_as == "npy":
                np.save(f"{complete_file_name}.npy", extracted_results)
            else:
                results_concat.append(extracted_results)
            continue
        elif save_as == ".json":
            with open(f"{complete_file_name}.json", "w") as f:
                json_result = simulator.data.export()
                json_result.update({"meta_data": meta_data_i})
                f.write(json.dumps(json_result, indent=4, cls=NumpyEncoder))
        else:
            raise NotImplementedError(f"The save as type {save_as} is not implemented.")


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
