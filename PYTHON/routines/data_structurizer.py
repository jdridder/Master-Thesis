import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from casadi import MX, SX, vertcat
from numpy.lib.stride_tricks import sliding_window_view
from routines.utils import NumpyEncoder


class DataStructurizer:
    """A class to simplify the handling of the input vectors for NARX Models.
    The vector is flattened, where the time horizon is stacked into the feature dimension.
    The order hierachy is (state, measurement_position, time_horizon)(input, time_horizon)(tvp, time_horizon).
    It can handle data matrices, that have a batch size and time trajectories as well as single vectors.
    The precise scheme can be obtained from the list self.all_expanded_keys.
    The Structurizer converts between two base classes: The data-matrix and the NARX input vector with the time horizon stacked.
    """

    def __init__(
        self,
        n_measurements: int,
        n_initial_measurements: int,
        time_horizon: int,
        state_keys: List[str],
        input_keys: List[str],
        tvp_keys: List[str],
        aux_keys: Optional[List[str]] = [],
    ):

        self.n_measurements = n_measurements
        self.n_initial_measurements = n_initial_measurements
        self.time_horizon = time_horizon
        self.state_keys = state_keys
        self.n_states = len(self.state_keys)
        self.input_keys = input_keys
        self.n_inputs = len(self.input_keys)
        self.tvp_keys = tvp_keys
        self.n_tvps = len(self.tvp_keys)
        self.aux_keys = aux_keys
        self.expandend_state_keys = self.expand_state_keys_measurements(state_keys)  # expands the keys for all measurements
        self.all_expanded_keys = self.expand_keys_in_time()  # expands the keys for all time points in the moving window
        self.key_to_idx = {key: i for i, key in enumerate(self.all_expanded_keys)}
        self.dompc_keys, self.t0_indizes = self.get_dompc_scheme()
        self.rom_parameters = {}

    def load_from_cfg(cfg: Dict):
        return DataStructurizer(
            n_initial_measurements=cfg["simulation"].get("N_finite_diff"),
            n_measurements=cfg["narx"].get("n_measurements"),
            time_horizon=cfg["narx"].get("time_horizon"),
            state_keys=cfg["states"].get("keys"),
            input_keys=cfg["inputs"].get("all_keys"),
            tvp_keys=cfg["tvps"].get("keys"),
            aux_keys=cfg["aux"].get("keys"),
        )

    def get_key_to_reduced_data_idx(self) -> Dict:
        """Returns the key index pairs fot the feature column of the data matrix with reduced measurements."""
        data_keys = self.expandend_state_keys + self.input_keys + self.tvp_keys
        key_to_idx_map = {key: i for i, key in enumerate(data_keys)}
        return key_to_idx_map

    def expand_state_keys_measurements(self, unique_state_keys: List[str]) -> List[str]:
        """Creates the state keys for each measurement point."""
        state_keys = []
        for state_key in unique_state_keys:
            for i in range(self.n_measurements):
                state_keys.append(f"{state_key}_{i}")
        return state_keys

    def expand_keys_in_time(self) -> List[str]:
        """Multiplies the keys for each instant of the backward time horizon of every state, input and time varying parameter.
        Args:
            keys (List[str]): List of state, input and tvp keys.
        Returns:
            List[str]: Expanded list of state, input and tvp keys for every time step in the moving window.
        """
        expanded_keys = []
        for key_list in [self.expandend_state_keys, self.input_keys, self.tvp_keys]:
            for t_lag in range(0, -self.time_horizon, -1):
                for key in key_list:
                    expanded_keys.append(f"{key}_t{t_lag}")
        return expanded_keys

    @property
    def vector_size(self) -> int:
        return len(self.all_expanded_keys)

    def reduce_measurements(self, vector: Union[np.ndarray, SX], n_initial_measurements: int = None):
        """Reduces the amount of measurement points in z direction of the states to the specified number of measurements.
        The measurements are equidistant leaving out the measurement at the very beginning of the reactor.
        """
        n_initial_measurements = n_initial_measurements or self.n_initial_measurements
        reduced_measurement_mask = []
        step = int(n_initial_measurements // self.n_measurements)
        for s, _ in enumerate(self.state_keys):
            for i in range(self.n_measurements):
                idx_initial_vector = ((i + 1) * step) + s * n_initial_measurements - 1
                reduced_measurement_mask.append(idx_initial_vector)

        if isinstance(vector, np.ndarray):
            reduced_state_vector = vector[..., reduced_measurement_mask]
            # readd the resiual of the vector that may contain inputs and tvps
            start = n_initial_measurements * len(self.state_keys)
            return np.concatenate([reduced_state_vector, vector[..., start:]], axis=-1)
        elif isinstance(vector, SX):
            # casadi SX does not allow batching. Thus the shape of the vector will be (n_states * n_initial_measurements)
            return vector[reduced_measurement_mask]

    def _time_series_from_data(self, full_data_matrix: np.ndarray) -> np.ndarray:
        """Turns the full data matrix to a time series data matrix with shape (batch_size, time_steps, features, time_horizon).
        Args:
            states_matrix (np.ndarray): State data in the form (batch_size, time_steps, features) or (time_steps, features)
        Returns:
            np.ndarray: Time series matrix in the shape (batch_size, time_steps, features, time_horizon).
        """
        assert full_data_matrix.shape[-2] >= self.time_horizon, "Not enough time points to fill the time horizon."
        time_series_data = sliding_window_view(full_data_matrix, window_shape=self.time_horizon, axis=-2)
        # reverse the last dimesion to let it increase from current to previous to past
        return time_series_data[..., ::-1]

    def _stack_time_series(self, time_series_data: np.ndarray) -> np.ndarray:
        """Stacks time series structured data with shape (batch_size, time_steps, features, time_horizon) and stacks into a time series vector of shape (batch_size, time_steps-time_horizon, features*time_horizon).
        To have grouped blocks by features that are stacked after time points, the order "F" is used.
        The time series are stacked such that the most recent time points appear at the top and decrease chronologically to the bottom.
        """
        start_input = len(self.state_keys) * self.n_measurements
        start_tvp = start_input + len(self.input_keys)
        grouped_features = (
            time_series_data[..., :start_input, :],
            time_series_data[..., start_input:start_tvp, :],
            time_series_data[..., start_tvp:, :],
        )
        time_stacked_feature_groups = []
        for feature_group in grouped_features:
            stacked = feature_group.reshape((*feature_group.shape[:-2], -1), order="F")
            time_stacked_feature_groups.append(stacked)
        return np.concatenate(time_stacked_feature_groups, axis=-1)

    def _time_series_from_stacked(self, stacked: np.ndarray) -> np.ndarray:
        """Reverts the stacking to yield the data as time series matrix with shape (batch_size, time_steps, features, time_horizon).
        The time series is represented by the last axis."""
        start_input = len(self.state_keys) * self.n_measurements * self.time_horizon
        start_tvp = start_input + len(self.input_keys) * self.time_horizon
        grouped_features = stacked[..., :start_input], stacked[..., start_input:start_tvp], stacked[..., start_tvp:]
        time_series_feature_groups = []
        for feature_group in grouped_features:
            stacked = feature_group.reshape((*feature_group.shape[:-1], -1, self.time_horizon), order="F")
            time_series_feature_groups.append(stacked)
        return np.concatenate(time_series_feature_groups, axis=-2)

    def stack_data(self, full_data_matrix: np.ndarray) -> np.ndarray:
        """Turns the full data matrix to a stacked time series vector with shape (batch_size, time_steps-time_horizon, features*time_horizon).
        It is stacked such that the time runs from current to past from top to bottom of the vector."""
        time_series_data = self._time_series_from_data(full_data_matrix)
        return self._stack_time_series(time_series_data)

    def make_training_XY(self, data_matrix: np.ndarray) -> Tuple[np.ndarray]:
        """Takes a data_matrix of shape (batch_size, time_steps, features) and turns it into X and Y for NARX training. X, Y are structured in a time series manner with the time horizon in the last dimension.
        They are of shape (n_datapoints, features, time_horizon).
        For X all features of the data are considered. Y is reduced only to the states.
        """
        stop = len(self.state_keys) * self.n_measurements
        X, Y = data_matrix, data_matrix[..., :stop]
        X, Y = self._time_series_from_data(X), self._time_series_from_data(Y)
        # apply the autoregressive shift: drop first y and last x in the time dimension
        # for Y only keep the next time step which is at index 0 in the time_horizon dimension shape: (batch_size, time_steps, features, time_horizon)
        X, Y = X[..., :-1, :, :], Y[..., 1:, :, 0]  # exclude the current time step for x and the last time step for Y
        X = X[..., ::-1, :, :]
        Y = Y[..., ::-1, :]  # revert the time dimension, so the most recent time point appears at the top, for training this is redundant as data is randomized, this is for readability
        return self._stack_time_series(X), Y

    def get_states_from_vector(self, vector: np.ndarray) -> np.ndarray:
        """Extracts all states at all positions and reconstructs the time series from the stacked vector.
        It is returned in the shape (batch_size, time_steps, states, time_horizon)
        Args:
            vector (np.ndarray): NARX vector. The features must be contained in the last axis. For do-mpc or casadi use, reshape the vector accordingly.
        Returns:
            np.ndarray: The states of the NARX vector with shape (n_states, time_horizon).
        """
        as_time_series = self._time_series_from_stacked(vector)
        start = 0
        end = len(self.state_keys) * self.n_measurements
        return as_time_series[..., start:end, :]

    def get_states_at_measurement_from_vector(self, vector: np.ndarray, measurement: int) -> np.ndarray:
        """Returns the states at a specific measurement index for all time points of the window.
        It has shape (batch_size, time_steps, n_states, time_window).
        Args:
            vector (np.ndarray): Stacked autoregressive vector in shape (batch_size, time_steps, features*time_window)
            measurement (int): Index of the measurement.
        Returns:
            np.ndarray: States at the measurement location with shape (batch_size, time_steps, n_states, time_window).
        """
        assert measurement < self.n_measurements, "The measurement position is too high for the count of measurement positions."
        n_unique_states = len(self.state_keys)
        as_time_series = self.get_states_from_vector(vector)
        states_at_measurement = np.zeros((*as_time_series.shape[:-2], n_unique_states, self.time_horizon))
        for s in range(n_unique_states):
            idx = s * self.n_measurements + measurement
            states_at_measurement[..., s, :] = as_time_series[..., idx, :]
        return states_at_measurement

    def get_states_at_measurements(self, data: np.ndarray) -> np.ndarray:
        """
        Reshape and reorder state data to separate measurement coordinates.

        Expands an input array of shape (n_batch, t_steps, n_measurements * features)
        into (n_batch, t_steps, n_measurements, n_states), where `n_states` corresponds
        to the number of features per measurement.

        Args:
            data (np.ndarray):
                Input array containing concatenated measurement features.

        Returns:
            np.ndarray:
                Reshaped array with separated measurement and state dimensions.
        """
        return np.swapaxes(data.reshape(*data.shape[:2], -1, self.n_measurements), axis1=-1, axis2=-2)

    def get_states_at_measurement_from_data(self, data: np.ndarray, measurement: int, n_measurements: int = None) -> np.ndarray:
        """Picks the states a given measurement position from a data matrix with a given number of measurements."""
        if n_measurements is None:
            n_measurements = self.n_measurements
        assert measurement < n_measurements, "The measurement position is too high for the count of measurement positions."
        states_at_measurement = np.zeros((*data.shape[:-1], len(self.state_keys)))
        for s, _ in enumerate(self.state_keys):
            idx = s * n_measurements + measurement
            states_at_measurement[..., s] = data[..., idx]
        return states_at_measurement

    def filter_arr_for_state_and_position(self, arr, positions: List[int], state_indices: List[int], n_positions: Optional[int] = None):
        """
        Filters a flattened state array to extract specific spatial positions and state features.
        The input array is reshaped to explicitly expose the spatial dimension, allowing
        selection of the desired (position, state) subset.
        Args:
            arr: The input state array, typically with shape (..., n_positions * n_states).
            positions: List of spatial indices (e.g., [0, 5]) to keep.
            state_indices: List of state feature indices (e.g., [0] for T, [1] for chi_E) to keep.
            n_positions: The total number of spatial discretization points. Defaults to self.n_measurements.
        Returns:
            np.ndarray: The filtered array with shape (..., len(positions), len(state_indices)).
        """
        n_positions = n_positions or self.n_measurements
        arr = arr.reshape((*arr.shape[:2], n_positions, -1), order="F")
        return arr[..., positions, state_indices]

    def filter_dict_data_for_state_and_position(self, dict_data: Dict[str, Dict[str, np.ndarray]], positions: List, state_indices: List, n_positions: Optional[int] = None):
        """
        Applies `filter_arr_for_state_and_position` to every NumPy array nested within
        a two-level dictionary structure (e.g., {surrogate: {case: array}}).
        The input dictionary is modified in place, and the filtered dictionary is returned.
        Args:
            dict_data: Nested dictionary containing arrays to be filtered.
                    Keys: {surrogate_key: {case_key: np.ndarray}}.
            positions: List of spatial indices to keep.
            state_indices: List of state feature indices to keep.
            n_positions: Total number of spatial discretization points. Defaults to self.n_measurements.
        Returns:
            Dict[str, Dict[str, np.ndarray]]: The dictionary containing the filtered arrays.
        """
        filtered = {}
        n_positions = n_positions or self.n_measurements
        for surrogate_key, case_dict in dict_data.items():
            for case_key, arr in case_dict.items():
                arr = self.filter_arr_for_state_and_position(arr, positions, state_indices, n_positions=n_positions)
                if surrogate_key not in filtered.keys():
                    filtered[surrogate_key] = {}
                filtered[surrogate_key][case_key] = arr
        return filtered

    def get_states_from_data(self, data: Union[np.ndarray, Dict], n_measurements: int = None, state: str = None) -> np.ndarray:
        if isinstance(data, Dict):
            return data[state]
        if n_measurements is None:
            n_measurements = self.n_measurements
        if state is None:
            start = 0
            stop = n_measurements * len(self.state_keys)
        else:
            index_map = self.get_key_to_reduced_data_idx()
            for key in index_map.keys():
                if key == f"{state}_0":
                    start = index_map[key]
                    stop = start + n_measurements
        return data[..., start:stop]

    def get_inputs_from_data(self, data: np.ndarray) -> np.ndarray:
        """Extracts the inputs from a data matrix that has the shape (batch_size, time_steps, features)."""
        start = len(self.state_keys) * self.n_measurements
        stop = start + len(self.input_keys)
        return data[..., start:stop]

    def get_tvps_from_data(self, data: np.ndarray) -> np.ndarray:
        """Extracts the tvps from a data matrix that has the shape (batch_size, time_steps, features)."""
        start = len(self.state_keys) * self.n_measurements + len(self.input_keys)
        stop = start + len(self.tvp_keys)
        return data[..., start:stop]

    def get_inputs_from_stacked(self, vector: np.ndarray, time_instant: int = None) -> np.ndarray:
        as_time_series = self._time_series_from_stacked(vector)
        start = len(self.state_keys) * self.n_measurements
        end = start + len(self.input_keys)
        if time_instant is None:
            return as_time_series[..., start:end, :]
        return as_time_series[..., start:end, time_instant]

    def get_tvps_from_stacked(self, vector: np.ndarray) -> np.ndarray:
        as_time_series = self._time_series_from_stacked(vector)
        start = len(self.state_keys) * self.n_measurements + len(self.input_keys)
        end = start + len(self.input_keys)
        return as_time_series[..., start:end, :]

    def get_dompc_scheme(self):
        """Returns the vector scheme after which the state vector for the dompc model must be set up.
        This is exactly the scheme that arises from the function to_dompc_vector().
        This function is for readabitliy and to help construct the symbolic state vector.
        In the dompc model, the current input and the current tvp are not of type state. Thus, they simply need to be deleted from the vector.
        """
        start_input_t0 = len(self.state_keys) * self.n_measurements * self.time_horizon
        stop_input_t0 = start_input_t0 + len(self.input_keys)
        start_tvp_t0 = start_input_t0 + len(self.input_keys) * self.time_horizon
        stop_tvp_t0 = start_tvp_t0 + len(self.tvp_keys)
        dompc_keys = []
        t0_indizes = []
        for i, _ in enumerate(self.all_expanded_keys):
            if start_input_t0 <= i < stop_input_t0 or start_tvp_t0 <= i < stop_tvp_t0:
                t0_indizes.append(i)
                continue
            dompc_keys.append(self.all_expanded_keys[i])
        return dompc_keys, t0_indizes

    def to_key_dict(self, vector: np.ndarray, vector_type: str) -> Dict:
        if vector_type == "dompc":
            assert vector.ndim == 2, "dompc vectors must be of shape (n_features, 1)"
            keys = self.dompc_keys
            vector = vector.flatten()
        elif vector_type == "narx":
            keys = self.all_expanded_keys
        else:
            raise ValueError("Choose 'dompc' or 'narx' as vector_type.")
        assert vector.shape[-1] == len(keys), f"The feature length {vector.shape[-1]} does not match the numer of corresponding keys {len(keys)}."
        key_dict = dict()
        for i, key in enumerate(keys):
            if vector.ndim == 1:
                key_dict[key] = vector[i]
            else:
                key_dict[key] = vector[..., i]
        return key_dict

    def to_dompc_vector(self, reduced_data_matrix: np.ndarray) -> np.ndarray:
        """Takes a reduced data matrix in the form (batch_size, time_steps, features) and excludes the current input and the current tvp as they are not a state of the do-mpc model.
        It has to have enough time points to fill the moving window.
        Args:
            reduced_data_matrix (np.ndarray): Data matrixn with reduced measurements in the form (batch_size, time_steps, features). The features must be ordered (states, inputs, tvps)
        Returns:
            np.ndarray: Initial state vector to run a dompc model with the NARX model as surrogate. It has the order in the feature dimension: (states0, states-1, inputs-1, tvps-1, ...)
            It is transposed to meet the vector shape of casadi which has the features in the first dimension.
        """
        stacked = self.stack_data(reduced_data_matrix)
        dompc_vector = np.delete(stacked, self.t0_indizes, axis=-1)
        return np.swapaxes(dompc_vector, -1, -2)

    def slice_dompc_vector(self, dompc_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Slices a dompc_vector into states at all time points, inputs at all time points and tvps at all time points.
        The dompc vector must be of shape (n_features, 1)"""
        start_inputs = len(self.state_keys) * self.n_measurements * self.time_horizon
        start_tvps = start_inputs + len(self.input_keys) * (self.time_horizon - 1)  # it does not contain the current input nor the current tvps
        return dompc_vector[:start_inputs], dompc_vector[start_inputs:start_tvps], dompc_vector[start_tvps:]

    def update_dompc_vector(
        self,
        x_previous: np.ndarray,
        u_applied: np.ndarray,
        tvp_applied: np.ndarray,
        x_current_full: np.ndarray,
        n_full_measurements: int,
    ) -> np.ndarray:
        """All in puts must be of shape (features, 1)"""
        current_states = self.reduce_measurements(x_current_full.T, n_full_measurements).T
        past_states, past_inputs, past_tvps = self.slice_dompc_vector(x_previous)
        past_states = past_states[: -len(self.state_keys) * self.n_measurements]  # drop the oldest instance of states, inputs and tvps
        past_inputs = past_inputs[: -len(self.input_keys)]
        past_tvps = past_tvps[: -len(self.tvp_keys)]
        return np.concatenate([current_states, past_states, u_applied, past_inputs, tvp_applied, past_tvps], axis=0)

    def casadi_stack(self, current_states: MX, past_states: MX, current_inputs: MX, past_inputs: MX, current_tvps: MX, past_tvps: MX) -> MX:
        stacked = vertcat(current_states, past_states, current_inputs, past_inputs, current_tvps, past_tvps)
        return stacked

    def isolate_state(self, Y: np.ndarray, state_key: str, n_measurements: int = None) -> np.ndarray:
        """Takes the output data Y and isolates either the temperature which is the last state or all other states.
        Y must be of shape (..., n_all_states)
        """
        n_measurements = n_measurements or self.n_measurements
        assert Y.shape[-1] == len(self.state_keys) * n_measurements, f"The number of all states {Y.shape[-1]} does not match the number of states times their measurement positions."
        start = 0
        stop = (self.n_states - 1) * n_measurements
        if state_key == "T":
            start = stop
            stop = start + n_measurements
        return Y[..., start:stop]

    def import_rom_parameters(self, rom_parameters: Dict) -> None:
        params = {key: np.array(value) for key, value in rom_parameters.items()}
        self.rom_parameters = params

    def full_to_rom(self, vector: np.ndarray) -> np.ndarray:
        # shape of vector = (n_time_steps, n_states) or (n_states, )
        assert (
            vector.shape[-1] == self.rom_parameters["phi"].shape[0]
        ), f"The input vector has the wrong shape. The last dimension must be {self.rom_parameters["phi"].shape[0]} but it is {vector.shape[-1]}."

        vector = (vector - self.rom_parameters["mean"]) / self.rom_parameters["std"]  # white the data
        return vector @ self.rom_parameters["phi"]

    def rom_to_full(self, vector: Union[np.ndarray, SX, MX]) -> np.ndarray:
        # shape of vector = (n_time_steps, n_states) or (n_states, )
        assert (
            vector.shape[-1] == self.rom_parameters["phi"].shape[1]
        ), f"The input vector has the wrong shape. The last dimension must be {self.rom_parameters["phi"].shape[1]} but it is {vector.shape[-1]}."

        vector = vector @ self.rom_parameters["phi"].T
        vector_unscaled = vector * self.rom_parameters["std"] + self.rom_parameters["mean"]
        return vector_unscaled

    def load_data(self, data_dir: str, num_trajectories: int = -1, num_time_steps: int = -1) -> np.ndarray:
        """Loads the newest data as one concatenated np array."""
        all_batches = []
        for i, file in enumerate(os.listdir(data_dir)):
            if file.endswith(".npy"):
                path_to_file = os.path.join(data_dir, file)
                loaded = np.load(path_to_file, allow_pickle=True, mmap_mode="r")
                all_batches.append(loaded)
            if num_trajectories != -1 and (i + 1) >= num_trajectories:
                break
        all_batches = np.array(all_batches)
        return all_batches[:num_trajectories, :num_time_steps]


def get_latest_folder(data_dir: str) -> str:
    all_entries = os.listdir(data_dir)
    all_subdirs = []
    for entry in all_entries:
        full_path = os.path.join(data_dir, entry)
        if os.path.isdir(full_path):
            all_subdirs.append(entry)
    latest_folder_name = sorted(all_subdirs)[-1]
    path_to_newest_folder = os.path.join(data_dir, latest_folder_name)
    return path_to_newest_folder
