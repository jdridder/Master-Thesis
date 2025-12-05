import datetime
import inspect
import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


def get_kwargs_from_specs(function: Callable, specs: Dict) -> Dict:
    keywords = set(inspect.signature(function).parameters.keys())
    kwargs = {}
    for keyword in specs.keys():
        if keyword in keywords:
            kwargs[keyword] = specs[keyword]
    return kwargs


def get_directory_for_today(root_directory: str) -> str:
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d")
    path_to_directory = os.path.join(root_directory, folder_name)
    os.makedirs(path_to_directory, exist_ok=True)
    return path_to_directory


def _load_single_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        data = json.load(f)
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = np.array(value)
    return data


def load_json_results(result_dir: str, n_trajectories: int = -1) -> Dict[str, Any]:
    """
    Loads JSON result files from a directory and stacks their content into numpy arrays.

    The function identifies JSON files in the specified directory (excluding files with 'meta'
    in the name), loads them in parallel, and recursively merges the data. Leaf nodes (arrays
    or lists) from all files are stacked along the first dimension (axis 0).

    Args:
        result_dir (str): Path to the directory containing the JSON files.
        n_trajectories (int, optional): Number of files to sample randomly.
                                        If -1, all files are loaded. Defaults to -1.

    Returns:
        Dict[str, Any]: A dictionary preserving the structure of the input JSONs,
                        where the lowest level values are stacked numpy arrays.

    Raises:
        FileNotFoundError: If no valid JSON files are found in the directory.
    """
    all_files = [f for f in os.listdir(result_dir) if f.endswith(".json") and "meta" not in f]

    if not all_files:
        raise FileNotFoundError(f"No json result files in {result_dir} found.")

    if n_trajectories != -1:
        files_to_load = np.random.choice(all_files, size=min(n_trajectories, len(all_files)), replace=False)
    else:
        files_to_load = all_files

    file_paths = [os.path.join(result_dir, f) for f in files_to_load]

    json_list = Parallel(
        n_jobs=-1,
    )(delayed(_load_single_json)(path) for path in file_paths)

    if not json_list:
        return {}

    def _recursive_stack(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not list_of_dicts:
            return {}

        reference_dict = list_of_dicts[0]
        stacked_result = {}

        for key, value in reference_dict.items():
            if isinstance(value, dict):
                sub_list = [d[key] for d in list_of_dicts]
                stacked_result[key] = _recursive_stack(sub_list)
            else:
                arrays_to_stack = [d[key] for d in list_of_dicts]
                stacked_result[key] = np.stack(arrays_to_stack, axis=0)

        return stacked_result

    return _recursive_stack(json_list)


def df_from_double_dict(double_dict: Dict[str, Dict[str, np.ndarray]], column_names: List[str], arr_indices: Optional[np.ndarray] = None):
    records = [(parent, child, ind, val) for parent, children in double_dict.items() for child, array in children.items() for ind, val in zip(arr_indices, array)]
    df_final = pd.DataFrame(records, columns=column_names)
    return df_final


def apply_to_double_dict(double_dict: Dict[str, Dict[str, np.ndarray]], fn: Callable, **kwargs: Optional[Dict]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Applies a callable function (fn) to every np.ndarray value within a two-level nested dictionary.

    A new dictionary structure is returned, preserving the original keys.

    Parameters
    ----------
    double_dict : Dict[str, Dict[str, np.ndarray]]
        The nested dictionary containing arrays.
    fn : Callable
        The function to apply to each array.
    **kwargs : Optional[Dict]
        Additional keyword arguments passed to `fn`.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        A new nested dictionary containing the results.
    """
    new_dict = {}
    for parent_key, child_dict in double_dict.items():
        new_dict[parent_key] = {}
        for child_key, child_arr in child_dict.items():
            new_dict[parent_key][child_key] = fn(child_arr, **kwargs)
    return new_dict


def merge_dict(dict_list: List[Dict]) -> Dict:
    merged_dict = {}
    for i, item_dict in enumerate(dict_list):
        for surrogate_key, surrogate_stats in item_dict.items():
            if surrogate_key not in merged_dict:
                merged_dict[surrogate_key] = {}
            for stat_key, stat in surrogate_stats.items():
                if stat_key not in merged_dict[surrogate_key]:
                    merged_dict[surrogate_key][stat_key] = []
                merged_dict[surrogate_key][stat_key].append(np.array(stat))
                if i == len(dict_list) - 1:
                    merged_dict[surrogate_key][stat_key] = np.array(merged_dict[surrogate_key][stat_key])
    return merged_dict


def load_json_results_for_all(result_dir: str, n_trajectories: int = -1) -> Dict[str, Dict[str, np.ndarray]]:
    print("---- Loading .json results. ----")
    all_results = {}
    sub_dirs = [d.name for d in os.scandir(result_dir) if d.is_dir()]

    for directory in sub_dirs:
        directory_path = os.path.join(result_dir, directory)
        try:
            all_results[directory] = load_json_results(result_dir=directory_path, n_trajectories=n_trajectories)
        except FileNotFoundError as e:
            print(f"Error loading the file: {e}")
            continue

    return all_results


def filter_test_data_for_surrogates(test_data: np.ndarray, surrogate_results: Dict[str, Dict[str, np.ndarray]]):
    """
    Filters test data for each surrogate model based on the trajectory indices
    provided in the surrogate results.

    Args:
        test_data (np.ndarray): Array containing all test trajectories.
        surrogate_results (Dict[str, np.ndarray]): Dictionary of surrogate outputs,
            where each entry must include an "index" array specifying which
            trajectories belong to that surrogate.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping each surrogate key to its
        corresponding filtered subset of the test data.
    """
    filtered_test_data = {}
    for surr_key, surr_result in surrogate_results.items():
        assert "index" in surr_result.keys(), "The surrogate result dictionary must provide a numpy array that contains the indices of the simulated trajectories."
        filter = surr_result["index"]
        filtered_test_data[surr_key] = test_data[filter]

    return filtered_test_data
