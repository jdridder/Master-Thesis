import json
import os
import sys

import numpy as np
from do_mpc.model._pod_model import ProperOrthogonalDecomposition


def train_rom(training_data: np.ndarray, export_path: str, n_components: int):
    print(f"\n--- Performing incremental POD on the data for a subspace of {n_components} components model using {training_data.shape[0]*training_data.shape[1]} points. ---")
    # TODO: Use pytorch for the SVD which can run on GPU and is a lot faster.
    pod = ProperOrthogonalDecomposition()
    pod.perform_svd(snapshots=training_data, n_components=n_components)
    parameters = pod.export_parameters()
    with open(export_path, "w") as f:
        f.write(json.dumps(parameters, indent=4))
        print(f"--- Saved the trained POD model parameters to {export_path}. ---\n")


if __name__ == "__main__":
    import sys

    import yaml

    CURR_DIR = os.path.dirname(__file__)
    ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
    sys.path.append(ROOT_DIR)
    from routines.data_structurizer import DataStructurizer

    training_data_dir = os.path.abspath("/Users/jandavidridder/Desktop/Masterarbeit/src/experiments/001_certain_open_loop_kpis/2025-10-11/data/train")
    model_parameter_dir = os.path.abspath("/Users/jandavidridder/Desktop/Masterarbeit/src/experiments/001_certain_open_loop_kpis/2025-10-11/trained_models")
    sim_cfg = yaml.safe_load(open(os.path.join(ROOT_DIR, "configs", "etox_control_task.yaml"), "r"))

    structurizer = DataStructurizer(
        n_initial_measurements=sim_cfg["simulation"]["N_finite_diff"],
        n_measurements=sim_cfg["narx"]["n_measurements"],
        time_horizon=sim_cfg["narx"]["time_horizon"],
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        tvp_keys=sim_cfg["tvps"]["keys"],
    )

    data = structurizer.load_data(data_dir=training_data_dir, num_trajectories=32)
    data = structurizer.get_states_from_data(data, n_measurements=structurizer.n_initial_measurements)

    train_rom(training_data=data, model_parameter_dir=model_parameter_dir, n_components=24)
