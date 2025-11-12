import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from neurals.TorchPredictors import PCA_Encoder, StatePredictor
from routines.data_structurizer import DataStructurizer, get_latest_folder

from .train_narx_module import LossFuncType, train_narx_module
from .train_pca_encoder import train_pca_encoder
from .train_rom import train_rom


def run_training(
    model_parameter_dir: str,
    training_data_dir: str,
    data_structurizer: DataStructurizer,
    training_cfg: Dict,
    constraint_matrix: Optional[np.ndarray] = None,
    boundary_cond: Optional[np.ndarray] = None,
):
    """Executes the complete training pipeline. To train the following torch surrogates:
    1) PCA Encoder 2) NARX for the nominal states and nominal temperature 3) NARX for the 10% and 90% quantile of the temperature 4) NARX for the states conditioned on the temperature quantiles.
    This will take some time.
    Export paths must include the file name and extension.
    """
    save_paths = [os.path.join(model_parameter_dir, f"{key}.pth") for key in training_cfg.get("training_jobs").keys()]
    pca_encoder_path = os.path.join(model_parameter_dir, f"{training_cfg.get("pca_encoder").get("name")}.pth")
    [cfg.update({"save_path": f"{path}"}) for cfg, path in zip(training_cfg["training_jobs"].values(), save_paths)]

    if not np.all([os.path.exists(path) for path in save_paths]):
        with open(os.path.join(model_parameter_dir, "training_cfg.json"), "w") as f:
            f.write(json.dumps(training_cfg, indent=4))

        training_data = data_structurizer.load_data(data_dir=training_data_dir, num_trajectories=training_cfg.get("n_trajectories", 16))

        train_pca_encoder(
            data_structurizer=data_structurizer,
            training_cfg=training_cfg,
            save_cfg={"save_path": pca_encoder_path},
            training_data=training_data,
        )
        pca_encoder = torch.load(pca_encoder_path, weights_only=False)

        for individual_cfg in training_cfg["training_jobs"].values():
            # infer weights
            if individual_cfg.get("loss_function", "mse") == LossFuncType.Weighted.value:
                # create model parameter path from training cfg
                quantile = individual_cfg.get("quantile")

                for module_cfg in training_cfg["training_jobs"].values():
                    if LossFuncType.Quantile.value in module_cfg.keys():
                        if module_cfg["quantile"] == quantile:
                            path_to_quantile_temp_params = module_cfg.get("save_path")
                            break

                # check whether file exists
                if not os.path.exists(path_to_quantile_temp_params):
                    raise ValueError(f"The model parameters for the {quantile} temperature model do not exists under {path_to_quantile_temp_params}.")
                temp_model_params = torch.load(path_to_quantile_temp_params)
                # prepare weights
                weights, distances = infer_temperature_weights(
                    data_structurizer=data_structurizer, training_data=training_data, training_cfg=training_cfg, pca_encoder=pca_encoder, temp_model_state_dict=temp_model_params
                )
                if training_cfg.get("save_distances", False):
                    distances_file_path = os.path.join(model_parameter_dir, f"distances_{quantile}.npy")
                    np.save(arr=distances.numpy(), file=distances_file_path)
            else:
                weights = None

            train_narx_module(
                data_structurizer=data_structurizer,
                pca_encoder=pca_encoder,
                training_cfg=training_cfg,
                individual_cfg=individual_cfg,
                training_data=training_data,
                constraint_matrix=constraint_matrix,
                boundary_cond=boundary_cond,
                weights=weights,
            )


def infer_temperature_weights(data_structurizer: DataStructurizer, training_data: np.ndarray, pca_encoder: PCA_Encoder, training_cfg: Dict, temp_model_state_dict: Dict) -> Tuple[torch.Tensor]:
    training_data = data_structurizer.reduce_measurements(training_data)
    X, Y = data_structurizer.make_training_XY(data_matrix=training_data)
    X = torch.tensor(X.copy(), dtype=torch.float32)

    X_enc = pca_encoder(X)
    Y = data_structurizer.isolate_state(Y=Y, state_key="T")
    Y = torch.tensor(Y.copy(), dtype=torch.float32)

    temp_model = StatePredictor(n_input=X_enc.shape[-1], n_output=Y.shape[-1], hidden_units=training_cfg.get("hidden_units"))
    temp_model.load_state_dict(temp_model_state_dict)
    temp_model.to(torch.device("cpu"))
    temp_model.eval()

    with torch.no_grad():
        T_pred = temp_model(X_enc).detach()  # predicted temperature quantile (scaled)

        # T_sc = temp_model.out_scaler(Y)  # scale the training data
        sigma_weight = training_cfg.get("sigma_weight", 0.002)
        distances = torch.norm(T_pred - Y, p=2, dim=-1)

        weight_fn = lambda d: 2 / ((2 * torch.pi) ** 0.5 * sigma_weight) * torch.exp(-(d**2) / (2 * sigma_weight**2))
        # normal distribution to weight the points multiplied by 2 to have a surface area of 1
        weights = weight_fn(d=distances)
    return weights, distances


if __name__ == "__main__":
    CURR_DIR = os.path.dirname(__file__)
    ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
    PARENT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
    CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
    CONFIG_NAME = "etox_control_task.yaml"
    sys.path.insert(0, ROOT_DIR)

    with open(os.path.join(CONFIG_DIR, "etox_control_task.yaml"), "r") as f:
        sim_cfg = yaml.safe_load(f)

    backward_horizon = 8
    n_measurements = 4
    narx_vector_length = 24
    max_epochs = 256
    n_trajectories = -1

    structurizer = DataStructurizer(
        n_measurements=n_measurements,
        n_initial_measurements=sim_cfg["simulation"]["N_finite_diff"],
        time_horizon=backward_horizon,
        state_keys=sim_cfg["states"]["keys"],
        input_keys=sim_cfg["inputs"]["all_keys"],
        tvp_keys=sim_cfg["tvps"]["keys"],
    )
    model_parameter_dir = os.path.join(PARENT_DIR, "..", "trained_models", sim_cfg["model_name"])
    training_data_dir = os.path.abspath(os.path.join(ROOT_DIR, "..", "..", "data", "training"))
    newest_dir = get_latest_folder(data_dir=training_data_dir)

    run_training(
        model_parameter_dir=model_parameter_dir,
        training_data_dir=newest_dir,
        data_structurizer=structurizer,
        narx_vector_length=narx_vector_length,
        n_trajectories=n_trajectories,
        max_epochs=max_epochs,
    )
