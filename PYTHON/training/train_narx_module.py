import json
import os
from enum import Enum
from typing import Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":
    import sys

    import yaml

    CURR_DIR = os.path.dirname(__file__)
    ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, ".."))
    sys.path.append(ROOT_DIR)

from neurals.torch_training import EncodedNARXDataset, get_unit_dims_from_state_dict, train
from neurals.TorchPredictors import MinMaxScalerModule, PCA_Encoder, PCStatePredictor, StatePredictor
from routines.data_structurizer import DataStructurizer


class LastLayerAct(Enum):
    Vanilla = "vanilla"
    PhysicsConstrained = "physics_constrained"


class LossFuncType(Enum):
    Mse = "mse"
    Weighted = "weighted"
    Quantile = "quantile"


class PinballLoss(nn.Module):
    def __init__(self, quantile: float):
        super(PinballLoss, self).__init__()
        if not (0 < quantile < 1):
            raise ValueError("Quantile must be between 0 and 1.")
        self.tau = 1 - quantile

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        error = y_true - y_pred
        loss = torch.where(error >= 0, self.tau * error, (self.tau - 1) * error)
        return torch.mean(loss)


class WeigtedLoss(nn.Module):
    def __init__(self):
        super(WeigtedLoss, self).__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        loss = torch.norm(y_pred - y_true, p=2, dim=-1) ** 2
        return 1 / len(y_true) * weights @ loss


def train_narx_module(
    data_structurizer: DataStructurizer,
    pca_encoder: PCA_Encoder,
    individual_cfg: Dict[str, Dict],
    training_cfg: Dict,
    training_data: np.ndarray,
    constraint_matrix: Optional[np.ndarray] = None,
    boundary_cond: Optional[np.ndarray] = None,
    weights: Optional[torch.Tensor] = None,
):
    """Trains one torch module using the in the training_cfg and individual_cfg supplied hyperparemeters, network architecture
    type and loss function. Takes the full training data and formats it for the network to train. The pca encoder is trained separately."""
    data_matrix = data_structurizer.reduce_measurements(training_data)
    if individual_cfg.get("last_layer_activation") == LastLayerAct.PhysicsConstrained.value:
        assert constraint_matrix is not None, "You must provide at least a linear constraint matrix when training a physics constrained nn.Module."
        assert boundary_cond is not None, "You must provide a boundary condition for a physics constrained NN."

    X, Y = data_structurizer.make_training_XY(data_matrix)  # come with a batch dimension that contain the trajectories
    X = torch.tensor(X.copy(), dtype=torch.float32)
    X_enc = pca_encoder(X)  # supports batching
    Y = data_structurizer.isolate_state(Y, state_key=individual_cfg.get("state_key"))  # supports batching
    Y = torch.tensor(Y.copy(), dtype=torch.float32)

    input_scaler = MinMaxScalerModule(len=X_enc.shape[-1])
    # output_scaler = MinMaxScalerModule(len=Y.shape[-1])

    input_scaler.fit(X_enc)  # supports batching
    # output_scaler.fit(Y)  # supports batching
    # Y_scaled = output_scaler(Y)  # supports batching using the default broadcasting
    Y_scaled = Y
    prepared_data = {"x": X_enc, "y": Y_scaled}

    loss_func_type = individual_cfg.get("loss_function", LossFuncType.Mse.value)
    if loss_func_type == LossFuncType.Mse.value:
        # initalize MSE Loss function
        loss_function = nn.MSELoss()
    elif loss_func_type == LossFuncType.Quantile.value:
        # initialize quantile loss
        loss_function = PinballLoss(quantile=individual_cfg.get("quantile"))
    elif loss_func_type == LossFuncType.Weighted.value:
        assert weights is not None, "You need to provide weights for a weighted loss function."
        loss_function = WeigtedLoss()
        prepared_data["weights"] = weights
    else:
        raise NotImplementedError(f"Wrong loss function type declared {loss_func_type}.")

    last_layer_activation = individual_cfg.get("last_layer_activation", LastLayerAct.Vanilla.value)
    hidden_units = individual_cfg.get("hidden_units", training_cfg.get("hidden_units"))
    if last_layer_activation == LastLayerAct.Vanilla.value:
        # initialize vanilla narx
        model = StatePredictor(
            n_input=X_enc.shape[-1],
            n_output=Y.shape[-1],
            hidden_units=hidden_units,
            in_scaler=input_scaler,
            # out_scaler=output_scaler,
        )

    elif last_layer_activation == LastLayerAct.PhysicsConstrained.value:
        boundary_cond = torch.tensor(boundary_cond, dtype=torch.float32)
        # boundary_conditions = output_scaler(boundary_conditions)  # boundary conditions must be scaled using the same scaling as the output scaling
        # but some values are negative after the min max scaling and some are above 1 because the scaler is trained with data that excludes the boundary condition.
        # initialise physics constrained narx
        model = PCStatePredictor(
            n_input=X_enc.shape[-1],
            n_output=Y.shape[-1],
            hidden_units=hidden_units,
            boundary_cond=boundary_cond,
            n_constraints=constraint_matrix.shape[0],
            in_scaler=input_scaler,
            # out_scaler=output_scaler,
        )
        constraint_matrix = torch.tensor(constraint_matrix, dtype=torch.float32)
        model.set_linear_equality_constraint(A=constraint_matrix)

    else:
        raise NotImplementedError(f"You provided the wrong type of last layer activation function {last_layer_activation}.")

    # splits along the zerost axis
    splitted_data = train_test_split(*prepared_data.values(), test_size=1 - training_cfg.get("training_fraction", 0.8), random_state=42, shuffle=True)
    training = []
    validation = []
    for i, data_tensor in enumerate(splitted_data):
        if i % 2 == 0:
            training.append(data_tensor)
        elif i % 2 == 1:
            validation.append(data_tensor)

    train_loader = DataLoader(EncodedNARXDataset(**dict(zip(prepared_data.keys(), training))), batch_size=training_cfg.get("batch_size", 512), shuffle=True, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(EncodedNARXDataset(**dict(zip(prepared_data.keys(), validation))), batch_size=training_cfg.get("batch_size", 512), shuffle=False, num_workers=2, persistent_workers=True)

    print(f"\n--- Training {individual_cfg.get("name", "")} model using {X.shape[0] * X.shape[1]} points. --- \n")
    print(f"----- individual config: {individual_cfg}. ----")
    model, history = train(
        neural_model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        epochs=individual_cfg.get("max_epochs", training_cfg.get("max_epochs", 128)),
        lr=individual_cfg.get("lr", training_cfg.get("lr", 5e-5)),
        early_stopping=training_cfg.get("early_stopping", False),
        min_epochs=training_cfg.get("min_epochs", 0),
    )
    if training_cfg.get("save_history"):
        save_path = individual_cfg.get("save_path")
        with open(f"{save_path}_hist.json", "w") as f:
            f.write(json.dumps(history, indent=4))

    return model


def calibrate_quantile_models(
    quantile_state_dicts: List[Dict[str, torch.Tensor]],
    pca_encoder: PCA_Encoder,
    data_structurizer: DataStructurizer,
    calibration_data: np.ndarray,
    quantile_model_cfgs: List[Dict],
    training_cfg: Dict,
):
    """
    Calibrates upper and lower quantile neural networks using conformal prediction.
    Encodes calibration inputs, evaluates both quantile models, computes conformity
    scores, derives the quantile correction term, and attaches calibration buffers
    ('quantile', 'q_correction') to each model. Returns the calibrated models.
    """
    # 0 == upper quantile, # 1 == lower quantile
    if torch.cuda.is_available():
        device_name = "cuda"
        print("Training on CUDA.")
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    X_cal, Y_cal = data_structurizer.make_training_XY(calibration_data)
    n_cal = calibration_data.shape[0] * calibration_data.shape[1]
    significance_level = training_cfg.get("significance_level")
    quantiles = [1 - significance_level / 2, significance_level / 2]

    with torch.no_grad():
        X_cal = torch.tensor(X_cal.copy(), dtype=torch.float32, device="cpu")
        pca_encoder = pca_encoder.to("cpu")
        X_enc = pca_encoder(X_cal)
        X_enc = X_enc.to(device)
        Y_cal = data_structurizer.isolate_state(Y_cal, state_key=quantile_model_cfgs[0].get("state_key"))
        Y_cal = torch.tensor(Y_cal.copy(), dtype=torch.float32, device=device)
        Y_pred = []

        q_models = []
        for q_state_dict in quantile_state_dicts:  # upper, then lower model
            input_dim, hidden_units, output_dim = get_unit_dims_from_state_dict(q_state_dict)
            q_model = StatePredictor(n_input=input_dim, hidden_units=hidden_units, n_output=output_dim)
            q_model.load_state_dict(q_state_dict)
            q_model.to(device)
            Y_pred.append(q_model(X_enc).detach())
            q_models.append(q_model)

        conformity_high = Y_cal - Y_pred[0]
        conformity_low = Y_pred[1] - Y_cal
        conformity_scores = torch.concat([torch.unsqueeze(conformity_high, dim=3), torch.unsqueeze(conformity_low, dim=3)], dim=3)
        conformity_scores = torch.max(conformity_scores, dim=3).values

        q_corrected = (1 - significance_level) * (1 + 1 / n_cal)
        q_correction = torch.quantile(conformity_scores.reshape((-1, conformity_scores.shape[-1])), q=q_corrected, dim=0)

        for quantile, q_model in zip(quantiles, q_models):
            if quantile > 0.5:
                q_model.register_buffer("q_correction", q_correction)
            elif quantile < 0.5:
                q_model.register_buffer("q_correction", -q_correction)

    return q_models


# if __name__ == "__main__":
#     with open(os.path.join(ROOT_DIR, "configs", "etox_control_task.yaml")) as f:
#         sim_cfg = yaml.safe_load(f)
#     with open(os.path.join(ROOT_DIR, "models", "EtOxModel", "EtOxModel.yaml")) as f:
#         model_cfg = yaml.safe_load(f)

#     data_structurizer = DataStructurizer.load_from_cfg(cfg=sim_cfg)

#     from models.EtOxModel.EtOxModel import EtOxModel

#     meta_model = EtOxModel(
#         model_cfg=model_cfg,
#         state_keys=sim_cfg["states"].get("keys"),
#         input_keys=sim_cfg["inputs"].get("all_keys"),
#         N_finite_diff=sim_cfg["simulation"].get("N_finite_diff"),
#     )
#     constraint_matrix = meta_model.get_balance_constraint_matrix(num_stacks=4, include_temp_as_zero=False)

#     model_dir = os.path.abspath("/Users/jandavidridder/Desktop/trained_models")

#     training_cfg = {
#         "input_length": 16,
#         "hidden_units": [8],
#         "lr": 5e-5,
#         "max_epochs": 256,
#         "batch_size": 1024,
#         "training_ratio": 0.8,
#         "n_trajectories": 128,
#         "save_history": True,
#         "pca_encoder": {"name": "pca_encoder"},
#         "save_dir": "pc",
#         "training_jobs": {
#             "nominal_states": {
#                 "name": "nominal_states",
#                 "state_key": "x",
#                 "last_layer_activation": "physics_constrained",
#                 "loss_function": "mse",
#             },
#         },
#     }
#     [cfg.update({"save_path": f"{os.path.join(model_dir, name)}.pth"}) for name, cfg in training_cfg["training_jobs"].items()]

#     ind_cfg = training_cfg["training_jobs"].get("nominal_states")
#     path_pca_encoder = "/Users/jandavidridder/Desktop/trained_models/pca_encoder.pth"
#     pca_encoder = torch.load(path_pca_encoder, weights_only=False)

#     data_dir = "/Users/jandavidridder/Desktop/Masterarbeit/src/experiments/001_certain_open_loop_kpis/2025-10-28/data/train"
#     training_data = data_structurizer.load_data(data_dir=data_dir, num_trajectories=training_cfg.get("n_trajectories"))

#     bc = meta_model.get_bc_for_all_measurements(n_measurements=4)[:, :20]

#     train_narx_module(
#         data_structurizer=data_structurizer,
#         pca_encoder=pca_encoder,
#         individual_cfg=ind_cfg,
#         training_cfg=training_cfg,
#         training_data=training_data,
#         constraint_matrix=constraint_matrix,
#         boundary_cond=bc,
#     )
