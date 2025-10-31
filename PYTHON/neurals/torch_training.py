import glob
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from neurals.TorchPredictors import CombinedStatePredictor, MinMaxScalerModule, PCA_Encoder, StatePredictor
from routines.data_structurizer import DataStructurizer
from torch.utils.data import DataLoader, Dataset


class BatchVarNames(Enum):
    Weights = "weights"
    BoundaryConds = "boundary_cond"


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return

        improvement = current_score < self.best_score - self.min_delta if self.mode == "min" else current_score > self.best_score + self.min_delta

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class EncodedNARXDataset(Dataset):
    """Dataset that allows arbitrarily many batch items."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, **kwargs: Optional[Dict[str, torch.Tensor]]):
        assert len(x) == len(y), "Features and labels need to have same length."
        assert x.ndim == y.ndim == 3, f"Data {x.shape} and labels {y.shape} must contain a batch dimension that contains the different trajectories."
        for arg in kwargs.values():
            assert arg.ndim >= 2, f"All other variable batch items {arg.shape} must contain a batch dimension."

        self.n_trajectories = x.shape[0]
        self.n_time_steps = x.shape[1]

        (self.data, self.labels), self.kwargs = self._flatten_trajects_time(x, y, **kwargs)

    def _flatten_trajects_time(self, *args: Tuple[torch.Tensor], **kwargs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        flattened = []
        flattened_kwargs = {}
        for arg in args:
            flattened.append(arg.reshape((-1, arg.shape[-1])))
        if kwargs:
            for key, arg in kwargs.items():
                if arg.ndim == 3:
                    flattened_kwargs[key] = arg.reshape((-1, arg.shape[-1]))
                elif arg.ndim == 2:
                    flattened_kwargs[key] = arg.flatten()
        return flattened, flattened_kwargs

    def _encode_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.reshape((-1, inputs.shape[-1]))
        encoded = self.encoder(inputs)
        return encoded.reshape((self.n_trajectories, self.n_time_steps, encoded.shape[-1]))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {"x": self.data[idx], "y": self.labels[idx]}
        for key, tensor in self.kwargs.items():
            sample[key] = tensor[idx]
        return sample


def train(
    neural_model: Type[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: Optional[int] = 128,
    loss_function: Optional[Type[nn.Module]] = None,
    lr: Optional[float] = 1.0e-4,
    early_stopping: Optional[bool] = False,
) -> nn:
    if torch.cuda.is_available():
        device_name = "cuda"  # Train on NVIDIA GPU, if available
        print("Training on CUDA.")
    elif torch.backends.mps.is_available():
        device_name = "mps"  # For Apple silicon supported Macs, use the metal GPU
        print("Training on Apple Silicon GPU.")
    else:
        device_name = "cpu"  # Else train on cpu. Not recommended
        print("Training on CPU.")
    device = torch.device(device_name)
    neural_model.to(device)
    optimizer = optim.Adam(neural_model.parameters(), lr=lr)
    history = {"epoch": [], "loss": [], "val_loss": []}
    if early_stopping:
        early_stopping = EarlyStopping()
    loss_function = loss_function or nn.MSELoss()

    for epoch in range(epochs):
        neural_model.train()
        for batch_variables in train_loader:
            x_b, y_b = batch_variables["x"].to(device), batch_variables["y"].to(device)
            # if BatchVarNames.BoundaryConds.value in batch_variables.keys():
            #     boundary_cond = batch_variables[BatchVarNames.BoundaryConds.value].to(device)
            #     pred = neural_model.forward(x_b, boundary_cond)
            # else:
            pred = neural_model.forward(x_b)

            if BatchVarNames.Weights.value in batch_variables.keys():
                weights = batch_variables[BatchVarNames.Weights.value].to(device)
                loss = loss_function(pred, y_b, weights=weights)
            else:
                loss = loss_function(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        neural_model.eval()
        val_losses = np.zeros(len(val_loader))
        with torch.no_grad():
            for i, batch_variables in enumerate(val_loader):
                x_b, y_b = batch_variables["x"].to(device), batch_variables["y"].to(device)

                # if BatchVarNames.BoundaryConds.value in batch_variables.keys():
                #     boundary_cond = batch_variables[BatchVarNames.BoundaryConds.value].to(device)
                #     y_val = neural_model(x_b, x_ref=boundary_cond)
                # else:
                y_val = neural_model(x_b)

                if BatchVarNames.Weights.value in batch_variables.keys():
                    weights = batch_variables[BatchVarNames.Weights.value].to(device)
                    val_loss = loss_function(y_val, y_b, weights=weights)
                else:
                    val_loss = loss_function(y_val, y_b)
                val_losses[i] = val_loss

        history["epoch"].append(epoch)
        loss = loss.item()
        history["loss"].append(loss)
        val_loss = val_losses.mean()
        history["val_loss"].append(val_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train loss: {loss:.7f}, Val loss: {val_loss:.7f}")

        if early_stopping is not False:
            early_stopping(val_loss.item())
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}.")
                break
    return neural_model, history


def save_model(neural_model: Type[nn.Module], final_export_path: Path):
    """Saves the trained model to the specified run ID."""
    torch.save(neural_model.state_dict(), final_export_path)
    print(f"Model saved to {final_export_path}")


def load_state_predictor(model_configurations: Dict, model_dir: str) -> torch.nn.Module:
    models = {}
    pca_encoder = torch.load(os.path.join(model_dir, "pca_encoder.pth"), weights_only=False)
    for model_key, model_cfg in model_configurations.items():
        file_name = model_cfg["file_name"]
        model_params = torch.load(os.path.join(model_dir, file_name))

        weight_keys = [k for k in model_params.keys() if "weight" in k]
        if not weight_keys:
            raise ValueError("No tensors for ('weight') found.")
        input_dim = model_params[weight_keys[0]].shape[1]
        output_dim = model_params[weight_keys[-1]].shape[0]
        hidden_units = []

        if "constraint_rhs" in model_params.keys():
            kwargs = {"n_constraints": len(model_params["constraint_rhs"])}
        else:
            kwargs = {}

        for key in weight_keys[:-1]:
            hidden_units.append(model_params[key].shape[0])
        model = model_cfg["cls"](n_input=input_dim, n_output=output_dim, hidden_units=hidden_units, **kwargs)
        model.load_state_dict(model_params)
        models[model_key] = model

    state_predictor = CombinedStatePredictor(
        encoder=pca_encoder,
        state_predictor=models["state_model"],
        temp_predictor=models["temperature_model"],
    )
    return state_predictor


class NARXDataset(Dataset):

    def __init__(self, data_dir: str, data_structurizer: DataStructurizer, file_pattern="*.npy"):
        """
        Args:
            data_dir (string): Pfad zum Verzeichnis mit den .npy-Dateien.
            file_pattern (string): Muster zum Filtern der Dateien (z.B. '*.npy').
            transform (callable, optional): Optionale Transformationen, die angewendet werden sollen.
        """
        self.data_structurizer = data_structurizer
        self.n_in_features = (data_structurizer.n_states * data_structurizer.n_measurements + data_structurizer.n_inputs + data_structurizer.n_tvps) * data_structurizer.time_horizon
        self.n_out_features = data_structurizer.n_states * data_structurizer.n_measurements
        self.file_paths = glob.glob(os.path.join(data_dir, file_pattern))
        self.file_paths.sort()
        self.data = []

        for npy_path in sorted(os.listdir(data_dir)):  # load all data at once requires at least 16 GB Ram
            if npy_path.endswith(".npy"):
                arr = np.load(os.path.join(data_dir, npy_path), allow_pickle=True)
                arr = data_structurizer.reduce_measurements(arr)
                X, Y = data_structurizer.make_training_XY(arr)
                self.data.append((torch.from_numpy(X.copy()).float(), torch.from_numpy(Y.copy()).float()))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        return self.data[index]
