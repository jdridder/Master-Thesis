from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_pca import PCA


class StandardScalerModule(nn.Module):
    def __init__(self, len: int, mean_val: torch.Tensor = None, std_val: torch.Tensor = None):
        super().__init__()
        self.register_buffer("mean_val", mean_val if mean_val is not None else torch.empty(1, len))
        self.register_buffer("std_val", std_val if std_val is not None else torch.empty(1, len))

    def fit(self, data: torch.Tensor):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if data.ndim == 3:
            data = data.reshape((-1, data.shape[-1]))  # flatten the batch dimension to do the fit all at once for all batches

        self.mean_val = data.mean(dim=0, keepdim=True)
        self.std_val = data.std(dim=0, keepdim=True)
        self.std_val[self.std_val == 0] = 1e-8

        self.register_buffer("mean_val", self.mean_val)
        self.register_buffer("std_val", self.std_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_val is None or self.std_val is None:
            raise ValueError("Scaler has not been fitted. Call .fit() first.")

        return (x - self.mean_val) / self.std_val

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_val is None or self.std_val is None:
            raise ValueError("Scaler has not been fitted. Call .fit() first.")

        return x * self.std_val + self.mean_val


class MinMaxScalerModule(nn.Module):
    def __init__(
        self,
        len: int,
        min_val: torch.Tensor = None,
        max_val: torch.Tensor = None,
        range_val: torch.Tensor = None,
        feature_range=(0, 1),
    ):
        super().__init__()
        self.feature_range = feature_range
        self.register_buffer("min_val", min_val if min_val is not None else torch.empty(1, len))
        self.register_buffer("max_val", max_val if max_val is not None else torch.empty(1, len))
        self.register_buffer("range_val", range_val if range_val is not None else torch.empty(1, len))
        """Scales the input or the output of a neural network using min-max-scaling, so they fit into the intervall (0,1)
        """

    def fit(self, data):
        """Determines min and max values of the data and saves them as buffer."""
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if data.ndim == 3:
            data = data.reshape((-1, data.shape[-1]))  # flatten the batch dimension to do the fit all at once for all batches

        # dimension 0 is the batch dimension, dimension 1 contains the features
        self.min_val = data.min(dim=0, keepdim=True).values
        self.max_val = data.max(dim=0, keepdim=True).values
        self.range_val = self.max_val - self.min_val

        self.range_val[self.range_val == 0] = 1e-8  # prevent from dividing by 0

        self.register_buffer("min_val", self.min_val)
        self.register_buffer("max_val", self.max_val)
        self.register_buffer("range_val", self.range_val)

    def forward(self, x):
        """Scales the input"""
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted. Call .fit() first.")
        x_sc = (x - self.min_val) / self.range_val
        x_sc = x_sc * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return x_sc

    def scale_linear_equality_constraint(self, A: torch.Tensor, b: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b = b or torch.zeros(A.shape[0])
        A_scaled = A * self.range_val
        # A_scaled = A / self.range_val
        b_scaled = b - self.min_val @ A_scaled.T
        return A_scaled, b_scaled

    def reverse(self, x):
        """Reverts the scaling."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted. Call .fit() first.")
        x = (x - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        x = x * self.range_val + self.min_val
        return x


class PCA_Encoder(nn.Module):
    """PCA Encoder, that wraps a scaler and a torch PCA dimensionality reduction into one module.
    The encoder reduces a high dimensional input space into a smaller space before the input is feeded into a neural network.
    The features that should bypass the encoder but are still scaled using the scaler module are stacked ontop of the input vector x. To indicate the number of bypass features,
    pass a bypass_dummy to the module.
    """

    def __init__(self, pca_module: PCA, in_scaler: MinMaxScalerModule):
        super().__init__()
        self.in_scaler = in_scaler
        self.pca = pca_module

    def forward(self, x) -> torch.Tensor:
        x = self.in_scaler(x)
        x = self.pca.transform(x)
        return x


class StatePredictor(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        hidden_units: List[int],
        in_scaler: MinMaxScalerModule = None,
        out_scaler: MinMaxScalerModule = None,
    ):
        super().__init__()
        assert len(hidden_units) >= 1, "The number of hidden units must be greater or equal than one."
        self.in_scaler = in_scaler if in_scaler is not None else MinMaxScalerModule(n_input)
        self.out_scaler = out_scaler if out_scaler is not None else MinMaxScalerModule(n_output)
        self.n_output = n_output

        all_dims = [n_input] + hidden_units + [n_output]
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                layers.append(nn.GELU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.in_scaler(x) if self.in_scaler is not None else x
        x = self.network(x)
        # the reverse of the scaling must be done manually for inference
        return x


class PCStatePredictor(StatePredictor):
    "Physics constrained State predictor module with the analystic solution of the KKT conditions as last activation layer"

    def __init__(self, n_input, n_output, hidden_units, n_constraints: int, boundary_cond: torch.Tensor = None, in_scaler=None, out_scaler=None):
        super().__init__(n_input, n_output, hidden_units, in_scaler, out_scaler)
        self.register_buffer("projection_inverse", torch.empty((n_output + n_constraints, n_output + n_constraints)))
        self.register_buffer("constraint_matrix", torch.empty((n_constraints, n_output)))
        self.register_buffer("constraint_rhs", torch.empty((n_constraints,)))

        if boundary_cond is None:
            boundary_cond = torch.empty((1, n_output))
        assert boundary_cond.shape == (1, self.n_output), f"The boundary condition must be of shape (1, {self.n_output}). You have {boundary_cond.shape}"

        self.register_buffer("boundary_cond", boundary_cond)
        self.register_buffer("ru_vec", torch.empty((1, n_constraints)))

    def set_linear_equality_constraint(self, A: torch.Tensor, b: torch.Tensor = None):
        b = b or torch.zeros(A.shape[0], dtype=A.dtype)
        # A, b = self.out_scaler.scale_linear_equality_constraint(A=A)
        U = torch.cat([2 * torch.eye(A.shape[1], dtype=A.dtype, device=A.device), A.T], dim=1)
        L = torch.cat([A, torch.zeros((A.shape[0], A.shape[0]), dtype=A.dtype, device=A.device)], dim=1)
        projection_inverse = torch.linalg.inv(torch.cat([U, L], dim=0))
        ru_vec = self.boundary_cond @ A.T + b

        self.register_buffer("projection_inverse", projection_inverse)
        self.register_buffer("constraint_matrix", A)
        self.register_buffer("constraint_rhs", b)
        self.register_buffer("ru_vec", ru_vec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        # x = self.out_scaler.reverse(x)
        ru = torch.repeat_interleave(self.ru_vec, repeats=x.shape[0], dim=0)
        intermediate_vector = torch.cat([2 * x, ru], dim=-1)
        x = intermediate_vector @ self.projection_inverse
        x = x[..., : self.n_output]  # cutting dual variables
        # x = self.out_scaler(x)
        return x


class CombinedStatePredictor(nn.Module):
    """Wrapper class to combines two networks that take the same inputs and concatenates their output.
    The output is scaled back. This is intended for inference only."""

    def __init__(
        self,
        temp_predictor: StatePredictor,
        state_predictor: StatePredictor,
        encoder: nn.Module = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.temp_predictor = temp_predictor
        self.state_predictor = state_predictor

    def forward(self, x):
        x = self.encoder(x) if self.encoder else x
        state_pred = self.state_predictor(x)
        # state_pred = self.state_predictor.out_scaler.reverse(state_pred)
        temp_pred = self.temp_predictor(x)
        # temp_pred = self.temp_predictor.out_scaler.reverse(temp_pred)
        return torch.cat((state_pred, temp_pred), dim=1)
