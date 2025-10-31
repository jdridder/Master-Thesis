import os
from typing import Dict

import numpy as np
import torch
from neurals.torch_training import NARXDataset
from neurals.TorchPredictors import MinMaxScalerModule, PCA_Encoder, StandardScalerModule
from routines.data_structurizer import DataStructurizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch_incremental_pca.incremental_pca import IncrementalPCA as IPCA
from torch_pca import PCA


def train_pca_encoder(
    data_structurizer: DataStructurizer,
    training_data: np.ndarray,
    training_cfg: Dict,
    save_cfg: Dict,
):
    if not os.path.exists(save_cfg.get("save_path")):
        narx_vector_length = training_cfg.get("input_length")
        print(f"\n--- Training the PCA Encoder with number components {narx_vector_length}. ---")
        device = torch.device("cpu")
        data = data_structurizer.reduce_measurements(training_data, n_initial_measurements=128)
        X, _ = data_structurizer.make_training_XY(data)

        X = X.reshape((-1, X.shape[-1]))
        X = torch.tensor(X.copy(), device=device, dtype=torch.float32)

        scaler = StandardScalerModule(len=X.shape[-1])
        scaler.to(device)
        scaler.fit(X)
        X = scaler(X)  # scaled data lives between 0 and 1
        pca = PCA(n_components=narx_vector_length)
        pca.fit(X)

        if torch.isnan(pca.explained_variance_).any().item():
            print("Warning the pca failed. Explained variance is nan. The data matrix may be badly conditioned or not scaled.")

        pca_encoder = PCA_Encoder(pca_module=pca, in_scaler=scaler)
        torch.save(pca_encoder, save_cfg.get("save_path"))
