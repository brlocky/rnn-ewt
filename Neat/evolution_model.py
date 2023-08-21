
from abc import ABC, abstractmethod
import copy
import math
from re import S
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from nn_models import RNNModel, NNModel

from i_evolution_mode import IEvolutionModel


class EvolutionModel(IEvolutionModel):
    def create_model(self):
        # return RNNModel(2, 64, 1)
        return NNModel(2, 32, 1)

    def create_children(self, generation: int, id: int):
        emodel = EvolutionModel(None, generation, id, self.mutation_factor,
                                self.mutation_strength, self.mutation_probability)
        model_weights = copy.deepcopy(self.model.state_dict())
        xmodel = self.create_model()
        for attr_name in model_weights:
            source_weight = model_weights[attr_name]
            if torch.rand(1) < self.mutation_probability:
                model_weights[attr_name] += torch.randn_like(
                    source_weight) * self.mutation_strength

        xmodel.load_state_dict(model_weights)

        emodel.model = xmodel
        return emodel

    def get_data(self, device):
        self.num_samples = 10000

        # Define the dataset (dummy data for demonstration)
        # X1: Input features, y1: Target labels
        np.random.seed(42)
        num_samples = 100
        X = np.random.rand(num_samples, 2).astype(np.float32)
        y = (X[:, 0] * X[:, 1]).reshape(-1, 1).astype(np.float32)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.5, random_state=42, shuffle=True
        )

        # Convert data to PyTorch tensors and move to GPU if available
        X_train_tensor = torch.tensor(X_train, device=device)
        y_train_tensor = torch.tensor(y_train, device=device)
        X_val_tensor = torch.tensor(X_val, device=device)
        y_val_tensor = torch.tensor(y_val, device=device)

        return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
