import os
import csv
import time
from typing import List, Any, Dict, Union
from dataclasses import dataclass

from sympy import true

from lutech_quantum_cnn.net import ClassicNet, HybridNet

import torch
from torch import Tensor, no_grad, manual_seed
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.modules.loss import MSELoss, CrossEntropyLoss

manual_seed(42)

@dataclass
class TrainingResult:
    avg_epoch_train_costs: List[Tensor]
    avg_epoch_train_accuracies: List[Tensor]
    avg_epoch_test_costs: List[Tensor]
    avg_epoch_test_accuracies: List[Tensor]
    models: List[Dict[str, Any]]
    plot_path: str

class Trainer:
    def __init__(
        self,
        model: Union[ClassicNet, HybridNet],
        train_loader: DataLoader,
        test_loader: DataLoader,
        loss_fn: Union[MSELoss, CrossEntropyLoss],
        epochs: int,
        learning_rate: float,
        torch_device : torch.device
    ):
        # Set device
        self.device = torch_device
        
        # Move model to device BEFORE wrapping DataParallel
        model = model.to(self.device)
        self.model = DataParallel(model)
        
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

        if model.prob is None:
            path = 'classical'
        else:
            path = model.feature_map_reps + model.feature_map + model.ansatz_reps + model.ansatz + str(model.prob*100) + '%'

        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('plots'):
            os.makedirs('plots')

        self.csv_path = os.path.join('results', path)
        self.plot_path = os.path.join('plots', path)

    def train_and_validate(self) -> Union[TrainingResult, None]:
        model = self.model
        results = TrainingResult([], [], [], [], [], self.plot_path)

        with open(self.csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [
                    "Epoch",
                    "Train Loss",
                    "Train Accuracy",
                    "Test Loss",
                    "Test Accuracy",
                ]
            )

            for epoch in range(self.epochs):
                start_epoch_time = time.time()
                epoch_train_costs: List[Tensor] = []
                epoch_train_accuracies: List[Tensor] = []
                epoch_test_costs: List[Tensor] = []
                epoch_test_accuracies: List[Tensor] = []

                optimizer = Adam(params=model.parameters(), lr=self.learning_rate)

                model.train()

                for batch_index, (inputs, labels) in enumerate(self.train_loader):
                    inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                    optimizer.zero_grad()

                    output = model(inputs).to(self.device)

                    _, predicted_labels = torch.max(output, 1)
                    predicted_labels = predicted_labels.to(self.device)
                    true_labels = torch.argmax(labels, dim=1)
                    correct_train_predictions: Tensor = (predicted_labels == true_labels).sum()
                    train_accuracy: Tensor = correct_train_predictions / inputs.size(0)

                    train_cost_fn: Tensor = self.loss_fn(output, labels.float())
                    train_cost_fn.backward()
                    optimizer.step()

                    epoch_train_costs.append(train_cost_fn.detach())
                    epoch_train_accuracies.append(train_accuracy.detach())

                model.eval()
                with no_grad():
                    for batch_index, (inputs, labels) in enumerate(self.test_loader):
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        output = model(inputs).to(self.device)

                        test_cost_fn = self.loss_fn(output.float(), labels.float())

                        _, predicted_labels = torch.max(output, 1)
                        predicted_labels = predicted_labels.to(self.device)
                        true_labels = torch.argmax(labels, dim=1)
                        correct_predictions: Tensor = (predicted_labels == true_labels).sum()
                        test_accuracy: Tensor = correct_predictions / inputs.size(0)

                        epoch_test_costs.append(test_cost_fn.detach())
                        epoch_test_accuracies.append(test_accuracy.detach())

                avg_epoch_train_cost = sum(epoch_train_costs) / len(epoch_train_costs)
                avg_epoch_train_accuracy = sum(epoch_train_accuracies) / len(epoch_train_accuracies)
                avg_epoch_test_cost = sum(epoch_test_costs) / len(epoch_test_costs)
                avg_epoch_test_accuracy = sum(epoch_test_accuracies) / len(epoch_test_accuracies)

                results.models.append(model.state_dict())

                results.avg_epoch_train_costs.append(avg_epoch_train_cost)
                results.avg_epoch_train_accuracies.append(avg_epoch_train_accuracy)
                results.avg_epoch_test_costs.append(avg_epoch_test_cost)
                results.avg_epoch_test_accuracies.append(avg_epoch_test_accuracy)

                csvwriter.writerow(
                    [
                        epoch,
                        avg_epoch_train_cost.item(),
                        avg_epoch_train_accuracy.item(),
                        avg_epoch_test_cost.item(),
                        avg_epoch_test_accuracy.item(),
                    ]
                )
                end_epoch_time = time.time()
                epoch_time = end_epoch_time - start_epoch_time

                print(
                    f"EPOCH: {epoch+1}/{self.epochs} ||| TIME: {int(epoch_time)}s"
                    f" ||| TRAIN COST: {avg_epoch_train_cost.item():.2f}"
                    f" ||| TRAIN ACCURACY: {avg_epoch_train_accuracy.item():.2f}"
                    f" ||| TEST COST: {avg_epoch_test_cost.item():.2f}"
                    f" ||| TEST ACCURACY: {avg_epoch_test_accuracy.item():.2f}"
                )

        return results
