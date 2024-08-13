import csv
import time
from typing import List, Any, Dict, Union
from dataclasses import dataclass

from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch import Tensor, no_grad, argmax
from torch import max as torch_max
from torch.nn import DataParallel
from torch.nn.modules.loss import MSELoss, CrossEntropyLoss

from src.net import ClassicNet, HybridNet


@dataclass
class TrainingResult:
    avg_epoch_train_costs: List[Tensor]
    avg_epoch_train_accuracies: List[Tensor]
    avg_epoch_validation_costs: List[Tensor]
    avg_epoch_validation_accuracies: List[Tensor]
    models: List[Dict[str, Any]]


class Trainer:
    """Class to train and validate a module.

    Attributes
    ----------
    model : DataParallel[Union[ClassicNet,HybridNet]]
        The model to be trained.
    train_loader : DataLoader
        The data loader of the training set.
    validation_loader : DataLoader
        The data loader of the validation set.
    loss_fn : Union[MSELoss, CrossEntropyLoss]
        The loss function used to optimize the parameters.
    epochs : int
        The number of epochs of the training.
    learning_rate : float
        The learning rate used by the optimizer.
    csv_path : str
        The path of the csv file into which the training metrics are saved.

    Methods
    -------
    train_and_validate
        Performs both training and validation on the dataset and saves the
        metrics along the way.
    """

    def __init__(
        self,
        model: DataParallel[Union[ClassicNet, HybridNet]],
        train_loader: DataLoader,
        validation_loader: DataLoader,
        loss_fn: Union[MSELoss, CrossEntropyLoss],
        epochs: int,
        learning_rate: float,
        csv_path: str,
    ):
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.csv_path = csv_path

    def train_and_validate(self) -> Union[TrainingResult, None]:
        model = self.model
        # Initialize the results object
        results = TrainingResult([], [], [], [], [])

        with open(self.csv_path, "w", newline="") as csvfile:
            # Create a csv writer object
            csvwriter = csv.writer(csvfile)

            # Write the header
            csvwriter.writerow(
                [
                    "Epoch",
                    "Train Loss",
                    "Train Accuracy",
                    "Validation Loss",
                    "Validation Accuracy",
                ]
            )

            for epoch in range(self.epochs):
                epoch_train_costs: List[Tensor] = []
                epoch_train_accuracies: List[Tensor] = []
                epoch_validation_costs: List[Tensor] = []
                epoch_validation_accuracies: List[Tensor] = []

                # Initialize the optimizer
                optimizer = Adam(params=model.parameters(), lr=self.learning_rate)

                # Train the model
                model.train()

                for batch_index, (inputs, labels) in enumerate(self.train_loader):
                    # Start recording time
                    start_train_time = time.time()

                    optimizer.zero_grad()

                    output = model(inputs)

                    # Compute accuracy
                    _, predicted_labels = torch_max(output, 1)
                    true_labels = argmax(labels, dim=1)
                    correct_train_predictions: Tensor = (
                        predicted_labels == true_labels
                    ).sum()
                    train_accuracy: Tensor = correct_train_predictions / inputs.size(0)

                    # Optimize parameters
                    train_cost_fn: Tensor = self.loss_fn(output, labels.float())
                    train_cost_fn.backward()
                    optimizer.step()

                    # Add metrics to lists
                    epoch_train_costs.append(train_cost_fn)
                    epoch_train_accuracies.append(train_accuracy)

                    # End recording time and compute total time
                    end_train_time = time.time()
                    train_time = end_train_time - start_train_time

                    print(
                        "\r\033[KEPOCH: "
                        + str(epoch + 1)
                        + "/"
                        + str(self.epochs)
                        + " "
                        + "TRAIN: "
                        + str(batch_index + 1)
                        + "/"
                        + str(len(self.train_loader))
                        + " TIME: "
                        + str(train_time)
                        + "s",
                        end="",
                    )

                model.eval()
                with no_grad():
                    for batch_index, (inputs, labels) in enumerate(
                        self.validation_loader
                    ):
                        output = model(inputs)

                        # Compute cost function
                        validation_cost_fn = self.loss_fn(
                            output.float(), labels.float()
                        )

                        # Compute correct predictions
                        _, predicted_labels = torch_max(output, 1)
                        true_labels = argmax(labels, dim=1)
                        correct_predictions: Tensor = (
                            predicted_labels == true_labels
                        ).sum()
                        validation_accuracy: Tensor = correct_predictions / inputs.size(
                            0
                        )

                        # Add metrics to lists
                        epoch_validation_costs.append(validation_cost_fn)
                        epoch_validation_accuracies.append(validation_accuracy)

                        print(
                            "\r\033[KEPOCH: "
                            + str(epoch + 1)
                            + "/"
                            + str(self.epochs)
                            + " "
                            + "VALIDATION: "
                            + str(batch_index + 1)
                            + "/"
                            + str(len(self.validation_loader)),
                            end="",
                        )

                # Compute epoch averages for graphical representation
                avg_epoch_train_cost = sum(epoch_train_costs) / len(epoch_train_costs)
                avg_epoch_train_accuracy = sum(epoch_train_accuracies) / len(
                    epoch_train_accuracies
                )
                avg_epoch_validation_cost = sum(epoch_validation_costs) / len(
                    epoch_validation_costs
                )
                avg_epoch_validation_accuracy = sum(epoch_validation_accuracies) / len(
                    epoch_validation_accuracies
                )

                # Record the model's parameters
                results.models.append(model.state_dict())
                
                if (
                    type(avg_epoch_train_cost) == Tensor
                    and type(avg_epoch_train_accuracy) == Tensor
                    and type(avg_epoch_validation_cost) == Tensor
                    and type(avg_epoch_validation_accuracy) == Tensor
                ):
                    
                    # Record training metrics
                    results.avg_epoch_train_costs.append(avg_epoch_train_cost.detach())
                    results.avg_epoch_train_accuracies.append(
                        avg_epoch_train_accuracy.detach()
                    )
                    results.avg_epoch_validation_costs.append(
                        avg_epoch_validation_cost.detach()
                    )
                    results.avg_epoch_validation_accuracies.append(
                        avg_epoch_validation_accuracy.detach()
                    )

                    # Update csv file
                    csvwriter.writerow(
                        [
                            epoch,
                            avg_epoch_train_cost.item(),
                            avg_epoch_train_accuracy.item(),
                            avg_epoch_validation_cost.item(),
                            avg_epoch_validation_accuracy.item(),
                        ]
                    )
        return results