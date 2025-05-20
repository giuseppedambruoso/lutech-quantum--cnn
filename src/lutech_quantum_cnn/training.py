import os
import csv
import time
from typing import List, Any, Dict, Union
from dataclasses import dataclass

from lutech_quantum_cnn.net import ClassicNet, HybridNet

from torch import Tensor, no_grad, argmax, manual_seed
from torch import max as torch_max
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
    """Class to train and validate a module.

    Attributes
    ----------
    model : Union[ClassicNet,HybridNet]
        The model to be trained.
    train_loader : DataLoader
        The data loader of the training set.
    test_loader : DataLoader
        The data loader of the test set.
    loss_fn : Union[MSELoss, CrossEntropyLoss]
        The loss function used to optimize the parameters.
    epochs : int
        The number of epochs of the training.
    learning_rate : float
        The learning rate used by the optimizer.

    Methods
    -------
    train_and_validate
        Performs both training and test on the dataset and saves the
        metrics along the way.
    """

    def __init__(
        self,
        model: Union[ClassicNet, HybridNet],
        train_loader: DataLoader,
        test_loader: DataLoader,
        loss_fn: Union[MSELoss, CrossEntropyLoss],
        epochs: int,
        learning_rate: float,
    ):
        self.model = DataParallel(model)
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

        path : str
        if model.prob is None:
            path = 'classical'
        else :
            path = str(model.prob) + '%'

        # Create the output folder if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('plots'):
            os.makedirs('plots')

        self.csv_path = os.path.join('results', path + '.csv')
        self.plot_path = os.path.join('plots', path + '.pdf')

    def train_and_validate(self) -> Union[TrainingResult, None]:
        model = self.model
        # Initialize the results object
        results = TrainingResult([], [], [], [], [], self.plot_path)

        with open(self.csv_path, "w", newline="") as csvfile:
            # Create a csv writer object
            csvwriter = csv.writer(csvfile)

            # Write the header
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

                # Initialize the optimizer
                optimizer = Adam(params=model.parameters(), lr=self.learning_rate)

                # Train the model
                model.train()

                for batch_index, (inputs, labels) in enumerate(self.train_loader):
                    # print('EPOCH: ', epoch + 1)
                    # print('TRAIN BATCH: ', batch_index + 1)
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

                    # print(
                    #     "\r\033[KEPOCH: "
                    #     + str(epoch + 1)
                    #     + "/"
                    #     + str(self.epochs)
                    #     + "|||"
                    #     + "TRAIN: "
                    #     + str(batch_index + 1)
                    #     + "/"
                    #     + str(len(self.train_loader))
                    #     + "|||"
                    #     + "TIME: "
                    #     + str(int(train_time))
                    #     + "s"
                    #     + "|||"
                    #     + "COST: "
                    #     + str(train_cost_fn.item()),
                    #     end="",
                    # )

                model.eval()
                with no_grad():
                    for batch_index, (inputs, labels) in enumerate(
                        self.test_loader
                    ):
                        # print('TEST BATCH: ', batch_index + 1)
                        output = model(inputs)

                        # Compute cost function
                        test_cost_fn = self.loss_fn(
                            output.float(), labels.float()
                        )

                        # Compute correct predictions
                        _, predicted_labels = torch_max(output, 1)
                        true_labels = argmax(labels, dim=1)
                        correct_predictions: Tensor = (
                            predicted_labels == true_labels
                        ).sum()
                        test_accuracy: Tensor = correct_predictions / inputs.size(
                            0
                        )

                        # Add metrics to lists
                        epoch_test_costs.append(test_cost_fn)
                        epoch_test_accuracies.append(test_accuracy)


                        # print(
                        #     "\r\033[KEPOCH: "
                        #     + str(epoch + 1)
                        #     + "/"
                        #     + str(self.epochs)
                        #     + "|||"
                        #     + "TEST: "
                        #     + str(batch_index + 1)
                        #     + "/"
                        #     + str(len(self.test_loader))
                        #     + "|||"
                        #     + "ACCURACY: "
                        #     + str(test_accuracy.item()),
                        #     end="",
                        # )

                # Compute epoch averages for graphical representation
                avg_epoch_train_cost = sum(epoch_train_costs) / len(epoch_train_costs)
                avg_epoch_train_accuracy = sum(epoch_train_accuracies) / len(
                    epoch_train_accuracies
                )
                avg_epoch_test_cost = sum(epoch_test_costs) / len(
                    epoch_test_costs
                )
                avg_epoch_test_accuracy = sum(epoch_test_accuracies) / len(
                    epoch_test_accuracies
                )

                # Record the model's parameters
                results.models.append(model.state_dict())
                
                if (
                    type(avg_epoch_train_cost) == Tensor
                    and type(avg_epoch_train_accuracy) == Tensor
                    and type(avg_epoch_test_cost) == Tensor
                    and type(avg_epoch_test_accuracy) == Tensor
                ):
                    
                    # Record training metrics
                    results.avg_epoch_train_costs.append(avg_epoch_train_cost.detach())
                    results.avg_epoch_train_accuracies.append(
                        avg_epoch_train_accuracy.detach()
                    )
                    results.avg_epoch_test_costs.append(
                        avg_epoch_test_cost.detach()
                    )
                    results.avg_epoch_test_accuracies.append(
                        avg_epoch_test_accuracy.detach()
                    )

                    # Update csv file
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
                        "EPOCH: "
                        + str(epoch + 1)
                        + "/"
                        + str(self.epochs)
                        + "|||"
                        + "TIME: "
                        + str(int(epoch_time))
                        + "s"
                        + "|||"
                        + "TRAIN COST: "
                        + str(round(avg_epoch_train_cost.item(),2))
                        + "|||"
                        + "TRAIN ACCURACY: "
                        + str(round(avg_epoch_train_accuracy.item(),2))
                        + "|||"
                        + "TEST COST: "
                        + str(round(avg_epoch_test_cost.item(),2))
                        + "|||"
                        + "TEST ACCURACY: "
                        + str(round(avg_epoch_test_accuracy.item(),2)),
                    )
        return results