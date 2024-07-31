from typing import List, Any, Dict, Union

from torch import no_grad
from torch.optim.adam import Adam
from torch import Tensor, argmax, tensor
from torch import max as torch_max
from torch.nn import MSELoss, CrossEntropyLoss, DataParallel

from dataclasses import dataclass

from quantorch.src2.net2 import ClassicNet, HybridNet

import csv

@dataclass
class TrainingResult:
    avg_epoch_train_costs : List[Union[Tensor, float]]
    avg_epoch_train_accuracies : List[Union[Tensor, float]]
    avg_epoch_validation_costs : List[Union[Tensor, float]]
    avg_epoch_validation_accuracies : List[Union[Tensor, float]]
    models : List[Dict[str, Any]]

def train_and_validate(
        model: DataParallel[Union[ClassicNet, HybridNet]],
        train_loader: Tensor,
        validation_loader: Tensor,
        learning_rate: float,
        loss_fn : Union[MSELoss, CrossEntropyLoss],
        epochs: int,
) -> Union[TrainingResult, None]:
    """
    Train and validate a neural network model.

    Args:
        model (ClassicNet or HybridNet): The neural network model to 
            be trained and validated.
        train_loader (Tensor): DataLoader for training data.
        validation_loader (Tensor): DataLoader for validation data.
        learning_rate (float): Learning rate for the optimizer.
        loss_fn (_Loss): Loss function used for training.
        epochs (int): Number of epochs to train the model.

    Returns:
        tuple[List[Tensor]]: Returns a tuple containing four lists:
            - avg_epoch_train_costs: Average training costs per epoch.
            - avg_epoch_train_accuracies: Average training accuracies 
                per epoch.
            - avg_epoch_validation_costs: Average validation costs per
                epoch.
            - avg_epoch_validation_accuracies: Average validation accuracies
                per epoch.
    """
    with open('training_metrics.csv', 'w', newline='') as csvfile:
        # Create a csv writer object
        csvwriter = csv.writer(csvfile)
        
        # Write the header
        csvwriter.writerow(
            ['Epoch',
            'Train Loss',
            'Train Accuracy',
            'Validation Loss',
            'Validation Accuracy'
            ]
        )
        
        
        results = TrainingResult([], [], [], [], [])
        optimizer = Adam(params=model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_train_costs : List[Tensor] = []
            epoch_train_accuracies: List[Tensor] = []
            epoch_validation_costs : List[Tensor] = []
            epoch_validation_accuracies : List[Tensor] = []

            # Train the model
            model.train()
            for batch_index, (input, labels) in enumerate(train_loader):
                print('\r\033[KEPOCH: '+str(epoch+1)+'/'+str(epochs)+' '+\
                      'TRAIN: '+str(batch_index+1)+'/16', end='')
                optimizer.zero_grad()

                output = model(input)

                # Compute accuracy
                _, predicted_labels = torch_max(output, 1)
                true_labels = argmax(labels, dim=1)
                correct_train_predictions : Tensor = \
                    (predicted_labels == true_labels).sum()
                train_accuracy : Tensor = correct_train_predictions / input.size(0) 

                # Optimize parameters
                train_cost_fn : Tensor = loss_fn(output, labels.float())
                train_cost_fn.backward()
                optimizer.step()

                # Add metrics to lists
                epoch_train_costs.append(train_cost_fn)
                epoch_train_accuracies.append(train_accuracy)

            # Evaluate the model
            model.eval()
            with no_grad():
                for batch_index, (input, labels) in enumerate(validation_loader):
                    print('\r\033[KEPOCH: '+str(epoch+1)+'/'+str(epochs)+' '+\
                      'VALIDATION: '+str(batch_index+1)+'/4', end='')
                    output = model(input)

                    # Compute cost function
                    validation_cost_fn = \
                        loss_fn(output.float(), labels.float())

                    # Compute correct predictions
                    _, predicted_labels = torch_max(output, 1)
                    true_labels = argmax(labels, dim=1)
                    correct_predictions : Tensor = \
                        (predicted_labels == true_labels).sum()
                    validation_accuracy : Tensor = \
                        correct_predictions / input.size(0)

                    # Add metrics to lists
                    epoch_validation_costs.append(validation_cost_fn)
                    epoch_validation_accuracies.append(validation_accuracy)

            # Compute epoch averages for graphical representation
            avg_epoch_train_cost = sum(epoch_train_costs)/len(epoch_train_costs)
            avg_epoch_train_accuracy = \
                sum(epoch_train_accuracies)/len(epoch_train_accuracies)
            avg_epoch_validation_cost = \
                sum(epoch_validation_costs) / len(epoch_validation_costs)
            avg_epoch_validation_accuracy = \
                sum(epoch_validation_accuracies) / len(epoch_validation_accuracies)

            # Record results
            results.avg_epoch_train_costs.append(\
                avg_epoch_train_cost.detach())
            results.avg_epoch_train_accuracies.append(\
                avg_epoch_train_accuracy.detach())
            results.avg_epoch_validation_costs.append(\
                avg_epoch_validation_cost.detach())
            results.avg_epoch_validation_accuracies.append(\
                avg_epoch_validation_accuracy.detach())
            results.models.append(model.state_dict())
            
            # Update csv file
            csvwriter.writerow(
                [epoch,
                avg_epoch_train_cost.item(),
                avg_epoch_train_accuracy.item(),
                avg_epoch_validation_cost.item(),
                avg_epoch_validation_accuracy.item()]
            )

    return results