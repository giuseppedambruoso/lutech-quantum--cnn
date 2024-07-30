from typing import List, Any, Dict

from torch import no_grad
from torch.optim.adam import Adam
from torch import Tensor
from torch import max as torch_max
from torch.nn import MSELoss, CrossEntropyLoss, DataParallel

from quantorch.src2.net2 import ClassicNet, HybridNet

def train_and_validate(
        model: DataParallel[ClassicNet | HybridNet],
        train_loader: Tensor,
        validation_loader: Tensor,
        learning_rate: float,
        loss_fn : MSELoss | CrossEntropyLoss,
        epochs: int,
) -> None | tuple[
    list[Tensor | float],
    list[Tensor | float],
    list[Tensor | float],
    list[Tensor | float],
    list[Dict[str, Any]]
]:
    """
    Train and validate a neural network model.

    Args:
        model (ClassicNet | HybridNet): The neural network model to 
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
    avg_epoch_train_costs : List[Tensor | float] = []
    avg_epoch_train_accuracies : List[Tensor | float] = []
    avg_epoch_validation_costs: List[Tensor | float] = []
    avg_epoch_validation_accuracies : List[Tensor | float] = []
    models : List[Dict[str, Any]] = []

    for epoch in range(epochs):
        epoch_train_costs : List[Tensor] = []
        epoch_train_accuracies: List[Tensor] = []
        epoch_validation_costs : List[Tensor] = []
        epoch_validation_accuracies : List[Tensor] = []
        
        # Train the model
        model.train()
        for batch_index, (input, labels) in enumerate(train_loader):
            print('TRAIN'+str(epoch)+str(batch_index))
            optimizer = Adam(params=model.parameters(), lr=learning_rate)
            optimizer.zero_grad()

            output = model(input)

            # Compute accuracy
            _, predicted_labels = torch_max(output, 1)
            correct_train_predictions : Tensor = \
                (predicted_labels == labels).sum()
            train_accuracy : Tensor = correct_train_predictions / input.size(0) 

            # Optimize parameters
            train_cost_fn : Tensor = loss_fn(output, labels.float())
            train_cost_fn.backward()
            optimizer.step()

            # Add metrics to lists
            epoch_train_costs.append(train_cost_fn)
            epoch_train_accuracies.append(train_accuracy)

        print()
        
        # Evaluate the model
        model.eval()
        with no_grad():
            for batch_index, (input, labels) in enumerate(validation_loader):
                print('VALIDATION'+str(epoch)+str(batch_index))
                output = model(input)

                # Compute cost function
                validation_cost_fn = \
                    loss_fn(output.float(), labels.float())

                # Compute correct predictions
                _, predicted_labels = torch_max(output, 1)
                correct_predictions : Tensor = (predicted_labels == labels).sum()
                validation_accuracy : Tensor = \
                    correct_predictions / input.size(0)
                
                # Add metrics to lists
                epoch_validation_costs.append(validation_cost_fn)
                epoch_validation_accuracies.append(validation_accuracy)

        # Compute epoch averages for graphical representation
        avg_epoch_train_cost = sum(epoch_train_costs)/len(epoch_train_costs)
        evg_epoch_train_accuracy = \
            sum(epoch_train_accuracies)/len(epoch_train_accuracies)
        avg_epoch_validation_cost = \
            sum(epoch_validation_costs) / len(epoch_validation_costs)
        avg_epoch_validation_accuracy = \
            sum(epoch_validation_accuracies) / len(epoch_validation_accuracies)
        
        # Add metrics to lists
        avg_epoch_train_costs.append(avg_epoch_train_cost)
        avg_epoch_train_accuracies.append(evg_epoch_train_accuracy)
        avg_epoch_validation_costs.append(avg_epoch_validation_cost)
        avg_epoch_validation_accuracies.append(avg_epoch_validation_accuracy)

        # Save the model
        models.append(model.state_dict())

        return (
            avg_epoch_train_costs,
            avg_epoch_train_accuracies,
            avg_epoch_validation_costs,
            avg_epoch_validation_accuracies,
            models
        )

print()