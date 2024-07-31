from torch import Tensor
import matplotlib.pyplot as plt
from quantorch.src2.training2 import TrainingResult

def plot_results(results : TrainingResult):
    
    avg_epoch_train_costs = results.avg_epoch_train_costs
    avg_epoch_train_accuracies = results.avg_epoch_train_accuracies
    avg_epoch_validation_costs = results.avg_epoch_validation_costs
    avg_epoch_validation_accuracies = \
        results.avg_epoch_validation_accuracies
    
    plt.figure(figsize=(10, 12))  # Adjusting figure size for four subplots

    # Plotting the first subplot (train cost)
    plt.subplot(4, 1, 1)
    plt.plot(
        avg_epoch_train_costs,
        label='Train cost function'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Cost Function')
    plt.legend()
    plt.grid(True)

    # Plotting the second subplot (validation cost)
    plt.subplot(4, 1, 2)
    plt.plot(
        avg_epoch_validation_costs,
        label='Train cost function'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Cost Function')
    plt.legend()
    plt.grid(True)

    # Plotting the third subplot (train accuracies)
    plt.subplot(4, 1, 3)
    plt.plot(
        avg_epoch_train_accuracies,
        label='Train accuracy'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.legend()
    plt.grid(True)

    # Plotting the fourth subplot (validation accuracies)
    plt.subplot(4, 1, 4)
    plt.plot(
        avg_epoch_validation_accuracies,
        label='Validation accuracy'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()