import matplotlib.pyplot as plt
import os
from lutech_quantum_cnn.training import TrainingResult

def plot_results(results: TrainingResult):
    """Plot the results of training and test, saving each plot individually as a PDF
    into a specified folder.

    Arguments:
    ----------
    results : TrainingResult
        The results to be plotted.
    output_folder : str, optional
        The name of the folder where the plots will be saved.
        Defaults to "training_plots".
    """
    output_folder = results.plot_path

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    avg_epoch_train_costs = results.avg_epoch_train_costs
    avg_epoch_train_accuracies = results.avg_epoch_train_accuracies
    avg_epoch_test_costs = results.avg_epoch_test_costs
    avg_epoch_test_accuracies = results.avg_epoch_test_accuracies

    # Plot and save Train Cost as PDF
    plt.figure(figsize=(6, 4))
    plt.plot(avg_epoch_train_costs, label="Train cost function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cost on the training set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "train_cost.pdf")) # Save in the specified folder
    plt.close()

    # Plot and save Test Cost as PDF
    plt.figure(figsize=(6, 4))
    plt.plot(avg_epoch_test_costs, label="Test cost function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cost on the test set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "test_cost.pdf")) # Save in the specified folder
    plt.close()

    # Plot and save Train Accuracy as PDF
    plt.figure(figsize=(6, 4))
    plt.plot(avg_epoch_train_accuracies, label="Train accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on the training set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "train_accuracy.pdf")) # Save in the specified folder
    plt.close()

    # Plot and save Test Accuracy as PDF
    plt.figure(figsize=(6, 4))
    plt.plot(avg_epoch_test_accuracies, label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on the test set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "test_accuracy.pdf")) # Save in the specified folder
    plt.close()