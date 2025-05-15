import matplotlib.pyplot as plt
from src_pennylane.training import TrainingResult

def plot_results(results: TrainingResult):
    """Plot the results of training and test.

    Arguments:
    ----------
    results : TrainingResult
        The results to be plotted.
    """

    avg_epoch_train_costs = results.avg_epoch_train_costs
    avg_epoch_train_accuracies = results.avg_epoch_train_accuracies
    avg_epoch_test_costs = results.avg_epoch_test_costs
    avg_epoch_test_accuracies = results.avg_epoch_test_accuracies

    plt.figure(figsize=(10, 12))  # Adjusting figure size for four subplots

    # Plotting the first subplot (train cost)
    plt.subplot(4, 1, 1)
    plt.plot(avg_epoch_train_costs, label="Train cost function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cost on the training set")
    plt.legend()
    plt.grid(True)

    # Plotting the second subplot (test cost)
    plt.subplot(4, 1, 2)
    plt.plot(avg_epoch_test_costs, label="Test cost function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cost on the test set")
    plt.legend()
    plt.grid(True)

    # Plotting the third subplot (train accuracies)
    plt.subplot(4, 1, 3)
    plt.plot(avg_epoch_train_accuracies, label="Train accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on the training set")
    plt.legend()
    plt.grid(True)

    # Plotting the fourth subplot (test accuracies)
    plt.subplot(4, 1, 4)
    plt.plot(avg_epoch_test_accuracies, label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy on the test set")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
