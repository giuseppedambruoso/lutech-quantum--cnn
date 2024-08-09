""" This library allows to investigate the performance of a noisy hybrid quantum-classical convolutional neural networks and compare it to its classical counterpart.

The library allows to create, train and validate CNNs composed of:
- a single quantum convolutional layer;
- a flattening operation;
- a single fully-connected layer;
- a softmax layer.

As in the classical case, the quantum convolutional layer acts on the input image by extracting sliding blocks from it and performing an operation - called filtering - on each of these blocks. However, unlike the ordinary case, the filtering operation relies upon the execution of a (variational) quantum circuit. More specifically, the $N$ pixel values of each sliding block are mapped into a $N$-qubit variational quantum circuit (VQC) by means of a particular arrangement of non-trainable parametric quantum gates, which compose the so-called "feature map". The remaining part of the VQC, usually referred to as "ansatz", features trainable parametric quantum gates. Finally, the VQC is executed a number of times and measurements in the computational basis on the output quantum state are performed, so to give an estimate for its $2^N$ probability coefficients. The obtained $2^N$ real values are fed into the $2^N$ output images, which are then created by means of a single filter. The number of trainable parameters in the ansatz can be chosen arbitrarily, so that one could in principle explore the chance to obtain better performance with fewer parameters.

The parameters update of the whole net is performed by means of the mini-batch gradient descent algorithm. In order to differentiate the quantum filter's output, the parameter-shift rule is applied.

The whole model is inspired to that proposed by Junhua Liu (see README.md).

Users have to possibility to decide wether to make the execution of the VQC contained within the filter noiseless or noisy. Moreover, they can choose to introduce one or more noise models, each one with an arbitary probability.
"""

# Import necessary libraries
from src.dataset import load_dataset
from src.noise import create_backend
from src.qnn import QNN
from src.net import create_cnn
from src.plot import plot_results
from src.training import Trainer

from torch import manual_seed
from torch.nn import MSELoss, CrossEntropyLoss

import random
import numpy as np
import hydra
from omegaconf import DictConfig
from typing import Union

import logging

logging.getLogger("qiskit").setLevel(logging.CRITICAL)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    # Define configuration
    SEED = config["seed"]
    BATCH_SIZE = config["batch_size"]
    LEARNING_RATE = config["learning_rate"]
    EPOCHS = config["epochs"]
    NUM_QUBITS = config["num_qubits"]
    SHOTS = config["shots"]
    feature_map_name = config["feature_map_name"]
    feature_map_entanglement = config["feature_map_entanglement"]
    FEATURE_MAP_DEPTH = config["feature_map_depth"]
    ansatz_name = config["ansatz_name"]
    ansatz_entanglement = config["ansatz_entanglement"]
    ANSATZ_DEPTH = config["ansatz_depth"]
    CONVOLUTION_OUT_CHANNELS = config["convolution_out_channels"]
    dataset_folder_path = config["dataset_folder_path"]
    error_name = config["error_name"]
    error_probability = config["error_probability"]
    hybrid = config["hybrid"]
    csv_path = config["csv_path"]

    # Set random seed for reproducibility
    manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Load data
    train_loader, validation_loader, _ = load_dataset(
        dataset_folder_path=dataset_folder_path,
        batch_size=BATCH_SIZE,
    )

    # Create the backend
    backend = create_backend(
        error_name=error_name,
        error_probability=error_probability,
    )

    # Create the quantum layer
    qnn = QNN(
        num_qubits=NUM_QUBITS,
        feature_map_name=feature_map_name,
        feature_map_depth=FEATURE_MAP_DEPTH,
        feature_map_entanglement=feature_map_entanglement,
        ansatz_name=ansatz_name,
        ansatz_depth=ANSATZ_DEPTH,
        ansatz_entanglement=ansatz_entanglement,
        backend=backend,
        shots=SHOTS,
    )

    # Create the cnn
    model = create_cnn(
        hybrid=hybrid,
        dataset_folder_path=dataset_folder_path,
        train_loader=train_loader,
        kernel_size=NUM_QUBITS,
        convolution_output_channels=CONVOLUTION_OUT_CHANNELS,
        quantum_filter=qnn.qnn,
    )

    # Define the loss function
    loss_fn: Union[MSELoss, CrossEntropyLoss]
    if config["loss_function"] == "MSE":
        loss_fn = MSELoss()
    elif config["loss_function"] == "CrossEntropy":
        loss_fn = CrossEntropyLoss()

    # Perform training and validation of the model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        csv_path=csv_path,
    )

    # Get results
    results = trainer.train_and_validate()

    # Plot results
    if results is not None:
        plot_results(results)


if __name__ == "__main__":
    main()