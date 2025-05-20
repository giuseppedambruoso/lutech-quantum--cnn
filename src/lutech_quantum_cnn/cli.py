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
from lutech_quantum_cnn.dataset import load_dataset, num_classes
from lutech_quantum_cnn.net import create_cnn
from lutech_quantum_cnn.plot import plot_results
from lutech_quantum_cnn.training import Trainer

from torch.nn import MSELoss, CrossEntropyLoss

import os
import hydra
from omegaconf import DictConfig
from typing import Union, List
from torch import manual_seed
import pennylane as qml
from pennylane.devices.device_api import Device

manual_seed(42)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    # Define configuration
    dataset_folder_path : str
    if config["dataset_folder_path"] == 'Tetris':
        dataset_folder_path = os.getcwd() + \
            "/src/lutech_quantum_cnn/" + config["dataset_folder_path"]
    else:
        dataset_folder_path = config["dataset_folder_path"]
    BATCH_SIZE = config["batch_size"]
    noise = config["noise"]
    NOISE_PROB = config["noise_probability"]
    KERNEL_SIZE = config["kernel_size"]
    feature_map = config["feature_map"]
    ansatz = config["ansatz"]
    FEATURE_MAP_REPS = config["feature_map_reps"]
    ANSATZ_REPS = config["ansatz_reps"]
    CLASSES : int = num_classes(dataset_folder_path=dataset_folder_path)
    show_circuit = config["show_circuit"]
    loss_fn : Union[MSELoss, CrossEntropyLoss] = \
        MSELoss() if config["loss_function"] == "MSE" \
        else CrossEntropyLoss()
    EPOCHS = config["epochs"]
    LEARNING_RATE = config["learning_rate"]
    
    # Set random seed for reproducibility
    manual_seed(42)
    
    # Load data
    train_loader, test_loader, _ = load_dataset(
        dataset_folder_path=dataset_folder_path,
        batch_size=BATCH_SIZE
    )

    # Create device
    num_qubits : int = int(KERNEL_SIZE * KERNEL_SIZE)
    wires : List = list(range(num_qubits))
    # device = qml.device("default.mixed", wires=wires)
    device : Device | None
    if isinstance(NOISE_PROB, float):
        if NOISE_PROB > 1 or NOISE_PROB < 0:
            raise ValueError("NOISE_PROB must be in the range [0, 1]")
        elif NOISE_PROB > 0:
            device = qml.device("default.mixed", wires=wires)
        elif NOISE_PROB == 0:
            device = qml.device("default.qubit", wires=wires)
    else:
        device = None

    # Create the cnn
    model = create_cnn(
        train_loader = test_loader,
        dataset_folder_path = dataset_folder_path,
        kernel_size = KERNEL_SIZE,
        device = device,
        noise = noise,
        noise_prob = NOISE_PROB,
        feature_map = feature_map,
        ansatz = ansatz,
        feature_map_reps = FEATURE_MAP_REPS,
        ansatz_reps = ANSATZ_REPS,
        classes = CLASSES,
        show_circuit = show_circuit,
    )

    # Perform training and validation of the model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # Get results
    results = trainer.train_and_validate()

    # Plot results
    if results is not None:
        plot_results(results)


if __name__ == "__main__":
    main()