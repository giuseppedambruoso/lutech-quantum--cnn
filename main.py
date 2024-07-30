# Import necessary libraries

from quantorch.src2.qnn2 import QNN
from quantorch.src2.dataset2 import load_dataset
from quantorch.src2.noise2 import create_backend
from quantorch.src2.net2 import create_cnn, ClassicNet, HybridNet
from quantorch.src2.training2 import train_and_validate
from quantorch.src2.plot2 import plot_results


from torch import manual_seed, Tensor
from torch.nn import MSELoss, CrossEntropyLoss, DataParallel

import random
import numpy as np
import hydra
from typing import List, Dict, Any
from omegaconf import DictConfig
from qiskit_aer import AerSimulator

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(config: DictConfig) -> None:

    # Define configuration
    SEED : int = config['seed']
    BATCH_SIZE : int = config['batch_size']
    LEARNING_RATE : float = config['learning_rate']
    EPOCHS : int = config['epochs']
    NUM_QUBITS : int = config['num_qubits']
    SHOTS : int | None = config['shots']
    feature_map_name : str = config['feature_map_name']
    feature_map_entanglement : str | List[List[int]] = \
        config['feature_map_entanglement']
    FEATURE_MAP_DEPTH : int = config['feature_map_depth']
    ansatz_name = config['ansatz_name']
    ansatz_entanglement :  str | List[List[int]] = \
        config['ansatz_entanglement']
    ANSATZ_DEPTH : int = config['ansatz_depth']
    CONVOLUTION_OUT_CHANNELS : int = config['quanvolution_out_channels']
    dataset_folder_name : str = config['dataset_folder_name']
    error_name : str = config['error_name']
    error_probability : float = config['error_probability']
    hybrid : bool = config['hybrid']
    
    # Set random seed for reproducibility
    manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Load data
    train_loader, validation_loader, _ = load_dataset(
        folder_name=dataset_folder_name,
        batch_size=BATCH_SIZE,
    )

    # Create backend
    backend : AerSimulator | None = create_backend(
        error_name=error_name,
        error_probability=error_probability,
    )

    # Create the quantum layer
    qnn : QNN = QNN(
        num_qubits=NUM_QUBITS,
        feature_map_name=feature_map_name,
        feature_map_depth=FEATURE_MAP_DEPTH,
        feature_map_entanglement=feature_map_entanglement,
        ansatz_name=ansatz_name,
        ansatz_depth=ANSATZ_DEPTH,
        ansatz_entanglement=ansatz_entanglement,
        backend=backend,
        shots=SHOTS
    )

    # Create the cnn
    model : DataParallel[ClassicNet | HybridNet] = create_cnn(
        hybrid=hybrid,
        dataset_folder_name=dataset_folder_name,
        train_set=train_loader,
        kernel_size=int(np.sqrt(NUM_QUBITS)),
        convolution_output_channels=CONVOLUTION_OUT_CHANNELS,
        quantum_filter=qnn.qnn
    )

    # Define the loss
    loss_fn : MSELoss | CrossEntropyLoss
    if config['loss_function'] == 'MSE':
        loss_fn = MSELoss()
    elif config['loss_function'] == 'CrossEntropy':
        loss_fn = CrossEntropyLoss()

    # Train and validate the cnn
    results : None | tuple[
    list[Tensor | float],
    list[Tensor | float],
    list[Tensor | float],
    list[Tensor | float],
    list[Dict[str, Any]]
    ] = train_and_validate(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        learning_rate=LEARNING_RATE,
        loss_fn=loss_fn,
        epochs=EPOCHS
    )

    # Plot the results
    if not results == None:
        plot_results(
            avg_epoch_train_costs=results[0],
            avg_epoch_train_accuracies=results[1],
            avg_epoch_validation_costs=results[2],
            avg_epoch_validation_accuracies=results[3]
        )   

if __name__ == "__main__": main()