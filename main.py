# Import necessary libraries
from quantorch.src.dataset import load_dataset, num_classes
from quantorch.src.noise import NoiseCreator
from quantorch.src.pre_main import ModelsCreator
from quantorch.src.training import Trainer

from torch import manual_seed
from torch.nn import MSELoss, CrossEntropyLoss

import random
import numpy as np
import hydra
from omegaconf import DictConfig
from typing import Union

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(config: DictConfig) -> None:
    # Define configuration
    SEED = config['seed']
    BATCH_SIZE = config['batch_size']
    LEARNING_RATE = config['learning_rate']
    EPOCHS = config['epochs']
    loss_fn = MSELoss() if config['loss_function'] == 'MSE' else CrossEntropyLoss() if config['loss_function'] == 'CrossEntropy' else None
    NUM_QUBITS = config['num_qubits']
    SHOTS = config['shots']
    feature_map_name = config['feature_map_name']
    feature_map_entanglement = config['feature_map_entanglement']
    FEATURE_MAP_DEPTH = config['feature_map_depth']
    ansatz_name = config['ansatz_name']
    ansatz_entanglement = config['ansatz_entanglement']
    ANSATZ_DEPTH = config['ansatz_depth']
    quanvolution_name = config['quanvolution_name']
    QUANVOLUTION_OUT_CHANNELS = config['quanvolution_out_channels']
    local = config['local']
    dataset = config['dataset']
    errors_names = config['errors_names']
    errors_probabilities = config['errors_probabilities']
    NUM_CLASSES = num_classes(dataset=dataset, local=local)
    
    # Set random seed for reproducibility
    manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Load data
    train_loader, validation_loader = load_dataset(
                                            dataset=dataset,
                                            batch_size=BATCH_SIZE,
                                            local=local)
    
    # Define the noise classes, models and backends
    noise = NoiseCreator(errors_names=errors_names,
                         errors_probabilities=errors_probabilities)

    # Create the models
    models = ModelsCreator(
             num_qubits=NUM_QUBITS,
             shots=SHOTS,
             feature_map_name=feature_map_name,
             feature_map_depth=FEATURE_MAP_DEPTH,
             feature_map_entanglement=feature_map_entanglement,
             ansatz_name=ansatz_name,
             ansatz_depth=ANSATZ_DEPTH,
             ansatz_entanglement=ansatz_entanglement,
             quanvolution_name=quanvolution_name,
             quanvolution_out_channels=QUANVOLUTION_OUT_CHANNELS,
             noise=noise,
             local=local,
             dataset=dataset)

    # Perform training and validation of the models to be compared
    results = Trainer(models_complete=models,
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                    loss_fn=loss_fn,
                    epochs=EPOCHS,
                    learning_rate=LEARNING_RATE,
                    local=local)

if __name__ == "__main__": main()