"""

Quantum Machine Learning Library
===============================

Author: `Giuseppe D'Ambruoso <

This library is a collection of quantum machine learning algorithms
and tools for quantum computing. 

# TODOs

High Priority:

- [ ] Make automatic consistent output saving in different folders
- [ ] Make dataclass for simulation parameters (QuantumSim)

Low Priority:

- [ ] Change print in logging
- [ ] Add type hints
- [ ] Move to hydra for configuration management 


"""

import torch
import yaml

import src.convolution as conv

from src.net import QuantumSim, Net
from src.filter import QuantumFilter


def main():

    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    q_filter = QuantumFilter(
        input_dim=config["num_qbits"],
        entanglement=config["entanglement"],
        shots=config["shots"],
        depth=config["depth"],
    )

    model = Net(
        num_qubits=config["num_qbits"],
        entanglement=config["entanglement"],
        shots=config["shots"],
        depth=config["depth"],
        quantum_filter=q_filter,
        quantum_conv=conv.quantum_conv_noloop,
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])


if __name__ == "__main__":
    main()
