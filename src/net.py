from dataclasses import dataclass
import torch


@dataclass
class QuantumSim:
    num_qbits: int
    entanglement: str
    shots: int
    depth: int


class Net(torch.nn.Module):
    def __init__(
        self, num_qubits, entanglement, shots, depth, quantum_filter, quantum_conv
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.entanglement = entanglement
        self.shots = shots
        self.depth = depth
        self.q_filter = quantum_filter
        self.quantum_conv = quantum_conv

        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.flatten = torch.nn.Flatten()
        # FIXME: fix this line to work with no global variable
        self.fc = torch.nn.Linear(1, NUM_CLASSES)
        self.out = torch.nn.Softmax(dim=1)

    def forward(self, x):
        print("initiali shape: ", x.shape)
        x = self.quantum_conv(
            data_loader=x,
            num_qubits=self.num_qubits,
            entanglement=self.entanglement,
            shots=self.shots,
            output_channels=1,
            depth=self.depth,
        )
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.out(x)
        return x
