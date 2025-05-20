import numpy as np
from lutech_quantum_cnn.operations import RealAmplitudes, AngleEmbedding

import pennylane as qml
from pennylane.measurements import ProbabilityMP
from pennylane.qnn import TorchLayer
from pennylane.typing import TensorLike
from pennylane.ops.channel import DepolarizingChannel
from pennylane.ops.qubit.parametric_ops_multi_qubit import IsingZZ
from pennylane.devices.device_api import Device

from torch import Tensor, manual_seed
import torch.nn as nn
from torch.nn import functional as F
import torch

manual_seed(42)

def z_feature_map(
        input_features: Tensor,
        reps: int,
        noise: str | None = None,
        noise_prob: float | None = None,
    ) -> None:
    """Z feature map for the VQC."""
    if len(input_features) < 1:
        raise ValueError("Number of features must be at least 1.")
    if reps < 1:
        raise ValueError("Feature map repetitions must be at least 1.")

    for r in range(reps):
        for i in range(len(input_features)):
            qml.Hadamard(wires=i)
        AngleEmbedding(
            features=[2*feature for feature in input_features],
            wires=range(len(input_features)),
            rotation='Y',
            noise=noise,
            noise_prob=noise_prob
        )

def zz_feature_map(
        input_features: Tensor,
        reps: int,
        noise: str | None = None,
        noise_prob : float | None = None
    ) -> None:
    """ZZ feature map for the VQC."""

    if len(input_features) < 1:
        raise ValueError("Number of features must be at least 1.")
    if reps < 1:
        raise ValueError("Feature map repetitions must be at least 1.")

    for r in range(reps):
        for i in range(len(input_features)):
            qml.Hadamard(wires=i)
        AngleEmbedding(
            features=[2 * feature for feature in input_features],
            wires=range(len(input_features)),
            rotation='Y',
            noise=noise,
            noise_prob=noise_prob
        )
        for i in range(len(input_features) - 1):
            phi_val : TensorLike = 2 * (np.pi - input_features[i]) * (np.pi - input_features[i+1]) # type: ignore
            IsingZZ(wires=[i + 1,i], phi=phi_val)
        if noise == 'depolarizing':    
            for i in range(len(input_features)):
                DepolarizingChannel(p=noise_prob, wires=i)

def real_amplitudes_ansatz(
        num_qubits: int,
        reps: int,
        params: Tensor,
        noise: str | None,
        noise_prob: float | None
    ) -> None:
    """Ansatz for the VQC."""

    if reps < 1:
        raise ValueError("Feature map repetitions must be at least 1.")
    for r in range(reps):
        RealAmplitudes(
            weights=params,
            wires=range(num_qubits),
            noise=noise,
            noise_prob=noise_prob
        )

class Quanvolution(nn.Module):
    """Quanvolutional layer for quantum convolutional neural networks."""

    def __init__(
        self,
        device: Device,
        noise: str | None,
        noise_prob: float | None,
        feature_map: str,
        ansatz: str,
        feature_map_reps: int,
        ansatz_reps: int,
        qfilter_size: int,
        show_circuit: bool=False
    ) -> None:
        
        super(Quanvolution, self).__init__()
        self.device = device
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.feature_map_reps = feature_map_reps
        self.ansatz_reps = ansatz_reps
        self.show_circuit = show_circuit
        self.num_qubits : int = int(qfilter_size * qfilter_size)
        self.output_channels = int(2 ** self.num_qubits)
        self.qfilter_size = qfilter_size

        # Define the quantum filter
        @qml.qnode(
            device=device,
            interface='torch',
            diff_method='parameter-shift'
        )
        def qnode(
            inputs: Tensor,
            params: Tensor
        ) -> ProbabilityMP:
            """Quantum circuit for the VQC."""
            if feature_map not in ['z', 'zz']:
                raise ValueError("Feature map must be 'z' or 'zz'.")
            if ansatz not in ['real_amplitudes']:
                raise ValueError("Ansatz must be 'real_amplitudes'.")

            num_qubits: int = int(qfilter_size * qfilter_size)

            if feature_map == 'z':
                z_feature_map(input_features=inputs, reps=feature_map_reps)
            elif feature_map == 'zz':
                zz_feature_map(input_features=inputs, reps=feature_map_reps)

            if ansatz == 'real_amplitudes':
                real_amplitudes_ansatz(
                    num_qubits=num_qubits,
                    reps=ansatz_reps,
                    params=params,
                    noise=noise,
                    noise_prob=noise_prob
                )
            # if self.show_circuit:
            #     print(qml.draw(qnode)(inputs, params))
            
            return qml.probs(wires=range(num_qubits))

        # Calculate the shape of the parameters
        weight_shape = {"params": ((ansatz_reps + 1),(qfilter_size ** 2))}

        self.qfilter = TorchLayer(qnode=qnode, weight_shapes=weight_shape) # type: ignore

    def forward(self, data_loader: Tensor) -> Tensor:
        # Unfold the input tensor to prepare it for quantum processing
        # print('input data shape: ', data_loader.shape)
        input_unfolded: Tensor = F.unfold(
            input=data_loader,
            kernel_size=int(self.qfilter_size),
        ).transpose(1, 2)
        # print('input unfolded shape: ', input_unfolded.shape)

        # Reshape the unfolded input to extract sliding blocks
        input_unfolded_reshaped: Tensor = input_unfolded.reshape(
            input_unfolded.size(0) * input_unfolded.size(1), -1
        )
        # print('input unfolded reshaped shape: ', input_unfolded_reshaped.shape)

        # Apply the quantum filter every sliding block
        output_unfolded : Tensor = torch.zeros(size=(input_unfolded_reshaped.size(0), self.output_channels))
        for i in range(input_unfolded_reshaped.size(0)):
            sliding_block : Tensor = input_unfolded_reshaped[i].squeeze()
            output_unfolded[i] = self.qfilter(sliding_block)
        # print('output unfolded shape: ', output_unfolded.shape)

        # Reshape the output to match the original unfolded input shape
        output_unfolded_reshaped: Tensor = output_unfolded.view(
            input_unfolded.size(0), input_unfolded.size(1), -1
        )
        # print('output unfolded reshaped shape: ', output_unfolded_reshaped.shape)

        # Transpose the reshaped output to prepare for refolding
        output_unfolded_reshaped: Tensor = output_unfolded_reshaped.transpose(1, 2)
        # print('output unfolded reshaped transposed shape: ', output_unfolded_reshaped.shape)

        # Refold the output tensor to its original spatial dimensions
        output_refolded: Tensor = output_unfolded_reshaped.view(
            input_unfolded.size(0),
            output_unfolded.size(1),
            int(output_unfolded_reshaped.size(2) ** 0.5),
            int(output_unfolded_reshaped.size(2) ** 0.5),
        )
        # print('output refolded shape: ', output_refolded.shape)

        return output_refolded