import numpy as np
from lutech_quantum_cnn.operations import RealAmplitudes, AngleEmbedding

import pennylane as qml
from pennylane.measurements import ProbabilityMP
from pennylane.qnn import TorchLayer
from pennylane.typing import TensorLike
from pennylane.ops.channel import DepolarizingChannel
from pennylane.ops.qubit.parametric_ops_multi_qubit import IsingZZ
from pennylane.devices.device_api import Device

import torch.nn.functional as F
import torch
from torch import Tensor, manual_seed, vmap
import torch.nn as nn


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
        device,
        noise,
        noise_prob,
        feature_map,
        ansatz,
        feature_map_reps,
        ansatz_reps,
        qfilter_size,
        show_circuit=False
    ) -> None:
        
        super(Quanvolution, self).__init__()
        self.device = device
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.feature_map_reps = feature_map_reps
        self.ansatz_reps = ansatz_reps
        self.show_circuit = show_circuit
        self.num_qubits: int = int(qfilter_size * qfilter_size)
        self.output_channels = int(2 ** self.num_qubits)
        self.qfilter_size = qfilter_size

        @qml.qnode(
            device=device,
            interface='torch',
            diff_method='parameter-shift',
        )
        def qnode(inputs: torch.Tensor, params: torch.Tensor):
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
            
            return qml.probs(wires=range(num_qubits))

        weight_shape = {"params": ((ansatz_reps + 1), (qfilter_size ** 2))}
        self.qfilter = TorchLayer(qnode=qnode, weight_shapes=weight_shape) # type: ignore

    def forward(self, data_loader: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        
        print('input shape:', data_loader.shape)

        # Unfold input into sliding blocks
        input_unfolded: torch.Tensor = F.unfold(
            input=data_loader,
            kernel_size=int(self.qfilter_size),
        ).transpose(1, 2)  # shape: (batch_size, num_blocks, block_size)
        print('unfolded input shape:', input_unfolded.shape)

        # Reshape to (total_blocks, block_size)
        input_unfolded_reshaped: torch.Tensor = input_unfolded.reshape(
            -1, self.num_qubits
        ).to(device)
        print('reshaped unfolded input shape:', input_unfolded_reshaped.shape)

        # Vectorize qfilter with vmap
        batched_qfilter = vmap(self.qfilter)

        # Apply qfilter to all sliding blocks in parallel
        output_unfolded: torch.Tensor = batched_qfilter(input_unfolded_reshaped)
        print('output unfolded after qfilter shape:', output_unfolded.shape)

        # Reshape output to (batch_size, num_blocks, output_channels)
        batch_size = data_loader.size(0)
        num_blocks = input_unfolded.size(1)
        output_unfolded_reshaped: torch.Tensor = output_unfolded.view(
            batch_size, num_blocks, -1
        )
        print('reshaped output unfolded shape:', output_unfolded_reshaped.shape)

        # Transpose to (batch_size, output_channels, num_blocks)
        output_unfolded_reshaped = output_unfolded_reshaped.transpose(1, 2)
        print('transposed output unfolded shape:', output_unfolded_reshaped.shape)

        # Refold to (batch_size, output_channels, H, W)
        side_len = int(output_unfolded_reshaped.size(2) ** 0.5)
        output_refolded: torch.Tensor = output_unfolded_reshaped.view(
            batch_size,
            output_unfolded_reshaped.size(1),
            side_len,
            side_len,
        )
        print('refolded output shape:', output_refolded.shape)

        return output_refolded