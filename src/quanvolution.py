from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from qiskit_machine_learning.connectors import TorchConnector


class Quanvolution(nn.Module):
    """Implementation of a convolutional layer with a quantum filter.

    Attributes
    ----------
    quantum_filter : TorchConnector
        The quantum filter used in the quantum convolution.
    num_qubits : int
        The number of qubits the quantum filter's VQC is composed of. It is
        determined from the quantum filter.
    stride : int
        The stride used in the quantum convolution.
    padding : int
        The stride used in the quantum convolution.
    out_channels : Union[int, None]
        The number of output channels. Defaults to 2**num_qubits. Raises an
        error if it is larger than 2**num_qubits.

    Methods
    -------
    forward
        Apply the quantum convolution to a given dataset.
    """

    def __init__(
        self,
        quantum_filter: TorchConnector,
        stride: int,
        padding: int,
        out_channels: int,
    ):

        super(Quanvolution, self).__init__()
        self.quantum_filter = quantum_filter
        self.num_qubits = self.quantum_filter.neural_network.num_inputs
        self.stride = stride
        self.padding = padding

        # Set the number of output channels, ensuring it does not
        # exceed 2^num_qubits
        if out_channels > 2**self.num_qubits:
            raise ValueError(
                "The number of output channels must be lower than 2 ^ num_qubits!"
            )
        else:
            self.out_channels = out_channels

    def forward(self, data_loader: Tensor) -> Tensor:
        # Unfold the input tensor to prepare it for quantum processing
        input_unfolded: Tensor = F.unfold(
            input=data_loader,
            kernel_size=int(self.num_qubits**0.5),
            stride=self.stride,
            padding=self.padding,
        ).transpose(1, 2)

        # Reshape the unfolded input to extract sliding blocks
        input_unfolded_reshaped: Tensor = input_unfolded.reshape(
            input_unfolded.size(0) * input_unfolded.size(1), -1
        )

        # Apply the quantum filter every sliding block
        output_unfolded: Tensor = self.quantum_filter(input_unfolded_reshaped)[
            :, : self.out_channels
        ]

        # Reshape the output to match the original unfolded input shape
        output_unfolded_reshaped: Tensor = output_unfolded.view(
            input_unfolded.size(0), input_unfolded.size(1), -1
        )

        # Transpose the reshaped output to prepare for refolding
        output_unfolded_reshaped: Tensor = output_unfolded_reshaped.transpose(1, 2)

        # Refold the output tensor to its original spatial dimensions
        output_refolded: Tensor = output_unfolded_reshaped.view(
            input_unfolded.size(0),
            output_unfolded.size(1),
            int(output_unfolded_reshaped.size(2) ** 0.5),
            int(output_unfolded_reshaped.size(2) ** 0.5),
        )

        return output_refolded