from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from typing import Union
from qiskit_machine_learning.connectors import TorchConnector

class QuanvolutionSampler(nn.Module):
    def __init__(self,
                 quantum_filter: TorchConnector,
                 out_channels: Union[int, None] = None,
                 stride: int = 1,
                 padding: int = 0):
        super(QuanvolutionSampler, self).__init__()
        self.quantum_filter = quantum_filter
        self.num_qubits = self.quantum_filter.neural_network.num_inputs
        self.stride = stride
        self.padding = padding

        self.out_channels : int
        if out_channels == None:
            self.out_channels = int(2 ** self.num_qubits)
        elif out_channels > 2 ** self.num_qubits:
            raise ValueError("The number of output channels must be lower than 2 ^ num_qubits!")
        else:
            self.out_channels = out_channels

    def forward(self, data_loader: Tensor) -> Tensor:
        input_unfolded : Tensor = F.unfold(input=data_loader,
                                  kernel_size=int(self.num_qubits**0.5),
                                  stride=self.stride,
                                  padding=self.padding).transpose(1, 2)
        
        input_unfolded_reshaped : Tensor = input_unfolded.reshape(
                                  input_unfolded.size(0)*input_unfolded.size(1), -1)
        
        output_unfolded : Tensor = self.quantum_filter(input_unfolded_reshaped)[:, : self.out_channels]

        output_unfolded_reshaped : Tensor = output_unfolded.view(
                                   input_unfolded.size(0),
                                   input_unfolded.size(1), -1)

        output_unfolded_reshaped : Tensor = output_unfolded_reshaped.transpose(1, 2)
    
        output_refolded : Tensor = output_unfolded_reshaped.view(
                          input_unfolded.size(0),
                          output_unfolded.size(1),
                          int(output_unfolded_reshaped.size(2)**0.5),
                          int(output_unfolded_reshaped.size(2)**0.5))

        return output_refolded

class QuanvolutionEstimator(nn.Module):
    def __init__(self,
                 quantum_filter: TorchConnector,
                 stride: int = 1,
                 padding: int = 0):
        super(QuanvolutionEstimator, self).__init__()
        self.quantum_filter = quantum_filter
        self.stride = stride
        self.padding = padding

    def forward(self, data_loader: Tensor) -> Tensor:
        num_qubits = self.quantum_filter.neural_network.num_inputs

        input_unfolded = F.unfold(input=data_loader,
                                  kernel_size=int(num_qubits**0.5),
                                  stride=self.stride,
                                  padding=self.padding).transpose(1, 2)
                        
        input_unfolded_reshaped = input_unfolded.reshape(
                                  input_unfolded.size(0)*input_unfolded.size(1), -1)
        
        output_unfolded = self.quantum_filter(input_unfolded_reshaped)

        output_unfolded_reshaped = output_unfolded.view(
                                   input_unfolded.size(0),
                                   input_unfolded.size(1), -1)

        output_unfolded_reshaped = output_unfolded_reshaped.transpose(1, 2)
    
        output_refolded = output_unfolded_reshaped.view(
                          input_unfolded.size(0),
                          output_unfolded.size(1),
                          int(output_unfolded_reshaped.size(2)**0.5),
                          int(output_unfolded_reshaped.size(2)**0.5))

        return output_refolded