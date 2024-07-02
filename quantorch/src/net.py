from typing import Union

from torch.nn import (Conv2d,
                      BatchNorm1d,
                      BatchNorm2d,
                      ReLU,
                      MaxPool2d,
                      Linear,
                      Dropout,
                      Flatten,
                      Softmax,
                      Sequential,
                      Module)
from torch import Tensor

from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import NeuralNetwork

from quantorch.src.quanvolution import (QuanvolutionEstimator,
                                        QuanvolutionSampler)

from quantorch.src.dataset import num_classes

def ConvLayer(in_channels, out_channels, kernel_size):
    """
    Helper function to create a convolutional block.

    Parameters:
        - in_channels: Number of input channels.
        - out_channels: Number of output channels.
        - kernel_size: Size of the convolutional kernel.
        - padding: Padding type for the convolution.

    Returns:
        - model_cb: Convolutional block as a Sequential module.
    """
    model_cb = Sequential(
        Conv2d(in_channels, out_channels, kernel_size),
        BatchNorm2d(out_channels),
        ReLU(),
        MaxPool2d(kernel_size=2)
    )
    return model_cb

def DenseLayer(in_units, out_units):
    """
    Helper function to create a dense block.

    Parameters:
        - in_units: Number of input units.
        - out_units: Number of output units.

    Returns:
        - model_db: Dense block as a Sequential module.
    """
    model_db = Sequential(
        Linear(in_units, out_units),
        ReLU(),
        BatchNorm1d(out_units),
        Dropout(p=0.2)
    )
    return model_db
 
class ClassicNet(Module):
    def __init__(self,
                 quanvolution_name: str,
                 kernel_size: int,
                 dataset: str,
                 local: bool,
                 quanvolution_out_channels: Union[int, None] = None):
        super(ClassicNet, self).__init__()
        
        self.quanvolution_name : str = quanvolution_name
        self.dataset : str = dataset
        self.local : bool = local
        self.kernel_size : int = kernel_size
        
        self.quanvolution_out_channels : int
        if quanvolution_out_channels == None:
            self.quanvolution_out_channels = 2 ** (kernel_size ** 2) 
        elif quanvolution_out_channels > 2 ** (kernel_size ** 2):
            raise ValueError("The number of the quanvolution output channels must be lower than 2^num_qubits!")
        else :
            self.quanvolution_out_channels = quanvolution_out_channels

        n_classes : int = num_classes(dataset=self.dataset, local=self.local)

        if self.quanvolution_name == 'No':
            self.net = Sequential(
                    ConvLayer(in_channels=1, out_channels=256, kernel_size=5),
                    ConvLayer(in_channels=256, out_channels=128, kernel_size=5),
                    ConvLayer(in_channels=128, out_channels=64, kernel_size=5),
                    Flatten(),
                    DenseLayer(in_units=25600, out_units=128),
                    DenseLayer(in_units=128, out_units=64),
                    DenseLayer(in_units=64, out_units=n_classes),
                    Softmax(dim=1))
        
        elif self.quanvolution_name == 'quanvolution_estimator':
            self.net = Sequential(
                    Conv2d(in_channels=1, out_channels=4, kernel_size=2),
                    ReLU(),
                    Flatten(),
                    Linear(in_features=16 if self.dataset=='Tetris' else 729 if self.dataset=='breastMNIST' else 1, out_features=n_classes),
                    Softmax(dim=1))
        
        elif self.quanvolution_name == 'quanvolution_sampler':
            convolution = Conv2d(in_channels=1,
                                 out_channels=self.quanvolution_out_channels,
                                 kernel_size=kernel_size)

            conv_out_size : int
            if self.dataset == 'Tetris':
                conv_out_size = int((3 - (kernel_size - 1)) ** 2)
            elif self.dataset == 'breastMNIST':
                conv_out_size = int((28 - (kernel_size - 1)) ** 2)
            flatten_out_size : int = conv_out_size * self.quanvolution_out_channels

            self.net = Sequential(
                convolution,
                ReLU(),
                Flatten(),
                Linear(in_features=flatten_out_size, out_features=n_classes),
                Softmax(dim=1))
        
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
    
class HybridNet(Module):
    def __init__(self,
                 quantum_filter: 'NeuralNetwork',
                 quanvolution_name: str,
                 local: bool,
                 dataset: str,
                 quanvolution_out_channels: Union[int, None] = None):
        super(HybridNet, self).__init__()
             
        # initialize attributes
        self.quantum_filter = TorchConnector(quantum_filter)
        self.quanvolution_name = quanvolution_name
        self.dataset = dataset
        self.local = local
        self.quanvolution_out_channels = quanvolution_out_channels
        
        self.num_qubits = self.quantum_filter.neural_network.num_inputs
        n_classes = num_classes(dataset=self.dataset, local=self.local)
        
        if self.quanvolution_name == 'No':
            self.net = Sequential(
                ConvLayer(in_channels=1, out_channels=256, kernel_size=5),
                ConvLayer(in_channels=256, out_channels=128, kernel_size=5),
                ConvLayer(in_channels=128, out_channels=64, kernel_size=5),
                Flatten(),
                DenseLayer(in_units=25600, out_units=128),
                DenseLayer(in_units=128, out_units=64),
                DenseLayer(in_units=64, out_units=n_classes),
                TorchConnector(filter),
                Softmax(dim=1))

        elif self.quanvolution_name == 'quanvolution_estimator':
            self.net = Sequential(
                QuanvolutionEstimator(self.quantum_filter, self.quanvolution_out_channels),
                Flatten(),
                Linear(in_features=16 if self.dataset=='Tetris' else 729 if self.dataset=='breastMNIST' else 1,
                        out_features=n_classes),
                Softmax(dim=1))
        
        elif self.quanvolution_name == 'quanvolution_sampler':
            quanvolution = QuanvolutionSampler(self.quantum_filter, self.quanvolution_out_channels)
            kernel_size = int(quanvolution.num_qubits ** 0.5)

            conv_out_size : int
            if self.dataset == 'Tetris':
                conv_out_size = int((3 - (kernel_size - 1)) ** 2)
            elif self.dataset == 'breastMNIST':
                conv_out_size = int((28 - (kernel_size - 1)) ** 2)
            flatten_out_size : int = conv_out_size * quanvolution.out_channels

            self.net = Sequential(
                quanvolution,
                Flatten(),
                Linear(in_features=flatten_out_size, out_features=n_classes),
                Softmax(dim=1))
            
            self.quanv = quanvolution
            self.flatten = Flatten()
            self.fc = Linear(in_features=flatten_out_size, out_features=n_classes)
            self.out = Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)