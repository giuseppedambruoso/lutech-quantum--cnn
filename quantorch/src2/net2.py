import numpy as np
from typing import Union

from qiskit_machine_learning.neural_networks import NeuralNetwork
from qiskit_machine_learning.connectors import TorchConnector

from torch import Tensor
from torch.nn import (Conv2d,
                      ReLU,
                      Linear,
                      Flatten,
                      Softmax,
                      Sequential,
                      Module,
                      DataParallel)
from torch.utils.data import DataLoader

from quantorch.src2.dataset2 import num_classes
from quantorch.src2.quanvolution2 import Quanvolution

class ClassicNet(Module):
    def __init__(
              self,
              kernel_size: int,
              convolution_output_channels: int,
              classifier_input_features: int,
              classifier_output_features: int,
              stride: int = 1,
              padding: int = 0,
    ):
        super(ClassicNet, self).__init__()
        
        self.kernel_size = kernel_size
        self.convolution_output_channels = convolution_output_channels
        self.classifier_input_features = classifier_input_features
        self.classifier_output_features = classifier_output_features
        self.stride = stride
        self.padding = padding
        
        convolution = Conv2d(
              in_channels=1,
              out_channels=self.convolution_output_channels,
              kernel_size=kernel_size,
              stride=stride,
              padding=padding
        )

        self.net = Sequential(
            convolution,
            ReLU(),
            Flatten(),
            Linear(
                in_features=classifier_input_features,
                out_features=classifier_output_features
            ),
            Softmax(dim=1)
        )

        
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
    
class HybridNet(Module):
    def __init__(
              self,
              kernel_size: int,
              convolution_output_channels: int,
              classifier_input_features: int,
              classifier_output_features: int,
              quantum_filter: NeuralNetwork,
              stride : int = 1,
              padding : int = 0,
    ):
        super(HybridNet, self).__init__()
        
        self.kernel_size = kernel_size
        self.quantum_filter = TorchConnector(quantum_filter)
        self.convolution_output_channels = convolution_output_channels
        self.classifier_input_features = classifier_input_features
        self.classifier_output_features = classifier_output_features
        self.stride = stride
        self.padding = padding
        
        self.net = Sequential(
            Quanvolution(
                quantum_filter=self.quantum_filter,
                stride=self.stride,
                padding=self.padding,
                out_channels=self.convolution_output_channels,
            ),
            ReLU(),
            Flatten(),
            Linear(
                in_features=classifier_input_features,
                out_features=classifier_output_features
            ),
            Softmax(dim=1)
        )

        
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

def flatten_dimension(
        train_loader: DataLoader,
        kernel_size: int,
        convolution_output_channels: int
    ) -> int:
    
    # Determine the width of the images
    images, labels = next(iter(train_loader))
    in_width : int = images.shape[3]
    
    # Determine the width of the kernel
    k_width : int = int(np.sqrt(kernel_size))

    # Determine the width of the output images
    out_width : int = (in_width - k_width + 1)

    # Determine the number of pixels in each output image
    out_pixels : int = out_width ** 2

    # Determine the total number of pixel
    flatten_size : int = out_pixels * convolution_output_channels

    return flatten_size

def create_cnn(
    hybrid : bool,
    train_loader : DataLoader,
    dataset_folder_name : str,
    kernel_size : int,
    convolution_output_channels : int,
    quantum_filter : NeuralNetwork,
) -> DataParallel[Union[HybridNet,ClassicNet]]:
    
    # Determine the number of input features of the classifier
    classifier_input_features : int = flatten_dimension(
        train_loader=train_loader,
        kernel_size=kernel_size,
        convolution_output_channels=convolution_output_channels
    )

    # Determine the number of classes
    classifier_output_features : int = num_classes(dataset=dataset_folder_name)

    # Create either the classical or the hybrid cnn
    model : Module
    if hybrid == True:
        model = HybridNet(
            kernel_size=kernel_size,
            convolution_output_channels=convolution_output_channels,
            classifier_input_features=classifier_input_features,
            classifier_output_features=classifier_output_features,
            quantum_filter=quantum_filter
        )
    else :
        model = ClassicNet(
            kernel_size=int(np.sqrt(kernel_size)),
            convolution_output_channels=convolution_output_channels,
            classifier_input_features=classifier_input_features,
            classifier_output_features=classifier_output_features
        )
    
    model = DataParallel(model)
    return model