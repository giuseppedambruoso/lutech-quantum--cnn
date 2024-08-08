from re import X
import numpy as np
from typing import Union

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

from torch import Tensor
from torch.nn import (
    Conv2d,
    ReLU,
    Linear,
    Flatten,
    Softmax,
    Sequential,
    Module,
    DataParallel,
)
from torch.utils.data import DataLoader

from src.dataset import num_classes
from src.quanvolution import Quanvolution


class ClassicNet(Module):
    """Convolutional Neural Network composed of a single convolutional layer,
    followed by a single fully connected layer and a softmax.

    Attributes
    ----------
    kernel_size : int
        The width of the filter kernel used in the convolutional layer.
    stride : int
        The stride used in the convolutional layer.
    padding : int
        The padding used in the convolutional layer.
    convolution_output_channels : int
        The number of output channels from the convolutional layer.
    classifier_input_features : int
        The number of input features of the fully connected layer.
    classifier_output_features : int
        The number of output feature of the fully connected layer. It
        corresponds to the number of classes.
    net : Sequential
        The sequence of operations composing the CNN.

    Methods
    -------
    forward
        Execute the CNN on the specified data.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int,
        padding: int,
        convolution_output_channels: int,
        classifier_input_features: int,
        classifier_output_features: int,
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
            padding=padding,
        )

        self.net = Sequential(
            convolution,
            ReLU(),
            Flatten(),
            Linear(
                in_features=classifier_input_features,
                out_features=classifier_output_features,
            ),
            Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HybridNet(Module):
    """Convolutional Neural Network composed of a single convolutional layer,
    followed by a single fully connected layer and a softmax.

    Attributes
    ----------
    kernel_size : int
        The width of the filter kernel used in the quantum convolutional layer.
    quantum_filter : SamplerQNN
        The quantum filter used in the quantum convolutional layer.
    stride : int
        The stride used in the quantum convolutional layer.
    padding : int
        The padding used in the quantum convolutional layer.
    convolution_output_channels : int
        The number of output channels from the quantum convolutional layer. It
        must be less then or equal to 2**kernel_size.
    classifier_input_features : int
        The number of input features of the fully connected layer.
    classifier_output_features : int
        The number of output feature of the fully connected layer. It
        corresponds to the number of classes.
    net : Sequential
        The sequence of operations composing the CNN.

    Methods
    -------
    forward
        Execute the CNN on the specified data.
    """

    def __init__(
        self,
        quantum_filter: SamplerQNN,
        kernel_size: int,
        stride: int,
        padding: int,
        convolution_output_channels: int,
        classifier_input_features: int,
        classifier_output_features: int,
    ):
        super(HybridNet, self).__init__()

        self.kernel_size = kernel_size
        self.quantum_filter = TorchConnector(quantum_filter)
        self.convolution_output_channels = convolution_output_channels
        self.classifier_input_features = classifier_input_features
        self.classifier_output_features = classifier_output_features
        self.stride = stride
        self.padding = padding

        self.quanvolution = Quanvolution(
            quantum_filter=self.quantum_filter,
            stride=self.stride,
            padding=self.padding,
            out_channels=self.convolution_output_channels,
        )

        self.net = Sequential(
            self.quanvolution,
            Flatten(),
            Linear(
                in_features=classifier_input_features,
                out_features=classifier_output_features,
            ),
            Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def flatten_dimension(
    train_loader: DataLoader,
    kernel_size: int,
    stride: int,
    padding: int,
    convolution_output_channels: int,
) -> int:
    """Determine the number of neurons obtained by flattening the output
    images of the convolutional layer.

    Parameters
    ----------
    train_loader : DataLoader
        The data loader of the training set. It is used to determine the size
        of the input images.
    kernel_size : int
        The width of the filter kernel used in the (classical or quantum)
        convolutional layer.
    stride : int
        The stride used in the (classical or quantum) convolutional layer.
    padding : int
        The padding used in the (classical or quantum) convolutional layer.
    convolution_output_channels : int
        The number of output channels of the (classical or quantum)
        convolutional layer.

    Returns
    int
        The number of neurons obtained by flattening the output images of the
        convolutional layer.
    """

    # Determine the width of the images
    images, _ = next(iter(train_loader))
    in_width: int = images.shape[3]

    # Determine the width of the kernel
    k_width: int = int(np.sqrt(kernel_size))

    # Determine the width of the output images
    out_width: int = int((in_width + 2 * padding - k_width) // stride + 1)

    # Determine the number of pixels in each output image
    out_pixels: int = out_width**2

    # Determine the total number of pixel
    flatten_size: int = out_pixels * convolution_output_channels

    return flatten_size


def create_cnn(
    hybrid: bool,
    train_loader: DataLoader,
    dataset_folder_path: str,
    kernel_size: int,
    convolution_output_channels: int,
    quantum_filter: SamplerQNN,
    stride: int = 1,
    padding: int = 0,
) -> DataParallel[Union[HybridNet, ClassicNet]]:
    """Create either a classical or a hybrid convolutional neural network
    composed of a single convolutional layer, a single dense layer and a
    softmax.

    Parameters
    ----------
    hybrid : bool
        True if you want to create a hybrid CNN, False otherwise.
    train_loader : DataLoader
        The data loader of the training set. It is used to determine the number
        of neurons obtained after flattening.
    dataset_folder_path : str
        The path of the folder in which the whole dataset is contained. It is
        used for determining the number of classes, i.e. the number of output
        neurons of the dense layer.
    kernel_size : int
        The width of the kernel used in the (classical or quantum)
        convolutional layer.
    convolution_output_channels : int
        The number of output channels of the (classical or quantum)
        convolutional layer.
    quantum_filter : SamplerQNN
        The quantum filter used in the quantum convolutional layer.
    stride : int
        The stride used in the (classical or quantum) convolutional layer.
    padding : int
        The padding used in the (classical or quantum) convolutional layer.

    Returns
    DataParallel[Union[HybridNet,ClassicNet]]
        Either the classical or the hybrid CNN.

    """

    # Determine the number of input features of the classifier
    classifier_input_features: int = flatten_dimension(
        train_loader=train_loader,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        convolution_output_channels=convolution_output_channels,
    )

    # Determine the number of classes
    classifier_output_features: int = num_classes(
        dataset_folder_path=dataset_folder_path
    )

    # Create either the classical or the hybrid cnn
    model: Module
    if hybrid == True:
        model = HybridNet(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            convolution_output_channels=convolution_output_channels,
            classifier_input_features=classifier_input_features,
            classifier_output_features=classifier_output_features,
            quantum_filter=quantum_filter,
        )
    else:
        model = ClassicNet(
            kernel_size=int(np.sqrt(kernel_size)),
            stride=stride,
            padding=padding,
            convolution_output_channels=convolution_output_channels,
            classifier_input_features=classifier_input_features,
            classifier_output_features=classifier_output_features,
        )

    model = DataParallel(model)
    return model