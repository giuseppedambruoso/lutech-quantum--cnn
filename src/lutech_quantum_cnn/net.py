import pennylane as qml
from pennylane.devices.device_api import Device
from typing import Union

from lutech_quantum_cnn.dataset import num_classes
from lutech_quantum_cnn.quanvolution import Quanvolution

from torch import Tensor, manual_seed
from torch.nn import (
    Conv2d,
    ReLU,
    Linear,
    Flatten,
    Softmax,
    Sequential,
    Module
)
from torch.utils.data import DataLoader

manual_seed(42)

class ClassicNet(Module):
    """Convolutional Neural Network composed of a single convolutional layer,
    followed by a single fully connected layer and a softmax.
    """

    def __init__(
        self,
        kernel_size: int,
        convolution_output_channels: int,
        classifier_input_features: int,
        classifier_output_features: int
    ):
        super(ClassicNet, self).__init__()

        self.kernel_size = kernel_size
        self.convolution_output_channels = convolution_output_channels
        self.classifier_input_features = classifier_input_features
        self.classifier_output_features = classifier_output_features

        self.convolution = Conv2d(
            in_channels=1,
            out_channels=self.convolution_output_channels,
            kernel_size=kernel_size
        )

        self.net = Sequential(
            self.convolution,
            ReLU(),
            Flatten(),
            Linear(
                in_features=classifier_input_features,
                out_features=classifier_output_features
            ),
            Softmax(dim=1)
        )

        self.prob = None
    def forward(self, x: Tensor) -> Tensor:
        # print('Model parameters:', self.convolution.state_dict())
        return self.net(x)


class HybridNet(Module):
    """Convolutional Neural Network composed of a single convolutional layer,
    followed by a single fully connected layer and a softmax.
    """

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
        classifier_input_features: int,
        classifier_output_features: int,
        show_circuit: bool = False,
    ):
        super(HybridNet, self).__init__()

        self.prob = noise_prob
        
        self.quanvolution = Quanvolution(
            device=device,
            noise=noise,
            noise_prob=noise_prob,
            feature_map=feature_map,
            ansatz=ansatz,
            feature_map_reps=feature_map_reps,
            ansatz_reps=ansatz_reps,
            qfilter_size=qfilter_size,
            show_circuit=show_circuit
     )

        self.net = Sequential(
            self.quanvolution,
            Flatten(),
            Linear(
                in_features=classifier_input_features,
                out_features=classifier_output_features,
            ),
            Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        # print('Model parameters:', self.quanvolution.state_dict())
        return self.net(x)


def flatten_dimension(
    train_loader: DataLoader,
    kernel_size: int,
    convolution_output_channels: int,
) -> int:
    """Determine the number of neurons obtained by flattening the output
    images of the convolutional layer.
    """

    # Determine the width of the images
    images, _ = next(iter(train_loader))
    in_width: int = images.shape[3]

    # Determine the width of the kernel
    k_width: int = int(kernel_size)
#    print('Kernel size:', k_width)

    # Determine the width of the output images
    out_width: int = int(in_width - k_width + 1)

    # Determine the number of pixels in each output image
    out_pixels: int = int(out_width * out_width)
#    print('Output image size:', out_pixels)

    # Determine the total number of pixel
    flatten_size: int = out_pixels * convolution_output_channels
#    print('Flatten size:', flatten_size)

    return flatten_size


def create_cnn(
    train_loader: DataLoader,
    dataset_folder_path: str,
    kernel_size: int,
    device: Device | None,
    noise: str | None,
    noise_prob: float | None,
    feature_map: str,
    ansatz: str,
    feature_map_reps: int,
    ansatz_reps: int,
    classes: int,
    show_circuit: bool = False,
) -> Union[HybridNet, ClassicNet]:
    """Create either a classical or a hybrid convolutional neural network
    composed of a single convolutional layer, a single dense layer.
    """

    convolution_output_channels: int = int(2 ** (kernel_size * kernel_size))

    # Determine the number of input features of the classifier
    classifier_input_features: int = flatten_dimension(
        train_loader=train_loader,
        kernel_size=kernel_size,
        convolution_output_channels=convolution_output_channels,
    )

    # Determine the number of classes
    classifier_output_features: int = num_classes(
        dataset_folder_path=dataset_folder_path
    )

    # Create either the classical or the hybrid cnn
    model: Module
    if noise is None or noise_prob is None or device is None:
        model = ClassicNet(
            kernel_size=kernel_size,
            convolution_output_channels=convolution_output_channels,
            classifier_input_features=classifier_input_features,
            classifier_output_features=classifier_output_features,
        )
    else:
        model = HybridNet(
            device = device,
            noise = noise,
            noise_prob = noise_prob,
            feature_map = feature_map,
            ansatz = ansatz,
            feature_map_reps = feature_map_reps,
            ansatz_reps=ansatz_reps,
            qfilter_size=kernel_size,
            classifier_input_features = classifier_input_features,
            classifier_output_features = classes,
            show_circuit = show_circuit
        )

    return model