import pennylane as qml
from pennylane.devices.device_api import Device
from typing import Union

from lutech_quantum_cnn.dataset import num_classes
from lutech_quantum_cnn.quanvolution import Quanvolution

import torch
from torch import Tensor, manual_seed
from torch.nn import (
    Conv2d,
    ReLU,
    Linear,
    Flatten,
    Sequential,
    Module
)
from torch.utils.data import DataLoader

manual_seed(42)

class ClassicNet(Module):
    def __init__(
        self,
        kernel_size: int,
        convolution_output_channels: int,
        classifier_input_features: int,
        classifier_output_features: int,
        torch_device: torch.device,
    ):
        super(ClassicNet, self).__init__()

        self.kernel_size = kernel_size
        self.convolution_output_channels = convolution_output_channels
        self.classifier_input_features = classifier_input_features
        self.classifier_output_features = classifier_output_features
        self.torch_device = torch_device

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
        )

        self.prob = None

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HybridNet(Module):
    def __init__(
        self,
        device: Device,          
        torch_device: torch.device,
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

        self.device = device                  
        self.torch_device = torch_device  

        self.prob = noise_prob
        self.feature_map = feature_map
        self.feature_map_reps = str(feature_map_reps)
        self.ansatz = 'ra' if ansatz == 'real_amplitudes' else ansatz
        self.ansatz_reps = str(ansatz_reps)

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
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.net(x)
        return output


def flatten_dimension(
    train_loader: DataLoader,
    kernel_size: int,
    convolution_output_channels: int,
) -> int:
    images, _ = next(iter(train_loader))
    in_width: int = images.shape[3]

    k_width: int = int(kernel_size)
    out_width: int = int(in_width - k_width + 1)
    out_pixels: int = int(out_width * out_width)
    flatten_size: int = out_pixels * convolution_output_channels
    return flatten_size


def create_cnn(
    train_loader: DataLoader,
    dataset_folder_path: str,
    kernel_size: int,
    device: Device | None,                
    torch_device: torch.device | None,   
    noise: str | None,
    noise_prob: float | None,
    feature_map: str,
    ansatz: str,
    feature_map_reps: int,
    ansatz_reps: int,
    classes: int,
    show_circuit: bool = False,
) -> Union[HybridNet, ClassicNet]:

    convolution_output_channels: int = int(2 ** (kernel_size * kernel_size))

    classifier_input_features: int = flatten_dimension(
        train_loader=train_loader,
        kernel_size=kernel_size,
        convolution_output_channels=convolution_output_channels,
    )

    classifier_output_features: int = num_classes(
        dataset_folder_path=dataset_folder_path
    )

    if device is None or torch_device is None:
        model = ClassicNet(
            kernel_size=kernel_size,
            convolution_output_channels=convolution_output_channels,
            classifier_input_features=classifier_input_features,
            classifier_output_features=classifier_output_features,
            torch_device=torch.device('cpu')
        )
    else:
        model = HybridNet(
            device=device,
            torch_device=torch_device,
            noise=noise,
            noise_prob=noise_prob,
            feature_map=feature_map,
            ansatz=ansatz,
            feature_map_reps=feature_map_reps,
            ansatz_reps=ansatz_reps,
            qfilter_size=kernel_size,
            classifier_input_features=classifier_input_features,
            classifier_output_features=classes,
            show_circuit=show_circuit
        )

    return model