# Define the quantum convolution with 5 output channels and N output channels
import torch

from numpy import sqrt


# FIXME: fix the function to inject data and quantum filter
def quantum_conv(data_loader, num_qubits, output_channels, q_filter):
    """
    Applies quantum convolution to each image of a specified data loader.
    It operates by applying a quantum filter to every 2x2 sliding block
    extracted from the image.

    Args:
        data_loader: The data loader containing the images.
        num_qubits: Number of qubits in the quantum filter.
        entanglement: The entanglement structure of the quantum filter.
        backend: The backend associated with the quantum filter.
        shots: The number of shots associated with the quantum filter.
        num_output_channels: Number of output channels.

    Returns:
        The set of the convoluted images.
    """

    print("input image: ", data_loader[0][0])

    # Unfold each image of the data loader into 2x2 blocks
    input_unfolded = torch.nn.functional.unfold(
        data_loader, kernel_size=int(sqrt(num_qubits)), stride=1, padding=0
    ).unsqueeze(1)

    print("input unfolded: ", input_unfolded)
    print("input_unfolded.shape: ", input_unfolded.shape)

    # Apply the quantum filter to each 2x2 block
    output_unfolded = [[] for _ in range(input_unfolded.size(0))]

    for i in range(input_unfolded.size(0)):
        for j in range(input_unfolded.size(2)):
            block = input_unfolded[i, 0, j]
            output_channel = []
            for _ in range(output_channels):
                output_channel.append(q_filter.qnn(block))
            output_unfolded[i].append(output_channel)

    output_unfolded = torch.tensor(output_unfolded).transpose(2, 1).unsqueeze(3)
    print("output_unfolded: ", output_unfolded)
    print("output_unfolded.shape: ", output_unfolded.shape)

    # Refold the output images
    output_refolded = output_unfolded.view(
        output_unfolded.size(0),
        output_unfolded.size(1),
        int(sqrt(output_unfolded.size(2))),
        int(sqrt(output_unfolded.size(2))),
    )
    print("output_refolded: ", output_refolded)
    print("output_refolded.shape", output_refolded.shape)

    return output_refolded


# FIXME: make input and output of the two functions consistent
def quantum_conv_noloop(data_loader, num_qubits, q_filter):
    """
    Applies quantum convolution to each image of a specified data loader.
    It operates by applying a quantum filter to every 2x2 sliding block
    extracted from the image.

    Args:
        data_loader: The data loader containing the images.
        pqc: The quantum circuit associated with the quantum filter.
        backend: The backend associated with the quantum filter.
        shots: The number of shots associated with the quantum filter.

    Returns:
        The set of the convoluted images.
    """

    # Unfold each image of the data loader into 2x2 blocks
    input_unfolded = torch.nn.functional.unfold(
        data_loader, kernel_size=int(sqrt(num_qubits)), stride=1, padding=0
    ).unsqueeze(1)
    print(input_unfolded, input_unfolded.shape)

    output_unfolded = q_filter.qnn(input_unfolded)  # .squeeze()
    print("output unfolded shape no loop: ", output_unfolded.shape)

    # Refold the output images
    output_refolded = output_unfolded.view(
        output_unfolded.size(0),
        output_unfolded.size(1),
        int(sqrt(output_unfolded.size(2))),
        int(sqrt(output_unfolded.size(2))),
    )
    print("output_refolded: ", output_refolded)
    print("output_refolded.shape", output_refolded.shape)
    return output_refolded
