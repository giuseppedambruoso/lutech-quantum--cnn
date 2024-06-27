"""unit test for convolution.py"""

import torch
import numpy as np
from filter import QuantumFilter
from src.convolution import quantum_conv, quantum_conv_noloop


def test_quantum_convolution():
    data_loader = torch.tensor([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]])
    num_qubits = 2
    entanglement = "linear"
    shots = 100
    output_channels = 1
    depth = 1
    output_loop = quantum_conv(data_loader, q_filter)
    output_noloop = quantum_conv_noloop(data_loader, q_filter)

    np.testing.assert_array_almost_equal(output_loop, output_noloop, decimal=1)
