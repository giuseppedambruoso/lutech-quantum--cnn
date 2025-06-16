from torch import float64, Tensor
import torch
import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import RX, RY, RZ
from pennylane.ops.channel import DepolarizingChannel
from pennylane.wires import WiresLike
from pennylane.typing import TensorLike

ROT = {"X": RX, "Y": RY, "Z": RZ}

def to_numpy_safe(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

class AngleEmbedding(Operation):
    num_wires = AnyWires
    grad_method = None  # type: ignore

    def _flatten(self):
        hyperparameters = (("rotation", self._rotation),)
        return self.data, (self.wires, hyperparameters)

    def __repr__(self):
        return f"AngleEmbedding({self.data[0]}, wires={self.wires.tolist()}, rotation={self._rotation})"

    def __init__(self, features, wires, noise: str | None = None, noise_prob: float | None = None, rotation="X", id=None):
        if rotation not in ROT:
            raise ValueError(f"Rotation option {rotation} not recognized.")

        features = to_numpy_safe(features)
        features = torch.tensor(features, dtype=torch.float64)
        shape = qml.math.shape(features)[-1:]
        n_features = shape[0]

        if n_features > len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)} or less; got length {n_features}."
            )

        self._rotation = rotation
        self._hyperparameters = {
            "rotation": ROT[rotation],
            "noise": noise,
            "noise_prob": noise_prob
        }

        wires = wires[:n_features]
        super().__init__(features, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (1,)

    @staticmethod
    def compute_decomposition(features, wires, rotation, noise, noise_prob):
        features = to_numpy_safe(features)
        batched = qml.math.ndim(features) > 1
        features = qml.math.T(features) if batched else features
        decomposition = []

        for i in range(len(wires)):
            decomposition.append(rotation(features[i], wires=wires[i]))
            if noise == 'depolarizing' and noise_prob is not None and noise_prob > 0:
                decomposition.append(DepolarizingChannel(p=noise_prob, wires=wires[i]))

        return decomposition


class RealAmplitudes(Operation):
    num_wires = AnyWires
    grad_method = None  # type: ignore

    def __init__(
        self,
        weights,
        wires,
        noise: str | None = None,
        noise_prob: float | None = None,
        ranges=None,
        imprimitive=None,
        id=None
    ):
        shape = qml.math.shape(weights)[-2:]
        self.noise = noise
        self.noise_prob = noise_prob

        if shape[1] != len(wires):
            raise ValueError(
                f"Weights tensor must have second dimension of length {len(wires)}; got {shape[1]}"
            )

        if len(shape) != 2:
            raise ValueError(
                f"Weights tensor must have shape (n_layers, n_wires); got {shape}"
            )

        if ranges is None:
            if len(wires) > 1:
                ranges = tuple((l % (len(wires) - 1)) + 1 for l in range(shape[0]))
            else:
                ranges = (0,) * shape[0]
        else:
            ranges = tuple(ranges)
            if len(ranges) != shape[0]:
                raise ValueError(f"Range sequence must be of length {shape[0]}; got {len(ranges)}")
            for r in ranges:
                if r % len(wires) == 0:
                    raise ValueError(
                        f"Ranges must not be zero nor divisible by the number of wires; got {r}"
                    )

        self._hyperparameters = {
            "ranges": ranges,
            "imprimitive": imprimitive or qml.CNOT,
            "noise": noise,
            "noise_prob": noise_prob
        }

        super().__init__(weights, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(weights: TensorLike, wires, ranges, imprimitive, noise, noise_prob):
        n_layers = qml.math.shape(weights)[-2]
        wires = qml.wires.Wires(wires)
        op_list = []

        for l in range(n_layers):
            for i in range(len(wires)):
                op_list.append(
                    qml.RY(
                        weights[..., l, i],
                        wires=wires[i],
                    )
                )
                if noise == "depolarizing" and noise_prob is not None and noise_prob > 0:
                    op_list.append(DepolarizingChannel(p=noise_prob, wires=wires[i]))

            if len(wires) > 1:
                for i in range(len(wires)):
                    act_on = wires.subset([i, i + ranges[l]], periodic_boundary=True)
                    op_list.append(imprimitive(wires=act_on))
                    if noise == "depolarizing" and noise_prob is not None and noise_prob > 0:
                        op_list.append(DepolarizingChannel(p=noise_prob, wires=i))

        return op_list

    @staticmethod
    def shape(n_layers, n_wires):
        return n_layers, n_wires

    @staticmethod
    def compute_qfunc_decomposition(weights, *wires, ranges, imprimitive, noise, noise_prob):
        wires = qml.math.array(wires, like="jax")
        ranges = qml.math.array(ranges, like="jax")
        n_wires = len(wires)
        n_layers = weights.shape[0]

        @qml.for_loop(n_layers)
        def layers(l):
            @qml.for_loop(n_wires)
            def rot_loop(i):
                qml.RY(weights[l, i], wires=wires[i])
                if noise == "depolarizing" and noise_prob is not None and noise_prob > 0:
                    DepolarizingChannel(p=noise_prob, wires=wires[i])

            def imprim_true():
                @qml.for_loop(n_wires)
                def imprimitive_loop(i):
                    act_on = qml.math.array([i, i + ranges[l]], like="jax") % n_wires
                    imprimitive(wires=wires[act_on])
                    if noise == "depolarizing" and noise_prob is not None and noise_prob > 0:
                        DepolarizingChannel(p=noise_prob, wires=wires[act_on])

                imprimitive_loop()

            def imprim_false():
                pass

            rot_loop()
            qml.cond(n_wires > 1, imprim_true, imprim_false)()

        layers()
