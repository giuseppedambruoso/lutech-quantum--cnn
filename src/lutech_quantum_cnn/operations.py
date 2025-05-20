from torch import float64

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import RX, RY, RZ
from pennylane.ops.channel import DepolarizingChannel
from pennylane.wires import WiresLike
from pennylane.typing import TensorLike

ROT = {"X": RX, "Y": RY, "Z": RZ}

class AngleEmbedding(Operation):
    r"""
    Encodes :math:`N` features into the rotation angles of :math:`n` qubits, where :math:`N \leq n`.

    The rotations can be chosen as either :class:`~pennylane.ops.RX`, :class:`~pennylane.ops.RY`
    or :class:`~pennylane.ops.RZ` gates, as defined by the ``rotation`` parameter:

    * ``rotation='X'`` uses the features as angles of RX rotations

    * ``rotation='Y'`` uses the features as angles of RY rotations

    * ``rotation='Z'`` uses the features as angles of RZ rotations

    The length of ``features`` has to be smaller or equal to the number of qubits. If there are fewer entries in
    ``features`` than rotations, the circuit does not apply the remaining rotation gates.

    Args:
        features (tensor_like): input tensor of shape ``(N,)``, where N is the number of input features to embed,
            with :math:`N\leq n`
        wires (Any or Iterable[Any]): wires that the template acts on
        rotation (str): type of rotations used
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    Example:

        Angle embedding encodes the features by using the specified rotation operation.

        .. code-block:: python

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(feature_vector):
                qml.AngleEmbedding(features=feature_vector, wires=range(3), rotation='Z')
                qml.Hadamard(0)
                return qml.probs(wires=range(3))

            X = [1,2,3]

        Here, we have also used rotation angles :class:`RZ`. If not specified, :class:`RX` is used as default.
        The resulting circuit is:

        >>> print(qml.draw(circuit, level="device")(X))
        0: ──RZ(1.00)──H─┤ ╭Probs
        1: ──RZ(2.00)────┤ ├Probs
        2: ──RZ(3.00)────┤ ╰Probs

    """

    num_wires = AnyWires
    grad_method = None # type: ignore

    def _flatten(self):
        hyperparameters = (("rotation", self._rotation),)
        return self.data, (self.wires, hyperparameters)

    def __repr__(self):
        return f"AngleEmbedding({self.data[0]}, wires={self.wires.tolist()}, rotation={self._rotation})"

    def __init__(self,
                 features,
                 wires,
                 noise : str | None = None,
                 noise_prob : float | None = None,
                 rotation="X",
                 id=None):
        if rotation not in ROT:
            raise ValueError(f"Rotation option {rotation} not recognized.")

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
    def compute_decomposition(
        features,
        wires,
        rotation,
        noise,
        noise_prob
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.AngleEmbedding.decomposition`.

        Args:
            features (tensor_like): input tensor of dimension ``(len(wires),)``
            wires (Any or Iterable[Any]): wires that the operator acts on
            rotation (.Operator): rotation gate class

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> features = torch.tensor([1., 2.])
        >>> qml.AngleEmbedding.compute_decomposition(features, wires=["a", "b"], rotation=qml.RX)
        [RX(tensor(1.), wires=['a']),
         RX(tensor(2.), wires=['b'])]
        """
        batched = qml.math.ndim(features) > 1
        # We will iterate over the first axis of `features` together with iterating over the wires.
        # If the leading dimension is a batch dimension, exchange the wire and batching axes.
        features = qml.math.T(features) if batched else features
        
        decomposition = []
        for i in range(len(wires)):
            decomposition.append(rotation(features[i], wires=wires[i]))
            if noise == 'depolarizing' and noise_prob is not None and noise_prob > 0:
                decomposition.append(DepolarizingChannel(p=noise_prob, wires=wires[i]))

        return decomposition


class RealAmplitudes(Operation):
    r"""Layers consisting of single qubit rotations and entanglers, inspired by the circuit-centric classifier design
    `arXiv:1804.00633 <https://arxiv.org/abs/1804.00633>`_.

    The argument ``weights`` contains the weights for each layer. The number of layers :math:`L` is therefore derived
    from the first dimension of ``weights``.

    The 2-qubit gates, whose type is specified by the ``imprimitive`` argument,
    act chronologically on the :math:`M` wires, :math:`i = 1,...,M`. The second qubit of each gate is given by
    :math:`(i+r)\mod M`, where :math:`r` is a  hyperparameter called the *range*, and :math:`0 < r < M`.
    If applied to one qubit only, this template will use no imprimitive gates.

    This is an example of two 4-qubit strongly entangling layers (ranges :math:`r=1` and :math:`r=2`, respectively) with
    rotations :math:`RY` and CNOTs as imprimitives:

    .. figure:: ../../_static/layer_sec_ry.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    .. note::
        The two-qubit gate used as the imprimitive or entangler must not depend on parameters.

    Args:

        weights (tensor_like): weight tensor of shape ``(L, M)``
        wires (Iterable): wires that the template acts on
        ranges (Sequence[int]): sequence determining the range hyperparameter for each subsequent layer; if ``None``
                                using :math:`r=l \mod M` for the :math:`l` th layer and :math:`M` wires.
        imprimitive (type of pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`

    Example:

        There are multiple arguments that the user can use to customize the layer.

        The required arguments are ``weights`` and ``wires``.

        .. code-block:: python

            dev = qml.device('default.qubit', wires=4)

            @qml.qnode(dev)
            def circuit(parameters):
                qml.StronglyEntanglingLayers(weights=parameters, wires=range(4))
                return qml.expval(qml.Z(0))

            shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
            weights = np.random.random(size=shape)

        The shape of the ``weights`` argument decides the number of layers.

        The resulting circuit is:

        >>> print(qml.draw(circuit, level="device")(weights))
        0: ──RY(0.68)─╭●───────╭X──RY(0.94)─╭●────╭X────┤  <Z>
        1: ──RY(0.91)─╰X─╭●────│───RY(0.50)─│──╭●─│──╭X─┤
        2: ──RY(0.91)────╰X─╭●─│───RY(0.14)─╰X─│──╰●─│──┤
        3: ──RY(0.46)───────╰X─╰●──RY(0.87)────╰X────╰●─┤

        The default two-qubit gate used is :class:`~pennylane.ops.CNOT`. This can be changed by using the ``imprimitive`` argument.

        The ``ranges`` argument takes an integer sequence where each element
        determines the range hyperparameter for each layer. This range hyperparameter
        is the difference of the wire indices representing the two qubits the
        ``imprimitive`` gate acts on. For example, for ``range=[2,3]`` the
        first layer will have a range parameter of ``2`` and the second layer will
        have a range parameter of ``3``.
        Assuming ``wires=[0, 1, 2, 3]`` and a range parameter of ``2``, there will be
        an imprimitive gate acting on:

        * qubits ``(0, 2)``;
        * qubits ``(1, 3)``;
        * qubits ``(2, 0)``;
        * qubits ``(3, 1)``.

        .. code-block:: python

            dev = qml.device('default.qubit', wires=4)

            @qml.qnode(dev)
            def circuit(parameters):
                qml.StronglyEntanglingLayers(weights=parameters, wires=range(4), ranges=[2, 3], imprimitive=qml.ops.CZ)
                return qml.expval(qml.Z(0))

            shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
            weights = np.random.random(size=shape)

        The resulting circuit is:

        >>> print(qml.draw(circuit, level="device")(weights))
        0: ──RY(0.99)─╭●────╭Z──RY(0.02)──────────────────────╭●─╭Z───────┤  <Z>
        1: ──RY(0.55)─│──╭●─│──╭Z────────────────────RY(0.15)─│──╰●─╭Z────┤
        2: ──RY(0.79)─╰Z─│──╰●─│─────────────────────RY(0.73)─│─────╰●─╭Z─┤
        3: ──RY(0.30)────╰Z────╰●────────────────────RY(0.57)─╰Z───────╰●─┤

    .. details::
        :title: Usage Details

        **Parameter shape**

        The expected shape for the weight tensor can be computed with the static method
        :meth:`~.qml.StronglyEntanglingLayers.shape` and used when creating randomly
        initialised weight tensors:

        .. code-block:: python

            shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
            weights = np.random.random(size=shape)

    """

    num_wires = AnyWires
    grad_method = None # type: ignore

    def __init__(
            self,
            weights,
            wires,
            noise: str | None=None,
            noise_prob: float | None=None,
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
                # tile ranges with iterations of range(1, n_wires)
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
    def compute_decomposition(
        weights: TensorLike, wires, ranges, imprimitive, noise, noise_prob
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.StronglyEntanglingLayers.decomposition`.

        Args:
            weights (tensor_like): weight tensor
            wires (Any or Iterable[Any]): wires that the operator acts on
            ranges (Sequence[int]): sequence determining the range hyperparameter for each subsequent layer
            imprimitive (pennylane.ops.Operation): two-qubit gate to use

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> weights = torch.tensor([[-0.2, 0.1], [1.2, -2.]])
        >>> qml.StronglyEntanglingLayers.compute_decomposition(weights, wires=["a", "b"], ranges=[2], imprimitive=qml.CNOT)
        [RY(tensor(-0.2000), wires=['a']),
        RY(tensor(0.1000), wires=['b']),
        CNOT(wires=['a', 'a']),
        RY(tensor(1.2000), wires=['a']),
        RY(tensor(-2.), wires=['b']),
        CNOT(wires=['b', 'b'])]
        """
        n_layers = qml.math.shape(weights)[-2]
        wires = qml.wires.Wires(wires)
        op_list = []

        for l in range(n_layers):
            for i in range(len(wires)):  # pylint: disable=consider-using-enumerate
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
        r"""Returns the expected shape of the weights tensor.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires

        Returns:
            tuple[int]: shape
        """

        return n_layers, n_wires

    # pylint:disable = no-value-for-parameter
    @staticmethod
    def compute_qfunc_decomposition(
        weights, *wires, ranges, imprimitive, noise, noise_prob
    ):  # pylint: disable=arguments-differ
        wires = qml.math.array(wires, like="jax")
        ranges = qml.math.array(ranges, like="jax")

        n_wires = len(wires)
        n_layers = weights.shape[0]

        @qml.for_loop(n_layers)
        def layers(l):
            @qml.for_loop(n_wires)
            def rot_loop(i):
                qml.RY(
                    weights[l, i],
                    wires=wires[i],
                )
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