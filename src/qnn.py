from typing import List, Union
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN

from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler, BackendSampler
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap, ZZFeatureMap


class QNN:
    """Class for constructing a quantum neural network (QNN) starting from
    a variational quantum circuit (VQC).

    Attributes:
    num_qubits : int
        The number of qubits in the VQC.
    feature_map_name :str
        The ame of the feature map to use ('ZZ' or 'Z').
    feature_map_depth : int
        Theepth of VQC's feature map.
    feature_map_entanglement : Union[str, List[List[int]]]
        The entanglement strategy for the feature map.
    ansatz_name : str
        The name of the ansatz circuit to use ('RA').
    ansatz_depth : int
        The depth of the VQC's ansatz.
    ansatz_entanglement : Union[str, List[List[int]]]
        The entanglement pattern for the ansatz.
    backend : Union[AerSimulator, None]
        Backend simulator for running the VQC.
    shots : Union[int, None]
        The umber of shots for sampling. None for exact computation.
    quantum_circuit : QNNCircuit
        The VQC.
    qnn : SamplerQNN
        The VQC as QNN, i.e. the VQC provided with the input parameters and
        the learning algorithm (parameter shift gradient).

    Methods
    -------
    _create_quantum_circuit
        Create the VQC.
    _create_filter
        Transform the VQC into a QNN.

    """

    def __init__(
        self,
        num_qubits: int,
        feature_map_name: str,
        feature_map_depth: int,
        feature_map_entanglement: Union[str, List[List[int]]],
        ansatz_name: str,
        ansatz_depth: int,
        ansatz_entanglement: Union[str, List[List[int]]],
        backend: Union[AerSimulator, None],
        shots: Union[int, None] = None,
    ):

        self.num_qubits = num_qubits
        self.shots = shots
        self.feature_map_depth = feature_map_depth
        self.feature_map_entanglement = feature_map_entanglement
        self.feature_map_name = feature_map_name
        self.ansatz_depth = ansatz_depth
        self.ansatz_entanglement = ansatz_entanglement
        self.ansatz_name = ansatz_name
        self.backend = backend

        self.quantum_circuit = self._create_quantum_circuit()
        self.qnn = self._create_filter()

    def _create_quantum_circuit(self) -> QNNCircuit:
        """
        Create the variational quantum circuit by combining feature map and
        ansatz.
        """
        # Create the feature map
        if self.feature_map_name == "ZZ":
            self.feature_map = ZZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_depth,
                entanglement=self.feature_map_entanglement,
            )
        elif self.feature_map_name == "Z":
            self.feature_map = ZFeatureMap(
                feature_dimension=self.num_qubits, reps=self.feature_map_depth
            )
        else:
            raise AttributeError("The only possible feature maps are Z and ZZ!")

        # Create the ansatz
        if self.ansatz_name == "RA":
            self.ansatz = RealAmplitudes(
                num_qubits=self.num_qubits,
                entanglement=self.ansatz_entanglement,
                reps=self.ansatz_depth,
            )
        else:
            raise AttributeError("The only possible ansatz is RA!")

        # Combine the two parts to create a variational quantum circuit
        quantum_circuit = QNNCircuit(
            num_qubits=self.num_qubits, feature_map=self.feature_map, ansatz=self.ansatz
        )
        return quantum_circuit

    def _create_filter(self) -> SamplerQNN:
        qnn: SamplerQNN
        if self.backend == None:  # noiseless case
            qnn = SamplerQNN(
                circuit=self.quantum_circuit,
                sampler=Sampler(options={"shots": self.shots}),
            )
        else:  # noisy case
            qnn = SamplerQNN(
                circuit=self.quantum_circuit,
                sampler=BackendSampler(
                    backend=self.backend, options={"shots": self.shots}
                ),
            )
        return qnn
