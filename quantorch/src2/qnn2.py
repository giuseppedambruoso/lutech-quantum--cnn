from typing import List, Union
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.neural_networks import NeuralNetwork, SamplerQNN

from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler, BackendSampler
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap, ZZFeatureMap


class QNN:
    """
    Quantum Neural Network (QNN) class for constructing variational quantum 
    circuits for quantum machine learning tasks.

    Attributes:
        num_qubits (int): Number of qubits in the quantum circuit.
        feature_map_name (str): Name of the feature map to use ('ZZ' or 'Z').
        feature_map_depth (int): Depth of the feature map circuit.
        feature_map_entanglement (str or List[List[int]]): Entanglement 
            strategy for the feature map.
        ansatz_name (str): Name of the ansatz circuit to use ('RA').
        ansatz_depth (int): Depth of the ansatz circuit.
        ansatz_entanglement (str or List[List[int]]): Entanglement
            strategy for the ansatz circuit.
        backend (AerSimulator or None): Backend simulator for running the
            quantum circuit.
        shots (int or None): Number of shots for sampling. Non for exact
            calculation.
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
        if self.feature_map_name == 'ZZ':
            self.feature_map = ZZFeatureMap(
                  feature_dimension=self.num_qubits,
                  reps=self.feature_map_depth,
                  entanglement=self.feature_map_entanglement
            )
        elif self.feature_map_name == 'Z':
            self.feature_map = ZFeatureMap(
                  feature_dimension=self.num_qubits,
                  reps=self.feature_map_depth
            )
        else:
            raise AttributeError(
                'The only possible feature maps are Z and ZZ!'
            )
        
        # Create the ansatz
        if self.ansatz_name == 'RA':
            self.ansatz = RealAmplitudes(
                  num_qubits=self.num_qubits,
                  entanglement=self.ansatz_entanglement,
                  reps=self.ansatz_depth
            )
        else:
            raise AttributeError(
                'The only possible ansatz is RA!'
            )
        
        # Combine the two parts to create a variational quantum circuit
        quantum_circuit = QNNCircuit(
             num_qubits=self.num_qubits,
             feature_map=self.feature_map,
             ansatz=self.ansatz
        )
        return quantum_circuit
    
    def _create_filter(self) -> NeuralNetwork:
        qnn : NeuralNetwork
        if self.backend == None: # noiseless case
            qnn = SamplerQNN(
                    circuit=self.quantum_circuit,
                    sampler=Sampler(
                        options={'shots': self.shots}
                    )
            )
        else : # noisy case
            qnn = SamplerQNN(
                    circuit=self.quantum_circuit,
                    sampler=BackendSampler(
                        backend=self.backend,
                        options={'shots': self.shots}
                    )
            )
        return qnn