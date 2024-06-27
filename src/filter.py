from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_algorithms.gradients import ReverseEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN

def AngleEncoding(input_dim):
    num_qubits = input_dim
    params = ParameterVector("x", length=num_qubits)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    return qc


class QuantumFilter:
    """Quantum Filter for QNN"""

    def __init__(self, input_dim, entanglement, shots, depth):
        self.input_dim = input_dim
        self.entanglement = entanglement
        self.shots = shots
        self.feature_map = AngleEncoding(input_dim=self.input_dim)
        self.nontrainable_params = self.feature_map.parameters
        self.ansatz = RealAmplitudes(
            num_qubits=self.input_dim, entanglement=self.entanglement, reps=depth
        )
        self.trainable_params = self.ansatz.parameters
        self.pqc = self.build_circuit()
        self.qnn = self.build_qnn()

    def build_circuit(self):
        """Creates a PQC with arbitrary number of qubits and depth"""
        n_qubits = self.input_dim
        self.pqc = QuantumCircuit(n_qubits)

        self.pqc.compose(self.feature_map, inplace=True)
        self.pqc.compose(self.ansatz, inplace=True)
        return self.pqc

    def build_qnn(self):
        """Creates a QNN with the predefined PQC"""
        qnn = EstimatorQNN(
            circuit=self.pqc,
            estimator=Estimator(options={"shots": self.shots}),
            input_params=self.nontrainable_params,
            weight_params=self.trainable_params,
            gradient=ReverseEstimatorGradient,
            input_gradients=True,
        )
        qnn = TorchConnector(qnn)
        return qnn
