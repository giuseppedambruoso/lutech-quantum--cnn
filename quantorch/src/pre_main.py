from typing import Sequence, List, Union

from torch.nn import Module, DataParallel

from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.neural_networks import NeuralNetwork, SamplerQNN, EstimatorQNN

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.primitives import (Sampler, Estimator, BackendSampler,
                               BackendEstimator)
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap, ZZFeatureMap

from quantorch.src.noise import NoiseCreator
from quantorch.src.net import ClassicNet, HybridNet

class ModelsCreator:
    def __init__(self,
             num_qubits: int,
             shots: Union[int, None],
             feature_map_name: str,
             feature_map_depth: int,
             feature_map_entanglement: Union[str, List[List[int]]],
             ansatz_name: str,
             ansatz_depth: int,
             ansatz_entanglement: Union[str, List[List[int]]],
             quanvolution_name: str,
             noise: NoiseCreator,
             local: bool,
             dataset: str,
             quanvolution_out_channels: Union[int, None]=None):
        
        self.num_qubits = num_qubits
        self.shots = shots
        self.feature_map_depth = feature_map_depth
        self.feature_map_entanglement = feature_map_entanglement
        self.feature_map_name = feature_map_name
        self.ansatz_depth = ansatz_depth
        self.ansatz_entanglement = ansatz_entanglement
        self.ansatz_name = ansatz_name
            
        self.quanvolution_name = quanvolution_name
        self.noise = noise
        self.local = local
        self.dataset = dataset
        self.quanvolution_out_channels = quanvolution_out_channels
        
        self.quantum_circuit = self.create_quantum_circuit()
        self.filters = self.create_filters()
        self.models = self.create_models()

    def create_quantum_circuit(self) -> QNNCircuit:
        # Create the feature map
        if self.feature_map_name == 'ZZ':
            self.feature_map = ZZFeatureMap(feature_dimension=self.num_qubits,
                                    reps=self.feature_map_depth,
                                    entanglement=self.feature_map_entanglement)
        elif self.feature_map_name == 'Z':
            self.feature_map = ZFeatureMap(feature_dimension=self.num_qubits,
                                    reps=self.feature_map_depth)
        
        # Create the ansatz
        if self.ansatz_name == 'RA':
            self.ansatz = RealAmplitudes(num_qubits=self.num_qubits,
                                  entanglement=self.ansatz_entanglement,
                                  reps=self.ansatz_depth)
               
        # Combine the two parts to create a quantum circuit
        quantum_circuit = QNNCircuit(num_qubits=self.num_qubits,
                                     feature_map=self.feature_map,
                                     ansatz=self.ansatz)
        return quantum_circuit
    
    def create_filters(self) -> Sequence:
        # Define the quantum filters depending on the quanvolution type
        ideal_filters : List[NeuralNetwork]
        noisy_filters : List[NeuralNetwork]
        if self.quanvolution_name == 'quanvolution_sampler' or self.quanvolution_name == 'No':
            ideal_filters = [SamplerQNN(circuit=self.quantum_circuit,
                                    sampler=Sampler(
                                        options={'shots': self.shots}))]
            noisy_filters = [SamplerQNN(circuit=self.quantum_circuit,
                                    sampler=BackendSampler(
                                        backend=backend,
                                        options={'shots': self.shots}))
                                for backend in self.noise.noisy_backends]
        elif self.quanvolution_name == 'quanvolution_estimator':
            observables = [Pauli('ZIII'),
                        Pauli('IZII'),
                        Pauli('IIZI'),
                        Pauli('IIIZ')]
            ideal_filters = [EstimatorQNN(circuit=self.quantum_circuit,
                                    observables=observables,
                                    estimator=Estimator(
                                        options={'shots': self.shots}))]
            noisy_filters = [EstimatorQNN(circuit=self.quantum_circuit,
                                    observables=observables,
                                    estimator=BackendEstimator(
                                        backend=backend,
                                        options={'shots': self.shots}))
                                for backend in self.noise.noisy_backends]
        return ideal_filters + noisy_filters

    def create_models(self) -> Sequence[Module]:
        models : List[Union[ClassicNet, HybridNet]] = []
        classical_model = DataParallel(ClassicNet(
                                kernel_size=int(self.num_qubits**0.5),
                                quanvolution_name=self.quanvolution_name,
                                dataset=self.dataset,
                                local=self.local,
                                quanvolution_out_channels=self.quanvolution_out_channels))
        models.append(classical_model)
        # for i in range(len(self.filters)):
        #     hybrid_model = DataParallel(HybridNet(
        #                         local=self.local,
        #                         quantum_filter=self.filters[i],
        #                         quanvolution_name=self.quanvolution_name,
        #                         quanvolution_out_channels=self.quanvolution_out_channels,
        #                         dataset=self.dataset))
        #     models.append(hybrid_model)        
        return models