from typing import List, Any, Callable

from qiskit_aer import AerSimulator
from qiskit_aer.backends.aer_simulator import AerBackend
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error

class NoiseCreator:
    def __init__(self,
                 errors_names: List[str],
                 errors_probabilities: List[float]):

        self.errors_names = errors_names
        self.errors_probabilities = errors_probabilities
        self.noise_classes = self.create_noise_classes()
        self.noise_models = self.create_noise_models()
        self.noisy_backends = self.create_noisy_backends()

    def create_noise_classes(self) -> List:
        # Define the errors with their respective probabilities
        noise_classes : List[Any] = []
        for i in range(len(self.errors_names)):
            error_probability = self.errors_probabilities[i]
            error_name = self.errors_names[i]
            noise_class : Callable
            if error_name == 'bitflip':
                noise_class = pauli_error([('X', error_probability),
                                    ('I', 1 - error_probability)])
            elif error_name == 'phaseflip':
                noise_class = pauli_error([('Y', error_probability),
                                      ('I', 1 - error_probability)])
            elif error_name == 'bothflips':
                noise_class = pauli_error([('Z', error_probability),
                                      ('I', 1 - error_probability)])
            elif error_name == 'depolarizing':
                noise_class = depolarizing_error(error_probability*4/3,
                                                 num_qubits=1)
            noise_classes.append(noise_class)
        return noise_classes
    
    def create_noise_models(self) -> List[NoiseModel]:
        # Create noise models
        noise_models = []
        for i in range(len(self.noise_classes)):
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(self.noise_classes[i],
                                                    instructions=['id',
                                                                  'rz',
                                                                  'sx',
                                                                  'u1',
                                                                  'u2',
                                                                  'u3'])
        
            noise_models.append(noise_model)       
        return noise_models
    
    def create_noisy_backends(self) -> List[AerBackend]:
        return [AerSimulator(noise_model = noise_model)
                          for noise_model in self.noise_models]