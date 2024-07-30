from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, 
    depolarizing_error,
    phase_damping_error,
    QuantumError
)
from abc import ABC, abstractmethod

class Noise(ABC):
    """
    Abstract base class for creating noise models to be used with AerSimulator.

    Attributes:
        error_probability (float): The probability of error in the noise model.
        backend (AerSimulator): The AerSimulator backend with the noise model applied.
    """

    def __init__(self, error_probability: float) -> None:
        """
        Initialize the Noise class with a given error probability.

        Args:
            error_probability (float): The probability of error in the noise model.
        """
        self.error_probability = error_probability
        self.backend = self._create_backend()

    @abstractmethod
    def _create_noise_class(self) -> QuantumError:
        """
        Create the specific quantum error class.

        Returns:
            QuantumError: The quantum error class.
        """
        pass

    @abstractmethod
    def _create_noise_model(self) -> NoiseModel:
        """
        Create the noise model using the quantum error class.

        Returns:
            NoiseModel: The noise model to be applied to the simulator.
        """
        noise_class = self._create_noise_class()
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            noise_class,
            instructions=['id', 'rz', 'sx', 'u1', 'u2', 'u3']
        )
        return noise_model

    @abstractmethod
    def _create_backend(self) -> AerSimulator:
        """
        Create the AerSimulator backend with the noise model.

        Returns:
            AerSimulator: The AerSimulator backend with the noise model applied.
        """
        noise_model = self._create_noise_model()
        backend = AerSimulator(noise_model=noise_model)
        return backend

class DepolarizingNoise(Noise):
    """
    Class for creating depolarizing noise models.
    """

    def __init__(self, error_probability: float) -> None:
        """
        Initialize the DepolarizingNoise class with a given error probability.

        Args:
            error_probability (float): The probability of depolarizing error.
        """
        super(DepolarizingNoise, self).__init__(error_probability)

    def _create_noise_class(self) -> QuantumError:
        """
        Create the depolarizing error class.

        Returns:
            QuantumError: The depolarizing quantum error.
        """
        return depolarizing_error(self.error_probability * 4 / 3, num_qubits=1)

class DephasingNoise(Noise):
    """
    Class for creating dephasing noise models.
    """

    def __init__(self, error_probability: float) -> None:
        """
        Initialize the DephasingNoise class with a given error probability.

        Args:
            error_probability (float): The probability of dephasing error.
        """
        super(DephasingNoise, self).__init__(error_probability)

    def _create_noise_class(self) -> QuantumError:
        """
        Create the dephasing error class.

        Returns:
            QuantumError: The dephasing quantum error.
        """
        return phase_damping_error(self.error_probability * 4 / 3, num_qubits=1)

def create_backend(
    error_name: str | None,
    error_probability: float
) -> AerSimulator | None:
    """
    Create an AerSimulator backend with the specified noise model.

    Args:
        error_name (Union[str, None]): The type of error model ('depolarizing' or 'dephasing'). If None, no noise model is applied.
        error_probability (float): The probability of error in the noise model.

    Returns:
        Union[AerSimulator, None]: The AerSimulator backend with the specified noise model, or None if no noise model is applied.
    """
    if error_name == 'depolarizing':
        error = DepolarizingNoise(error_probability=error_probability)
        backend = error.backend
    elif error_name == 'dephasing':
        error = DephasingNoise(error_probability=error_probability)
        backend = error.backend
    elif error_name is None:
        backend = None
    else:
        raise ValueError("error_name can be either 'depolarizing' or 'dephasing'!")
    
    return backend

print()