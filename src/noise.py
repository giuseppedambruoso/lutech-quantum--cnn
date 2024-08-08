from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    phase_damping_error,
    QuantumError,
)

from abc import ABC, abstractmethod
from typing import Union


class Noise(ABC):
    """Abstract base class to create noisy quantum backends.

    Attributes
    ----------
    error_probability : float
    backend : AerSimulator

    Methods
    -------
    _create_noise_class
        Create the specific quantum error class.
    _create_noise_model
        Create the noise model corresponding to the quantum error class.
    _create_backend
        Create the noisy backend starting from the previously obtained noise
        model.
    """

    def __init__(self, error_probability: float) -> None:
        self.error_probability = error_probability
        self.backend = self._create_backend()

    @abstractmethod
    def _create_noise_class(self) -> QuantumError:
        pass

    @abstractmethod
    def _create_noise_model(self) -> NoiseModel:
        noise_class = self._create_noise_class()
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            noise_class, instructions=["id", "rz", "sx", "u1", "u2", "u3"]
        )
        return noise_model

    @abstractmethod
    def _create_backend(self) -> AerSimulator:
        noise_model = self._create_noise_model()
        backend = AerSimulator(noise_model=noise_model)
        return backend


class DepolarizingNoise(Noise):
    """
    Concrete class for creating a backend with depolarizing noise.

    Attributes
    ----------
    error_probability : float
    backend : AerSimulator

    Methods
    -------
    _create_noise_class
        Create the specific quantum error class.
    _create_noise_model
        Create the noise model corresponding to the quantum error class.
    _create_backend
        Create the noisy backend starting from the previously obtained noise
        model.
    """

    def __init__(self, error_probability: float) -> None:
        super(DepolarizingNoise, self).__init__(error_probability)

    def _create_noise_class(self) -> QuantumError:
        return depolarizing_error(self.error_probability * 4 / 3, num_qubits=1)

    def _create_noise_model(self) -> NoiseModel:
        return super()._create_noise_model()

    def _create_backend(self) -> AerSimulator:
        return super()._create_backend()


class DephasingNoise(Noise):
    """
    Concrete class for creating a backend with dephasing noise.

    Attributes
    ----------
    error_probability : float
    backend : AerSimulator

    Methods
    -------
    _create_noise_class
        Create the specific quantum error class.
    _create_noise_model
        Create the noise model corresponding to the quantum error class.
    _create_backend
        Create the noisy backend starting from the previously obtained noise
        model.
    """

    def __init__(self, error_probability: float) -> None:
        super(DephasingNoise, self).__init__(error_probability)

    def _create_noise_class(self) -> QuantumError:
        return phase_damping_error(1-(1-self.error_probability)**2)

    def _create_noise_model(self) -> NoiseModel:
        return super()._create_noise_model()

    def _create_backend(self) -> AerSimulator:
        return super()._create_backend()


def create_backend(
    error_name: Union[str, None], error_probability: float
) -> Union[AerSimulator, None]:
    """
    This function creates an AerSimulator backend with the specified noise model.

    Arguments
    ---------
    error_name : Union[str, None]
        The type of error. It can be either 'depolarizing', 'dephasing' or
        None. In the last case, the function will create a noiseless backend.
    error_probability : float
        The probability of error in the noisy backend.

    Returns:
    --------
    Union[AerSimulator, None]
        The AerSimulator backend with the specified noise model, or error_name
        was null.
    """
    if error_name == "depolarizing":
        error = DepolarizingNoise(error_probability=error_probability)
        backend = error.backend
    elif error_name == "dephasing":
        error = DephasingNoise(error_probability=error_probability)
        backend = error.backend
    elif error_name is None:
        backend = None
    else:
        raise ValueError(
            "error_name can be either \
                         'depolarizing' or 'dephasing'!"
        )

    return backend