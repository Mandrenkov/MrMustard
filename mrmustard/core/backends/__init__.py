from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, List, Tuple, Sequence, Callable, Union


class MathBackendInterface(ABC):
    def conj(self, array):
        ...

    def diag(self, array):
        ...

    def reshape(self, array, shape):
        ...

    def sum(self, array, axis=None):
        ...

    def arange(self, start, limit=None, delta=1):
        ...

    def outer(self, arr1, arr2):
        ...

    def identity(self, size: int):
        ...

    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype):
        ...

    def abs(self, array):
        ...

    def trace(self, array):
        ...

    def tensordot(self, a, b, axes, dtype=None):
        ...

    def transpose(self, a, perm):
        ...

    def block(self, blocks: List[List]):
        ...

    def concat(self, values, axis):
        ...

    def norm(self, array):
        ...

    def add(self, old, new: Optional, modes: List[int]):
        ...

    def sandwich(self, bread: Optional, filling, modes: List[int]):
        ...

    def matvec(self, mat: Optional, vec, modes: List[int]):
        ...

    def new_symplectic_parameter(
        self,
        init_value: Optional = None,
        trainable: bool = True,
        num_modes: int = 1,
        name: str = "symplectic",
    ):
        ...

    def unitary_to_orthogonal(self, U):
        ...

    def new_euclidean_parameter(
        self,
        init_value: Optional[Union[float, List[float]]] = None,
        trainable: bool = True,
        bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        shape: Optional[Sequence[int]] = None,
        name: str = "",
    ):
        ...

    def poisson(self, max_k: int, rate):
        ...

    def binomial_conditional_prob(self, success_prob, dim_out: int, dim_in: int):
        ...

    def convolve_probs_1d(self, prob, other_probs: List):
        ...

    def convolve_probs(self, prob, other):
        ...


class SymplecticBackendInterface(ABC):
    @abstractmethod
    def loss_X(self, transmissivity):
        ...

    @abstractmethod
    def loss_Y(self, transmissivity, hbar: float):
        ...

    @abstractmethod
    def thermal_X(self, nbar, hbar: float):
        ...

    @abstractmethod
    def thermal_Y(self, nbar, hbar: float):
        ...

    @abstractmethod
    def displacement(self, x, y, hbar: float):
        ...

    @abstractmethod
    def beam_splitter_symplectic(self, theta, phi):
        ...

    @abstractmethod
    def rotation_symplectic(self, phi):
        ...

    @abstractmethod
    def squeezing_symplectic(self, r, phi):
        ...

    @abstractmethod
    def two_mode_squeezing_symplectic(self, r, phi):
        ...


class OptimizerBackendInterface(ABC):
    @abstractmethod
    def loss_and_gradients(self, symplectic_params: Sequence, euclidean_params: Sequence, cost_fn: Callable):
        ...

    @abstractmethod
    def update_symplectic(self, symplectic_grads: Sequence, symplectic_params: Sequence):
        ...

    @abstractmethod
    def update_euclidean(self, euclidean_grads: Sequence, euclidean_params: Sequence):
        ...

    @abstractmethod
    def extract_symplectic_parameters(self, items: Sequence):
        ...

    @abstractmethod
    def extract_euclidean_parameters(self, items: Sequence):
        ...


class StateBackendInterface(ABC):
    @abstractmethod
    def number_means(self, cov, means, hbar: float):
        ...

    @abstractmethod
    def number_cov(self, cov, means, hbar: float):
        ...

    @abstractmethod
    def ABC(self, cov, means, mixed: bool = False, hbar: float = 2.0):
        ...

    @abstractmethod
    def fock_state(self, A, B, C, cutoffs: Sequence[int]):
        ...