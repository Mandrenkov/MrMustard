# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from mrmustard.utils.types import *
from mrmustard import settings
import importlib


def _set_backend(backend_name: str):
    "This private function is called by the Settings object to set the math backend in this module"
    Math = importlib.import_module(f"mrmustard.math.{backend_name}").Math
    globals()["math"] = Math()  # setting global variable only in this module's scope


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~ static functions ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def fock_representation(cov: Matrix, means: Vector, cutoffs: Sequence[int], mixed: bool) -> Tensor:
    r"""
    Returns the Fock representation of the phase space representation
    given a Wigner covariance matrix and a means vector. If the state is pure
    it returns the ket, if it is mixed it returns the density matrix.
    Args:
        cov: The Wigner covariance matrix.
        means: The Wigner means vector.
        cutoffs: The shape of the tensor.
        mixed: Whether the state vector is mixed or not.
    Returns:
        The Fock representation of the phase space representation.
    """
    assert len(cutoffs) == means.shape[-1] // 2 == cov.shape[-1] // 2
    A, B, C = hermite_parameters(cov, means, mixed)
    return math.hermite_renormalized(math.conj(-A), math.conj(B), math.conj(C), shape=cutoffs + cutoffs if mixed else cutoffs)


def bell_norm(r: float, cutoff: int) -> Scalar:
    return (np.tanh(r) ** np.arange(cutoff)) / np.cosh(r) + 0.0j


def normalize_choi_trick(unnormalized: Tensor, r: float) -> Tensor:
    r"""
    Normalizes the columns of an operator obtained by applying it to TMSV(r).
    Args:
        unnormalized: The unnormalized operator
        r: The value of the Choi squeezing
    Returns:
        The normalized operator.
    """
    col_cutoffs = unnormalized.shape[1::2]
    norm = math.reshape(bell_norm(r, col_cutoffs[0]), -1)
    for i, c in enumerate(col_cutoffs[1:]):
        norm = math.reshape(math.outer(norm, bell_norm(r, c)), -1)
    normalized = math.reshape(unnormalized, (-1, np.prod(col_cutoffs))) / norm[None, :]
    return math.reshape(normalized, unnormalized.shape)


def ket_to_dm(ket: Tensor) -> Tensor:
    r"""
    Maps a ket to a density matrix.
    Args:
        ket: The ket.
    Returns:
        The density matrix.
    """
    return math.outer(ket, math.conj(ket))


def ket_to_probs(ket: Tensor) -> Tensor:
    r"""
    Maps a ket to probabilities.
    Args:
        ket: The ket.
    Returns:
        The probabilities vector.
    """
    return math.abs(ket) ** 2


def dm_to_probs(dm: Tensor) -> Tensor:
    r"""
    Extracts the diagonals of a density matrix.
    Args:
        dm: The density matrix.
    Returns:
        The probabilities vector.
    """
    return math.all_diagonals(dm, real=True)


def hermite_parameters(cov: Matrix, means: Vector, mixed: bool) -> Tuple[Matrix, Vector, Scalar]:
    r"""
    Returns the A matrix, B vector and C scalar given a Wigner covariance matrix and a means vector of an N-mode state.
    The A, B, C triple is needed to compute the Fock representation of the state.
    If the state is pure, then A has shape (N, N), B has shape (N) and C has shape ().
    If the state is mixed, then A has shape (2N, 2N), B has shape (2N) and C has shape ().
    Args:
        cov: The Wigner covariance matrix.
        means: The Wigner means vector.
        mixed: Whether the state vector is mixed or not.
    Returns:
        The A matrix, B vector and C scalar.
    """
    num_indices = means.shape[-1]
    num_modes = num_indices // 2

    # cov and means in the amplitude basis
    R = math.rotmat(num_indices // 2)
    sigma = math.matmul(math.matmul(R, cov / settings.HBAR), math.dagger(R))
    beta = math.matvec(R, means / math.sqrt(settings.HBAR, dtype=means.dtype))

    sQ = sigma + 0.5 * math.eye(num_indices, dtype=sigma.dtype)
    sQinv = math.inv(sQ)
    X = math.Xmat(num_modes)
    A = math.matmul(X, math.eye(num_indices, dtype=sQinv.dtype) - sQinv)
    B = math.matvec(math.transpose(sQinv), math.conj(beta))
    exponent = -0.5 * math.sum(math.conj(beta)[:, None] * sQinv * beta[None, :])
    T = math.exp(exponent) / math.sqrt(math.det(sQ))
    N = 2 * num_modes if mixed else num_modes
    return (
        A[:N, :N],
        B[:N],
        T ** (1.0 if mixed else 0.5),
    )  # will be off by global phase because T is real even for pure states


def fidelity(state_a, state_b, a_pure: bool = True, b_pure: bool = True) -> Scalar:
    r"""computes the fidelity between two states in Fock representation"""
    if a_pure and b_pure:
        return math.abs(math.sum(math.conj(state_a) * state_b)) ** 2
    elif a_pure:
        a = math.reshape(state_a, -1)
        return math.real(math.sum(math.conj(a) * math.matvec(math.reshape(state_b, (len(a), len(a))), a)))
    elif b_pure:
        b = math.reshape(state_b, -1)
        return math.real(math.sum(math.conj(b) * math.matvec(math.reshape(state_a, (len(b), len(b))), b)))
    else:
        raise NotImplementedError("Fidelity between mixed states is not implemented yet")
