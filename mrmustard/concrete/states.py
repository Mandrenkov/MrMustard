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

from mrmustard._typing import *
from mrmustard.abstract import State, Parametrized
from mrmustard.plugins import gaussian, train
import mrmustard as mm

__all__ = ["Vacuum", "SqueezedVacuum", "Coherent", "Thermal", "DisplacedSqueezed", "TMSV", "Gaussian"]


class Vacuum(State):
    r"""
    The N-mode vacuum state.
    """

    def __init__(self, num_modes: int = None):
        cov = gaussian.vacuum_cov(num_modes, mm.hbar)
        means = gaussian.vacuum_means(num_modes, mm.hbar)
        super().__init__(False, cov, means)


class Coherent(Parametrized, State):
    r"""
    The N-mode coherent state.
    """

    def __init__(
        self,
        x: Union[Optional[float], Optional[List[float]]] = 0.0,
        y: Union[Optional[float], Optional[List[float]]] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        Parametrized.__init__(self, x=x, y=y, x_trainable=x_trainable, y_trainable=y_trainable, x_bounds=x_bounds, y_bounds=y_bounds)
        means = gaussian.displacement(x, y, mm.hbar)
        cov = gaussian.vacuum_cov(means.shape[-1] // 2, mm.hbar)
        State.__init__(self, False, cov=cov, means=means)

    @property
    def means(self):
        return gaussian.displacement(self.x, self.y, mm.hbar)


class SqueezedVacuum(Parametrized, State):
    r"""
    The N-mode squeezed vacuum state.
    """

    def __init__(
        self,
        r: Union[Scalar, Vector] = 0.0,
        phi: Union[Scalar, Vector] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        Parametrized.__init__(
            self, r=r, phi=phi, r_trainable=r_trainable, phi_trainable=phi_trainable, r_bounds=r_bounds, phi_bounds=phi_bounds
        )
        cov = gaussian.squeezed_vacuum_cov(r, phi, mm.hbar)
        means = gaussian.vacuum_means(cov.shape[-1] // 2, mm.hbar)
        State.__init__(self, False, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.squeezed_vacuum_cov(self.r, self.phi, mm.hbar)


class TMSV(Parametrized, State):
    r"""
    The 2-mode squeezed vacuum state.
    """

    def __init__(
        self,
        r: Union[Scalar, Vector] = 0.0,
        phi: Union[Scalar, Vector] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        Parametrized.__init__(
            self, r=r, phi=phi, r_trainable=r_trainable, phi_trainable=phi_trainable, r_bounds=r_bounds, phi_bounds=phi_bounds
        )
        cov = gaussian.two_mode_squeezed_vacuum_cov(r, phi, mm.hbar)
        means = gaussian.vacuum_means(2, mm.hbar)
        State.__init__(self, False, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.two_mode_squeezed_vacuum_cov(self.r, self.phi, mm.hbar)


class Thermal(Parametrized, State):
    r"""
    The N-mode thermal state.
    """

    def __init__(
        self,
        nbar: Union[Scalar, Vector] = 0.0,
        nbar_trainable: bool = False,
        nbar_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
    ):
        Parametrized.__init__(self, nbar=nbar, nbar_trainable=nbar_trainable, nbar_bounds=nbar_bounds)
        cov = gaussian.thermal_cov(nbar, mm.hbar)
        means = gaussian.vacuum_means(cov.shape[-1] // 2, mm.hbar)
        State.__init__(self, True, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.thermal_cov(self.nbar, mm.hbar)


class DisplacedSqueezed(Parametrized, State):
    r"""
    The N-mode displaced squeezed state.
    """

    def __init__(
        self,
        r: Union[Scalar, Vector] = 0.0,
        phi: Union[Scalar, Vector] = 0.0,
        x: Union[Scalar, Vector] = 0.0,
        y: Union[Scalar, Vector] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        x_trainable: bool = False,
        y_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        Parametrized.__init__(
            self,
            r=r,
            phi=phi,
            x=x,
            y=y,
            r_trainable=r_trainable,
            phi_trainable=phi_trainable,
            x_trainable=x_trainable,
            y_trainable=y_trainable,
            r_bounds=r_bounds,
            phi_bounds=phi_bounds,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        cov = gaussian.squeezed_vacuum_cov(r, phi, mm.hbar)
        means = gaussian.displacement(x, y, mm.hbar)
        State.__init__(self, False, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.squeezed_vacuum_cov(self.r, self.phi, mm.hbar)

    @property
    def means(self):
        return gaussian.displacement(self.x, self.y, mm.hbar)


class Gaussian(Parametrized, State):
    r"""
    The N-mode Gaussian state.
    """

    def __init__(
        self,
        num_modes: int,
        symplectic: Matrix = None,
        displacement: Vector = None,
        eigenvalues: Vector = None,
        symplectic_trainable: bool = False,
        displacement_trainable: bool = False,
        eigenvalues_trainable: bool = False,
        displacement_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        if symplectic is None:
            symplectic = train.new_symplectic(num_modes=num_modes)
        if displacement is None:
            displacement = gaussian.vacuum_means(num_modes, mm.hbar)
        if eigenvalues is None:
            eigenvalues = gaussian.backend.ones_like(displacement) * mm.hbar / 2  # TODO: concrete classes should not use the backend
        Parametrized.__init__(
            self,
            symplectic=symplectic,
            symplectic_trainable=symplectic_trainable,
            symplectic_bounds=(None, None),
            displacement=displacement,
            displacement_trainable=displacement_trainable,
            displacement_bounds=displacement_bounds,
            eigenvalues=eigenvalues,
            eigenvalues_trainable=eigenvalues_trainable,
            eigenvalues_bounds=(mm.hbar / 2, None),
        )
        cov = gaussian.gaussian_cov(symplectic, eigenvalues, mm.hbar)
        means = gaussian.vacuum_means(cov.shape[-1] // 2, mm.hbar)
        State.__init__(self, None, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.gaussian_cov(self.symplectic, self.eigenvalues, mm.hbar)

    @property
    def means(self):
        return self.displacement

    @property
    def is_mixed(self):
        return any(self.eigenvalues > mm.hbar / 2)

    @property
    def is_pure(self):
        return not self.is_mixed

    @property
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        return {
            "symplectic": [self.symplectic] * self._symplectic_trainable,
            "orthogonal": [],
            "euclidean": ([self.displacement] * self._displacement_trainable + [self.eigenvalues] * self._eigenvalues_trainable),
        }
