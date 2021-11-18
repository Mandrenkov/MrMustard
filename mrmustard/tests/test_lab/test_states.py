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
import pytest
from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays
from mrmustard.physics import gaussian as gp
from mrmustard.lab.states import *
from mrmustard.lab.gates import *
from mrmustard import settings
from mrmustard.tests import random


@given(st.integers(0, 10), st.floats(0.1, 5.0))
def test_vacuum_state(num_modes, hbar):
    cov, disp = gp.vacuum_cov(num_modes, hbar), gp.vacuum_means(num_modes, hbar)
    assert np.allclose(cov, np.eye(2 * num_modes) * hbar / 2)
    assert np.allclose(disp, np.zeros_like(disp))


@given(x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_coherent_state_single(x, y):
    state = Coherent(x, y)
    assert np.allclose(state.cov, np.array([[settings.HBAR / 2, 0], [0, settings.HBAR / 2]]))
    assert np.allclose(state.means, np.array([x, y]) * np.sqrt(2 * settings.HBAR))


@given(hbar=st.floats(0.5, 2.0), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_coherent_state_list(hbar, x, y):
    assert np.allclose(gp.displacement([x], [y], hbar), np.array([x, y]) * np.sqrt(2 * hbar))


@given(hbar=st.floats(0.5, 2.0), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_coherent_state_array(hbar, x, y):
    assert np.allclose(gp.displacement(np.array([x]), np.array([y]), hbar), np.array([x, y]) * np.sqrt(2 * hbar))


@given(r=st.floats(0.0, 10.0), phi=st.floats(0.0, 2 * np.pi), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_displaced_squeezed_state(r, phi, x, y):
    state = DisplacedSqueezed(r, phi, x, y)
    cov, means = state.cov, state.means
    S = Sgate(r=r, phi=phi)
    D = Dgate(x=x, y=y)
    state = D[0](S[0](Vacuum(num_modes=1)))
    assert np.allclose(cov, state.cov, rtol=1e-3)
    assert np.allclose(means, state.means)


@st.composite
def xy_arrays(draw):
    length = draw(st.integers(2, 10))
    return draw(arrays(dtype=np.float, shape=(2, length), elements=st.floats(-5.0, 5.0)))


n = st.shared(st.integers(2, 10))
arr = arrays(dtype=np.float, shape=(n), elements=st.floats(-5.0, 5.0))


@given(x=arr, y=arr)
def test_coherent_state_multiple(x, y):
    state = Coherent(x, y)
    assert np.allclose(state.cov, np.eye(2 * len(x)) * settings.HBAR / 2)
    assert len(x) == len(y)
    assert np.allclose(state.means, np.concatenate([x, y], axis=-1) * np.sqrt(2 * settings.HBAR))


@given(xy=xy_arrays())
def test_the_purity_of_a_pure_state(xy):
    x, y = xy
    state = Coherent(x, y)
    purity = gp.purity(state.cov, settings.HBAR)
    expected = 1.0
    assert np.isclose(purity, expected)


@given(nbar=st.floats(0.0, 3.0))
def test_the_purity_of_a_mixed_state(nbar):
    state = Thermal(nbar)
    purity = gp.purity(state.cov, settings.HBAR)
    expected = 1 / (2 * nbar + 1)
    assert np.isclose(purity, expected)


@given(r1=st.floats(0.0, 1.0), phi1=st.floats(0.0, 2 * np.pi), r2=st.floats(0.0, 1.0), phi2=st.floats(0.0, 2 * np.pi))
def test_join_two_states(r1, phi1, r2, phi2):
    S1 = Sgate(r=r1, phi=phi1)[0](Vacuum(1))
    S2 = Sgate(r=r2, phi=phi2)[0](Vacuum(1))
    S12 = Sgate(r=[r1, r2], phi=[phi1, phi2])[0,1](Vacuum(2))
    assert np.allclose((S1 & S2).cov, S12.cov)


@given(
    r1=st.floats(0.0, 1.0),
    phi1=st.floats(0.0, 2 * np.pi),
    r2=st.floats(0.0, 1.0),
    phi2=st.floats(0.0, 2 * np.pi),
    r3=st.floats(0.0, 1.0),
    phi3=st.floats(0.0, 2 * np.pi),
)
def test_join_three_states(r1, phi1, r2, phi2, r3, phi3):
    S1 = Sgate(r=r1, phi=phi1)[0](Vacuum(1))
    S2 = Sgate(r=r2, phi=phi2)[0](Vacuum(1))
    S3 = Sgate(r=r3, phi=phi3)[0](Vacuum(1))
    S123 = Sgate(r=[r1, r2, r3], phi=[phi1, phi2, phi3])[0, 1, 2](Vacuum(3))
    assert np.allclose((S1 & S2 & S3).cov, S123.cov)

@given(x=random.array_of_(random.medium_float, 2), y=random.array_of_(random.medium_float, 2))
def test_coh_state_is_same_as_dgate_on_vacuum(x, y):
    state = Coherent(np.array(x), np.array(y))
    expected = Vacuum(2) >> Dgate(x=x, y=y)
    assert np.allclose(state.cov, expected.cov)
    assert np.allclose(state.means, expected.means)


@given(r = st.floats(0.0, 1.0), phi = st.floats(0.0, 2 * np.pi))
def test_sq_state_is_same_as_sgate_on_vacuum(r, phi):
    state = SqueezedVacuum(r, phi)
    expected = Vacuum(1) >> Sgate(r=r, phi=phi)
    assert np.allclose(state.cov, expected.cov)
    assert np.allclose(state.means, expected.means)

@given(x=st.floats(-1.0, 1.0), y=st.floats(-1.0, 1.0), r = st.floats(0.0, 1.0), phi = st.floats(0.0, 2 * np.pi))
def test_dispsq_state_is_same_as_dsgate_on_vacuum(x, y, r, phi):
    state = DisplacedSqueezed(r,phi,x,y)
    expected = Vacuum(1) >> Sgate(r=r, phi=phi) >> Dgate(x=x, y=y)
    assert np.allclose(state.cov, expected.cov)
    assert np.allclose(state.means, expected.means)


def test_state_getitem():
    a = Gaussian(2)
    b = Gaussian(2)
    assert a == (a & b)[0, 1]
    assert b == (a & b)[2, 3]
