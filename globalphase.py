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
r"""
Calculate the global phase by give a Gaussian operator.
"""
import tensorflow as tf
import numpy as np

def global_phase(C1, C2, mu1, mu2, Sigma1, Sigma2):
    """
    global phase when given two successive gaussian operator's C, mu, Sigma.
    """
    M1 = np.array([[0,0,1,0,0,0],[1,0,0,0,0,0]], dtype=np.complex128)
    M2 = np.array([[0,1,0,0,0,0],[0,0,0,0,1,0]], dtype=np.complex128)
    Z = np.array([[1,1j,0,0,0,0],
              [1,-1j,0,0,0,0],
              [0,0,1,1j,0,0],
              [0,0,1,-1j,0,0],
              [0,0,0,0,1,1j],
              [0,0,0,0,1,-1j]
             ], dtype=np.complex128)
    Y = np.array([[0,1,0,0,0,0],
              [1,0,0,0,0,0],
              [0,0,0,0,0,0],
              [0,0,0,0,0,0],
              [0,0,0,0,0,0],
              [0,0,0,0,0,0]
             ], dtype=np.complex128)
    b = tf.linalg.matvec(Z,tf.linalg.matvec(M1,mu1,transpose_a=True) + tf.linalg.matvec(M2,mu2,transpose_a=True),transpose_a=True)
    A = tf.tensordot(tf.tensordot(tf.transpose(Z), Y + tf.tensordot(tf.tensordot(tf.transpose(M2), Sigma2, axes = 1), M2, axes = 1) + tf.tensordot(tf.tensordot(tf.transpose(M1), Sigma1, axes = 1), M1, axes = 1), axes = 1), Z, axes = 1)
    
    bl = b[:2]
    A1 = A[:2, :2]
    C_new = C1*C2*2/tf.sqrt(tf.linalg.det(A1))*np.exp(bl[None,:]@np.linalg.inv(A1)@bl[:,None])
    return C_new
