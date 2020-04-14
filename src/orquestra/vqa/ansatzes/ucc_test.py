import unittest
from ..utils import build_circuit_template
from zquantum.core.circuit import generate_random_ansatz_params, build_ansatz_circuit
import numpy as np
import os

class TestUCCSDUtils(unittest.TestCase):

    def test_build_singlet_uccsd_circuit(self):
        # Given
        n_alpha = 1
        n_beta = 1
        n_mo = 2
        # When
        ansatz = build_circuit_template('singlet UCCSD', n_mo, n_alpha, n_beta, transformation='Jordan-Wigner')
        params = generate_random_ansatz_params(ansatz, -1.0, 1.0)
        qprog = build_ansatz_circuit(ansatz, params)
        # Then
        self.assertEqual(len(qprog.get_qubits()), 4)

    def test_exponentiate_fermion_operator(self):
        pass
