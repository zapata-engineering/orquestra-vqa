import os
import unittest
from .utils import build_circuit_template

class TestGeneralPurpose(unittest.TestCase):

    def test_build_circuit_template(self):
        # Singlet H2O
        n_alpha = 4
        n_beta = 4
        n_mo = 12
        ansatz = build_circuit_template('singlet UCCSD', n_mo, n_alpha, n_beta, transformation='Jordan-Wigner')
        self.assertEqual(sum(ansatz['n_params']), 560)

        def build_invalid_ansatz():
            build_circuit_template('singlet UCCSD', n_mo, n_alpha-1, n_beta+1, transformation='Jordan-Wigner')

        self.assertRaises(RuntimeError, build_invalid_ansatz)