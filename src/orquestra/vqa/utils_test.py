import os
import unittest
from .utils import build_circuit_template
# import numpy as np
# import copy
# from openfermion.ops import FermionOperator, InteractionOperator, QubitOperator
# from openfermion.utils import hermitian_conjugated
# from openfermion.transforms import jordan_wigner, get_interaction_operator
# from pyquil.wavefunction import Wavefunction
# from zmachine.algorithms.vqe._api import (build_circuit_template, get_objective_function_from_orbital_frames,
#                                          get_interaction_rdm_from_expectations, optimize_marginals, create_full_fermion_hamiltonian)
# from zmachine.core.fermion import get_orbital_frames_decomposition, load_interaction_operator, load_interaction_rdm, transform_fermion_hamiltonian, hf_rdm
# from zmachine.core.measurement import expectation

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