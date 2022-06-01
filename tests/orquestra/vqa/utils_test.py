################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from orquestra.quantum.circuits import Circuit, X

from orquestra.vqa.utils import build_hartree_fock_circuit


class TestVQEUtils:
    def test_build_hartree_fock_circuit_jordan_wigner(self):
        number_of_qubits = 4
        number_of_alpha_electrons = 1
        number_of_beta_electrons = 1
        transformation = "Jordan-Wigner"
        expected_circuit = Circuit([X(0), X(1)], n_qubits=number_of_qubits)
        actual_circuit = build_hartree_fock_circuit(
            number_of_qubits,
            number_of_alpha_electrons,
            number_of_beta_electrons,
            transformation,
        )
        assert actual_circuit == expected_circuit

    def test_build_hartree_fock_circuit_bravyi_kitaev(self):
        number_of_qubits = 4
        number_of_alpha_electrons = 1
        number_of_beta_electrons = 1
        transformation = "Bravyi-Kitaev"
        expected_circuit = Circuit([X(0)], n_qubits=number_of_qubits)
        actual_circuit = build_hartree_fock_circuit(
            number_of_qubits,
            number_of_alpha_electrons,
            number_of_beta_electrons,
            transformation,
        )
        assert actual_circuit == expected_circuit
