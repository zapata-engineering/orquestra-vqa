################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import pytest

from orquestra.vqa.ansatz import HEAQuantumCompilingAnsatz
from orquestra.vqa.api.ansatz_test import AnsatzTests


class TestHEAQuantumCompilingAnsatz(AnsatzTests):
    @pytest.fixture(
        params=[
            1,
            2,
            3,
        ]
    )
    def number_of_layers(self, request):
        return request.param

    @pytest.fixture(
        params=[
            2,
            4,
            6,
            8,
        ]
    )
    def number_of_qubits(self, request):
        return request.param

    @pytest.fixture
    def ansatz(
        self,
        number_of_layers,
        number_of_qubits,
    ):
        return HEAQuantumCompilingAnsatz(
            number_of_layers=number_of_layers,
            number_of_qubits=number_of_qubits,
        )

    def test_init_asserts_number_of_layers(
        self,
        number_of_qubits,
    ):
        # Given
        incorrect_number_of_layers = 0

        # When/Then
        with pytest.raises(ValueError):
            _ = HEAQuantumCompilingAnsatz(
                number_of_layers=incorrect_number_of_layers,
                number_of_qubits=number_of_qubits,
            )

    @pytest.mark.parametrize("odd_number_of_qubits", [1, 3, 5, 7, 9])
    def test_init_asserts_number_of_qubits_is_even(
        self, number_of_layers, odd_number_of_qubits
    ):
        # When/Then
        with pytest.raises(AssertionError):
            _ = HEAQuantumCompilingAnsatz(
                number_of_layers=number_of_layers,
                number_of_qubits=odd_number_of_qubits,
            )

    def test_set_number_of_qubits(self, ansatz):
        # Given
        new_number_of_qubits = 14

        # When
        ansatz.number_of_qubits = new_number_of_qubits

        # Then
        assert ansatz.number_of_qubits == new_number_of_qubits

    def test_supports_parametrized_circuit(self, ansatz):
        # When/Then
        assert ansatz.supports_parametrized_circuits

    def test_circuit_built_properly(self, ansatz):
        expected_number_of_single_qubit_gates = (
            ansatz.number_of_layers * ansatz.number_of_qubits * 5 * 2
        )
        expected_number_of_two_qubit_gates = (
            ansatz.number_of_layers * int(ansatz.number_of_qubits / 2) * 2
        )

        assert (
            len(ansatz.parametrized_circuit.operations)
            == expected_number_of_single_qubit_gates
            + expected_number_of_two_qubit_gates
        )
