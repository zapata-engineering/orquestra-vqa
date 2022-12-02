################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import pytest
import sympy
from orquestra.quantum.circuits import RX, RZ, Circuit, H
from orquestra.quantum.operators import PauliSum, PauliTerm
from orquestra.quantum.utils import compare_unitary

from orquestra.vqa.ansatz import (
    QAOAFarhiAnsatz,
    create_all_x_mixer_hamiltonian,
    create_farhi_qaoa_circuits,
)
from orquestra.vqa.api.ansatz_test import AnsatzTests


class TestQAOAFarhiAnsatz(AnsatzTests):
    @pytest.fixture
    def ansatz(self):
        cost_hamiltonian = PauliSum("Z0+Z1")
        mixer_hamiltonian = PauliSum("X0+X1")
        return QAOAFarhiAnsatz(
            number_of_layers=1,
            cost_hamiltonian=cost_hamiltonian,
            mixer_hamiltonian=mixer_hamiltonian,
        )

    @pytest.fixture
    def beta(self):
        return sympy.Symbol("beta_0")

    @pytest.fixture
    def gamma(self):
        return sympy.Symbol("gamma_0")

    @pytest.fixture
    def symbols_map(self, beta, gamma):
        return {beta: 0.5, gamma: 0.7}

    @pytest.fixture
    def target_unitary(self, beta, gamma, symbols_map):
        target_circuit = Circuit()
        target_circuit += H(0)
        target_circuit += H(1)
        target_circuit += RZ(2 * gamma)(0)
        target_circuit += RZ(2 * gamma)(1)
        target_circuit += RX(2 * beta)(0)
        target_circuit += RX(2 * beta)(1)
        return target_circuit.bind(symbols_map).to_unitary()

    def test_set_cost_hamiltonian(self, ansatz):
        # Given
        new_cost_hamiltonian = PauliSum("Z0 + -1*Z1")

        # When
        ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        assert ansatz._cost_hamiltonian == new_cost_hamiltonian

    def test_set_cost_hamiltonian_invalidates_circuit(self, ansatz):
        # Given
        new_cost_hamiltonian = PauliSum("Z0 + -1*Z1")

        # When
        ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        assert ansatz._parametrized_circuit is None

    def test_set_mixer_hamiltonian(self, ansatz):
        # Given
        new_mixer_hamiltonian = PauliSum("X0 + -1*X1")

        # When
        ansatz.mixer_hamiltonian = new_mixer_hamiltonian

        # Then
        ansatz._mixer_hamiltonian == new_mixer_hamiltonian

    def test_set_mixer_hamiltonian_invalidates_circuit(self, ansatz):
        # Given
        new_mixer_hamiltonian = PauliSum("X0 + -1*X1")

        # When
        ansatz.mixer_hamiltonian = new_mixer_hamiltonian

        # Then
        assert ansatz._parametrized_circuit is None

    def test_get_number_of_qubits(self, ansatz):
        # Given
        new_cost_hamiltonian = PauliSum("Z0+Z1+Z2")
        target_number_of_qubits = 3

        # When
        ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        assert ansatz.number_of_qubits == target_number_of_qubits

    def test_get_number_of_qubits_with_pauli_term(self, ansatz):
        # Given
        new_cost_hamiltonian = PauliTerm("Z0*Z1*Z2")
        target_number_of_qubits = 3

        # When
        ansatz.cost_hamiltonian = new_cost_hamiltonian

        # Then
        assert ansatz.number_of_qubits == target_number_of_qubits

    def test_get_parametrizable_circuit(self, ansatz, beta, gamma):
        # Then
        assert ansatz.parametrized_circuit.free_symbols == [gamma, beta]

    def test_generate_circuit(self, ansatz, symbols_map, target_unitary):
        # When
        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        # Then
        assert compare_unitary(final_unitary, target_unitary, tol=1e-10)

    def test_generate_circuit_with_pauli_term(self, ansatz, beta, gamma, symbols_map):
        # Given
        target_1_qubit_circuit = Circuit()
        target_1_qubit_circuit += H(0)
        target_1_qubit_circuit += RZ(2 * gamma)(0)
        target_1_qubit_circuit += RX(2 * beta)(0)
        target_1_qubit_unitary = target_1_qubit_circuit.bind(symbols_map).to_unitary()

        # When
        ansatz.cost_hamiltonian = PauliTerm("Z0")
        ansatz.mixer_hamiltonian = PauliTerm("X0")

        parametrized_circuit = ansatz._generate_circuit()
        evaluated_circuit = parametrized_circuit.bind(symbols_map)
        final_unitary = evaluated_circuit.to_unitary()

        # Then
        assert compare_unitary(final_unitary, target_1_qubit_unitary, tol=1e-10)


def test_create_farhi_qaoa_circuits():
    # Given
    hamiltonians = [PauliTerm("Z0*Z1"), PauliSum("Z0+Z1")]
    number_of_layers = 2

    # When
    circuits = create_farhi_qaoa_circuits(hamiltonians, number_of_layers)

    # Then
    assert len(circuits) == len(hamiltonians)

    for circuit in circuits:
        assert isinstance(circuit, Circuit)


def test_create_farhi_qaoa_circuits_when_number_of_layers_is_list():
    # Given
    hamiltonians = [PauliTerm("Z0*Z1"), PauliSum("Z0+Z1")]
    number_of_layers = [2, 3]

    # When
    circuits = create_farhi_qaoa_circuits(hamiltonians, number_of_layers)

    # Then
    assert len(circuits) == len(hamiltonians)

    for circuit in circuits:
        assert isinstance(circuit, Circuit)


def test_create_farhi_qaoa_circuits_fails_when_length_of_inputs_is_not_equal():
    # Given
    hamiltonians = [PauliTerm("Z0*Z1"), PauliSum("Z0+Z1")]
    number_of_layers = [2]

    # When
    with pytest.raises(AssertionError):
        create_farhi_qaoa_circuits(hamiltonians, number_of_layers)


def test_create_all_x_mixer_hamiltonian():
    # Given
    number_of_qubits = 4
    target_operator = (
        PauliTerm("X0") + PauliTerm("X1") + PauliTerm("X2") + PauliTerm("X3")
    )

    # When
    operator = create_all_x_mixer_hamiltonian(number_of_qubits)

    # Then
    assert operator == target_operator
