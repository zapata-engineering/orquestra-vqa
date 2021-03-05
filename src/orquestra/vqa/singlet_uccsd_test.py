import pytest
import numpy as np
from zquantum.core.interfaces.ansatz_test import AnsatzTests
from zquantum.core.circuit import Circuit
from .singlet_uccsd import SingletUCCSDAnsatz


class TestSingletUCCSDAnsatz(AnsatzTests):
    @pytest.fixture
    def number_of_layers(self):
        return 1

    @pytest.fixture
    def number_of_spatial_orbitals(self):
        return 3

    @pytest.fixture
    def number_of_alpha_electrons(self):
        return 1

    @pytest.fixture
    def transformation(self):
        return "Jordan-Wigner"

    @pytest.fixture
    def ansatz(
        self,
        number_of_layers,
        number_of_spatial_orbitals,
        number_of_alpha_electrons,
        transformation,
    ):
        return SingletUCCSDAnsatz(
            number_of_layers=number_of_layers,
            number_of_spatial_orbitals=number_of_spatial_orbitals,
            number_of_alpha_electrons=number_of_alpha_electrons,
            transformation=transformation,
        )

    def test_init_asserts_number_of_layers(
        self,
        ansatz,
        number_of_spatial_orbitals,
        number_of_alpha_electrons,
        transformation,
    ):
        # Given
        incorrect_number_of_layers = 0

        # When/Then
        with pytest.raises(ValueError):
            ansatz = SingletUCCSDAnsatz(
                number_of_layers=incorrect_number_of_layers,
                number_of_spatial_orbitals=number_of_spatial_orbitals,
                number_of_alpha_electrons=number_of_alpha_electrons,
                transformation=transformation,
            )

    def test_init_asserts_number_of_spatial_orbitals(
        self,
        ansatz,
        number_of_layers,
        number_of_alpha_electrons,
        transformation,
    ):
        # Given
        incorrect_number_of_spatial_orbitals = 0

        # When/Then
        with pytest.raises(ValueError):
            ansatz = SingletUCCSDAnsatz(
                number_of_layers=number_of_layers,
                number_of_spatial_orbitals=incorrect_number_of_spatial_orbitals,
                number_of_alpha_electrons=number_of_alpha_electrons,
                transformation=transformation,
            )

    def test_set_number_of_layers(self, ansatz):
        # Given
        new_number_of_layers = 100

        # When/Then
        with pytest.raises(ValueError):
            ansatz.number_of_layers = new_number_of_layers

    def test_set_number_of_alpha_electrons(self, ansatz):
        # Given
        new_number_of_alpha_electrons = 2

        # When
        ansatz.number_of_alpha_electrons = new_number_of_alpha_electrons

        # Then
        assert ansatz.number_of_alpha_electrons == new_number_of_alpha_electrons

    def test_set_number_of_alpha_electrons_asserts_number_of_spatial_orbitals(
        self, ansatz
    ):
        # Given
        new_number_of_alpha_electrons = 3

        # When/Then
        with pytest.raises(ValueError):
            ansatz.number_of_alpha_electrons = new_number_of_alpha_electrons

    def test_get_number_of_beta_electrons(self, ansatz):
        # Given
        new_number_of_alpha_electrons = 2

        # When
        ansatz.number_of_alpha_electrons = new_number_of_alpha_electrons

        # Then
        assert ansatz._number_of_beta_electrons == new_number_of_alpha_electrons

    def test_get_number_of_qubits(self, ansatz):
        # Given
        new_number_of_spatial_orbitals = 4
        target_number_of_qubits = 8

        # When
        ansatz.number_of_spatial_orbitals = new_number_of_spatial_orbitals

        # Then
        assert ansatz.number_of_qubits == target_number_of_qubits

    def test_set_number_of_spatial_orbitals(self, ansatz):
        # Given
        new_number_of_spatial_orbitals = 4

        # When
        ansatz.number_of_spatial_orbitals = new_number_of_spatial_orbitals

        # Then
        assert ansatz._number_of_spatial_orbitals == new_number_of_spatial_orbitals

    def test_set_number_of_spatial_orbitals_asserts_when_smaller_than_2(self, ansatz):
        # Given
        new_number_of_spatial_orbitals = 1

        # When/Then
        with pytest.raises(ValueError):
            ansatz.number_of_spatial_orbitals = new_number_of_spatial_orbitals

    def test_set_number_of_spatial_orbitals_asserts_when_smaller_than_number_of_alpha_electrons(
        self, ansatz
    ):
        # Given
        ansatz.number_of_spatial_orbitals = 4
        ansatz.number_of_alpha_electrons = 3
        new_number_of_spatial_orbitals = 3

        # When/Then
        with pytest.raises(ValueError):
            ansatz.number_of_spatial_orbitals = new_number_of_spatial_orbitals

    def test_set_transformation(self, ansatz):
        # Given
        new_transformation = "transformation"

        # When
        ansatz.transformation = new_transformation

        # Then
        assert ansatz.transformation == new_transformation

    def test_does_not_support_parametrized_circuit(self, ansatz):
        # When/Then
        assert ansatz.supports_parametrized_circuits == False
        with pytest.raises(NotImplementedError):
            parametrized_circuit = ansatz.parametrized_circuit
