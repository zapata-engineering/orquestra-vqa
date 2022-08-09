################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
from openfermion import FermionOperator
from orquestra.quantum.circuits import Circuit

from orquestra.vqa.ansatz.singlet_uccsd import SingletUCCSDAnsatz
from orquestra.vqa.api.ansatz_test import AnsatzTests


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

    @pytest.fixture
    def raw_ccsd_fop(self):
        result_fop = FermionOperator()
        result_fop += -0.0252893416 * FermionOperator("4^ 0 5^ 1")
        result_fop += -0.0252893416 * FermionOperator("5^ 1 4^ 0")
        result_fop += -0.0142638609 * FermionOperator("6^ 0 7^ 1")
        result_fop += -0.0142638609 * FermionOperator("7^ 1 6^ 0")
        result_fop += -0.00821798305 * FermionOperator("2^ 0 3^ 1")
        result_fop += -0.00821798305 * FermionOperator("3^ 1 2^ 0")
        result_fop += -0.00783761365 * FermionOperator("2^ 0 7^ 1")
        result_fop += -0.00783761365 * FermionOperator("3^ 1 6^ 0")
        result_fop += -0.00783761365 * FermionOperator("6^ 0 3^ 1")
        result_fop += -0.00783761365 * FermionOperator("7^ 1 2^ 0")

        return result_fop

    @pytest.fixture
    def expected_mp2_based_guess(self):
        return np.array(
            [
                0.0,
                0.0,
                0.0,
                -0.00821798,
                -0.02528934,
                -0.01426386,
                0.0,
                -0.00783761,
                0.0,
            ]
        )

    def test_set_number_of_layers_invalidates_parametrized_circuit(self, ansatz):
        # This test overwrites test from base class AnsatzTests.
        pytest.xfail("Singlet UCCSD Ansatz can only have one layer")

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
            _ = SingletUCCSDAnsatz(
                number_of_layers=incorrect_number_of_layers,
                number_of_spatial_orbitals=number_of_spatial_orbitals,
                number_of_alpha_electrons=number_of_alpha_electrons,
                transformation=transformation,
            )

    def test_init_asserts_number_of_spatial_orbitals(
        self,
        number_of_layers,
        number_of_alpha_electrons,
        transformation,
    ):
        # Given
        incorrect_number_of_spatial_orbitals = 0

        # When/Then
        with pytest.raises(ValueError):
            _ = SingletUCCSDAnsatz(
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

    def test_set_number_of_spatial_orbitals_asserts_when_smaller_than_number_of_alpha_electrons(  # noqa: E501
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

    @pytest.mark.parametrize(
        "threshold,expected_new_size", [(0.0, 10), (1.0, 0), (0.025, 2)]
    )
    def test_screening_out_function(self, raw_ccsd_fop, threshold, expected_new_size):
        _, new_fop = SingletUCCSDAnsatz.screen_out_operator_terms_below_threshold(
            threshold, raw_ccsd_fop
        )

        assert len(new_fop.terms) == expected_new_size

    def test_computing_uccsd_vector_from_fermion_generator(
        self, raw_ccsd_fop, expected_mp2_based_guess
    ):
        ansatz = SingletUCCSDAnsatz(4, 1)

        np.testing.assert_array_almost_equal(
            ansatz.compute_uccsd_vector_from_fermion_generator(raw_ccsd_fop),
            expected_mp2_based_guess,
        )

    def test_generating_circuit_from_fermion_generator(
        self, raw_ccsd_fop, expected_mp2_based_guess
    ):
        ansatz = SingletUCCSDAnsatz(4, 1)

        circuit = ansatz.generate_circuit_from_fermion_generator(raw_ccsd_fop)
        assert isinstance(circuit, Circuit)

        # Let's create a simple mock here to check params
        ansatz.get_executable_circuit = lambda x: x

        passed_params = ansatz.generate_circuit_from_fermion_generator(raw_ccsd_fop)
        np.testing.assert_array_almost_equal(passed_params, expected_mp2_based_guess)
