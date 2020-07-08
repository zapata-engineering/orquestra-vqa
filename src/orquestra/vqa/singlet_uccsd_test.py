from zquantum.core.interfaces.ansatz_test import AnsatzTests
from zquantum.core.circuit import Circuit
from .singlet_uccsd import SingletUCCSDAnsatz
import unittest
import numpy as np


class TestSingletUCCSDAnsatz(unittest.TestCase, AnsatzTests):
    def setUp(self):
        self.number_of_layers = 1
        self.number_of_alpha_electrons = 1
        self.number_of_spatial_orbitals = 3
        self.transformation = "Jordan-Wigner"

        self.ansatz = SingletUCCSDAnsatz(
            number_of_layers=self.number_of_layers,
            number_of_spatial_orbitals=self.number_of_spatial_orbitals,
            number_of_alpha_electrons=self.number_of_alpha_electrons,
            transformation=self.transformation,
        )

    def test_init_asserts_number_of_layers(self):
        # Given
        incorrect_number_of_layers = 0

        # When/Then
        with self.assertRaises(ValueError):
            self.ansatz = SingletUCCSDAnsatz(
                number_of_layers=incorrect_number_of_layers,
                number_of_spatial_orbitals=self.number_of_spatial_orbitals,
                number_of_alpha_electrons=self.number_of_alpha_electrons,
                transformation=self.transformation,
            )

    def test_init_asserts_number_of_spatial_orbitals(self):
        # Given
        incorrect_number_of_spatial_orbitals = 0

        # When/Then
        with self.assertRaises(ValueError):
            self.ansatz = SingletUCCSDAnsatz(
                number_of_layers=self.number_of_layers,
                number_of_spatial_orbitals=incorrect_number_of_spatial_orbitals,
                number_of_alpha_electrons=self.number_of_alpha_electrons,
                transformation=self.transformation,
            )

    def test_set_number_of_layers(self):
        # Given
        new_number_of_layers = 100

        # When/Then
        with self.assertRaises(ValueError):
            self.ansatz.number_of_layers = new_number_of_layers

    def test_set_number_of_alpha_electrons(self):
        # Given
        new_number_of_alpha_electrons = 2

        # When
        self.ansatz.number_of_alpha_electrons = new_number_of_alpha_electrons

        # Then
        self.assertEqual(
            self.ansatz.number_of_alpha_electrons, new_number_of_alpha_electrons
        )

    def test_set_number_of_alpha_electrons_asserts_number_of_spatial_orbitals(self):
        # Given
        new_number_of_alpha_electrons = 3

        # When/Then
        with self.assertRaises(ValueError):
            self.ansatz.number_of_alpha_electrons = new_number_of_alpha_electrons

    def test_get_number_of_beta_electrons(self):
        # Given
        new_number_of_alpha_electrons = 2

        # When
        self.ansatz.number_of_alpha_electrons = new_number_of_alpha_electrons

        # Then
        self.assertEqual(
            self.ansatz._number_of_beta_electrons, new_number_of_alpha_electrons
        )

    def test_get_number_of_qubits(self):
        # Given
        new_number_of_spatial_orbitals = 4
        target_number_of_qubits = 8

        # When
        self.ansatz.number_of_spatial_orbitals = new_number_of_spatial_orbitals

        # Then
        self.assertEqual(self.ansatz.number_of_qubits, target_number_of_qubits)

    def test_set_number_of_spatial_orbitals(self):
        # Given
        new_number_of_spatial_orbitals = 4

        # When
        self.ansatz.number_of_spatial_orbitals = new_number_of_spatial_orbitals

        # Then
        self.assertEqual(
            self.ansatz._number_of_spatial_orbitals, new_number_of_spatial_orbitals
        )

    def test_set_number_of_spatial_orbitals_asserts_when_smaller_than_2(self):
        # Given
        new_number_of_spatial_orbitals = 1

        # When/Then
        with self.assertRaises(ValueError):
            self.ansatz.number_of_spatial_orbitals = new_number_of_spatial_orbitals

    def test_set_number_of_spatial_orbitals_asserts_when_smaller_than_number_of_alpha_electrons(
        self,
    ):
        # Given
        self.ansatz.number_of_spatial_orbitals = 4
        self.ansatz.number_of_alpha_electrons = 3
        new_number_of_spatial_orbitals = 3

        # When/Then
        with self.assertRaises(ValueError):
            self.ansatz.number_of_spatial_orbitals = new_number_of_spatial_orbitals

    def test_set_transformation(self):
        # Given
        new_transformation = "transformation"

        # When
        self.ansatz.transformation = new_transformation

        # Then
        self.assertEqual(self.ansatz.transformation, new_transformation)

    def test_does_not_support_parametrized_circuit(self):
        # When/Then
        self.assertFalse(self.ansatz.supports_parametrized_circuits)
        with self.assertRaises(Exception):
            parametrized_circuit = self.ansatz.parametrized_circuit
