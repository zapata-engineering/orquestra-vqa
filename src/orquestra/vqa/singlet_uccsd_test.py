from zquantum.core.interfaces.ansatz_test import AnsatzTests
from zquantum.core.circuit import Circuit
from .singlet_uccsd import SingletUCCSDAnsatz
import unittest
import numpy as np


class TestSingletUCCSDAnsatz(unittest.TestCase, AnsatzTests):
    def setUp(self):
        self.n_alpha = 1
        self.n_spatial_orbitals = 3
        self.n_layers = 1
        self.transformation = "Jordan-Wigner"

        self.ansatz = SingletUCCSDAnsatz(
            n_layers=self.n_layers,
            n_spatial_orbitals=self.n_spatial_orbitals,
            n_alpha=self.n_alpha,
            transformation=self.transformation,
        )

    def test_init_asserts_n_spatial_orbitals(self):
        # Given
        incorrect_n_spatial_orbitals = 0

        # When/Then
        with self.assertRaises(ValueError):
            self.ansatz = SingletUCCSDAnsatz(
                n_layers=self.n_layers,
                n_spatial_orbitals=incorrect_n_spatial_orbitals,
                n_alpha=self.n_alpha,
                transformation=self.transformation,
            )

    def test_set_n_alpha(self):
        # Given
        new_n_alpha = 2

        # When
        self.ansatz.n_alpha = new_n_alpha

        # Then
        self.assertEqual(self.ansatz._n_alpha, new_n_alpha)

    def test_set_n_alpha_asserts_n_spatial_orbitals(self):
        # Given
        new_n_alpha = 3

        # When/Then
        with self.assertRaises(ValueError):
            self.ansatz.n_alpha = new_n_alpha

    def test_get_n_beta(self):
        # Given
        new_n_alpha = 2

        # When
        self.ansatz.n_alpha = new_n_alpha

        # Then
        self.assertEqual(self.ansatz._n_beta, new_n_alpha)

    def test_get_n_qubits(self):
        # Given
        new_n_spatial_orbitals = 4
        target_n_qubits = 8

        # When
        self.ansatz.n_spatial_orbitals = new_n_spatial_orbitals

        # Then
        self.assertEqual(self.ansatz.n_qubits, target_n_qubits)

    def test_set_n_spatial_orbitals(self):
        # Given
        new_n_spatial_orbitals = 4

        # When
        self.ansatz.n_spatial_orbitals = new_n_spatial_orbitals

        # Then
        self.assertEqual(self.ansatz._n_spatial_orbitals, new_n_spatial_orbitals)

    def test_set_n_spatial_orbitals_asserts_when_smaller_than_2(self):
        # Given
        new_n_spatial_orbitals = 1

        # When/Then
        with self.assertRaises(ValueError):
            self.ansatz.n_spatial_orbitals = new_n_spatial_orbitals

    def test_set_n_spatial_orbitals_asserts_when_smaller_than_n_alpha(self):
        # Given
        self.ansatz.n_spatial_orbitals = 4
        self.ansatz.n_alpha = 3
        new_n_spatial_orbitals = 3

        # When/Then
        with self.assertRaises(ValueError):
            self.ansatz.n_spatial_orbitals = new_n_spatial_orbitals

    def test_set_transformation(self):
        # Given
        new_transformation = "transformation"

        # When
        self.ansatz.transformation = new_transformation

        # Then
        self.assertEqual(self.ansatz._transformation, new_transformation)

    def test_does_not_support_parametrized_circuit(self):
        # When/Then
        self.assertFalse(self.ansatz.supports_parametrized_circuits)
        with self.assertRaises(Exception):
            parametrized_circuit = self.ansatz.parametrized_circuit
