################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import unittest

import numpy as np
import pkg_resources
import pytest
from openfermion import fermi_hubbard
from openfermion.linalg import (
    get_sparse_operator,
    jw_get_ground_state_at_particle_number,
)
from openfermion.ops import FermionOperator
from openfermion.transforms import (
    get_fermion_operator,
    get_interaction_operator,
    jordan_wigner,
)

from orquestra.vqa.openfermion import load_interaction_operator


class TestFermionOperator(unittest.TestCase):
    def test_get_fermion_number_operator(self):
        from orquestra.vqa.openfermion import get_fermion_number_operator

        # Given
        n_qubits = 4
        n_particles = None
        correct_operator = get_interaction_operator(
            FermionOperator(
                """
        0.0 [] +
        1.0 [0^ 0] +
        1.0 [1^ 1] +
        1.0 [2^ 2] +
        1.0 [3^ 3]
        """
            )
        )

        # When
        number_operator = get_fermion_number_operator(n_qubits)

        # Then
        self.assertEqual(number_operator, correct_operator)

        # Given
        n_qubits = 4
        n_particles = 2
        correct_operator = get_interaction_operator(
            FermionOperator(
                """
        -2.0 [] +
        1.0 [0^ 0] +
        1.0 [1^ 1] +
        1.0 [2^ 2] +
        1.0 [3^ 3]
        """
            )
        )

        # When
        number_operator = get_fermion_number_operator(n_qubits, n_particles)

        # Then
        self.assertEqual(number_operator, correct_operator)


class TestOtherUtils(unittest.TestCase):
    def test_get_diagonal_component_polynomial_tensor(self):
        from orquestra.vqa.openfermion import (
            get_diagonal_component,
            get_polynomial_tensor,
        )

        fermion_op = FermionOperator("0^ 1^ 2^ 0 1 2", 1.0)
        fermion_op += FermionOperator("0^ 1^ 2^ 0 1 3", 2.0)
        fermion_op += FermionOperator((), 3.0)
        polynomial_tensor = get_polynomial_tensor(fermion_op)
        diagonal_op, remainder_op = get_diagonal_component(polynomial_tensor)
        self.assertTrue((diagonal_op + remainder_op) == polynomial_tensor)
        diagonal_qubit_op = jordan_wigner(get_fermion_operator(diagonal_op))
        remainder_qubit_op = jordan_wigner(get_fermion_operator(remainder_op))
        for term in diagonal_qubit_op.terms:
            for pauli in term:
                self.assertTrue(pauli[1] == "Z")
        for term in remainder_qubit_op.terms:
            is_diagonal = True
            for pauli in term:
                if pauli[1] != "Z":
                    is_diagonal = False
                    break
            self.assertFalse(is_diagonal)

    def test_get_diagonal_component_interaction_op(self):
        from orquestra.vqa.openfermion import get_diagonal_component

        fermion_op = FermionOperator("1^ 1", 0.5)
        fermion_op += FermionOperator("2^ 2", 0.5)
        fermion_op += FermionOperator("1^ 2^ 0 3", 0.5)
        diagonal_op, remainder_op = get_diagonal_component(
            get_interaction_operator(fermion_op)
        )
        self.assertTrue(
            (diagonal_op + remainder_op) == get_interaction_operator(fermion_op)
        )
        diagonal_qubit_op = jordan_wigner(diagonal_op)
        remainder_qubit_op = jordan_wigner(remainder_op)
        for term in diagonal_qubit_op.terms:
            for pauli in term:
                self.assertTrue(pauli[1] == "Z")
        is_diagonal = True
        for term in remainder_qubit_op.terms:
            for pauli in term:
                if pauli[1] != "Z":
                    is_diagonal = False
                    break
        self.assertFalse(is_diagonal)

    def test_get_ground_state_rdm_from_qubit_op(self):
        from orquestra.vqa.openfermion import get_ground_state_rdm_from_qubit_op

        # Given
        n_sites = 2
        U = 5.0
        fhm = fermi_hubbard(
            x_dimension=n_sites,
            y_dimension=1,
            tunneling=1.0,
            coulomb=U,
            chemical_potential=U / 2,
            magnetic_field=0,
            periodic=False,
            spinless=False,
            particle_hole_symmetry=False,
        )
        fhm_qubit = jordan_wigner(fhm)
        fhm_int = get_interaction_operator(fhm)
        e, wf = jw_get_ground_state_at_particle_number(
            get_sparse_operator(fhm), n_sites
        )

        # When
        rdm = get_ground_state_rdm_from_qubit_op(
            qubit_operator=fhm_qubit, n_particles=n_sites
        )

        # Then
        self.assertAlmostEqual(e, rdm.expectation(fhm_int))

    def test_remove_inactive_orbitals(self):
        from orquestra.vqa.openfermion import hf_rdm, remove_inactive_orbitals

        fermion_ham = load_interaction_operator(
            pkg_resources.resource_filename(
                "orquestra.quantum.testing", "hamiltonian_HeH_plus_STO-3G.json"
            )
        )
        frozen_ham = remove_inactive_orbitals(fermion_ham, 1, 1)
        self.assertEqual(frozen_ham.one_body_tensor.shape[0], 2)

        hf_energy = hf_rdm(1, 1, 2).expectation(fermion_ham)
        self.assertAlmostEqual(frozen_ham.constant, hf_energy)


# Hamiltonians and energies from Psi4 H2 minimal basis
# first one is RHF, second one is H2- doublet with ROHF
@pytest.mark.parametrize(
    "hamiltonian, ref_energy, nalpha",
    [
        (
            load_interaction_operator(
                pkg_resources.resource_filename(
                    "orquestra.quantum.testing", "hamiltonian_H2_minimal_basis.json"
                )
            ),
            -0.8543376267387818,
            1,
        ),
        (
            load_interaction_operator(
                pkg_resources.resource_filename(
                    "orquestra.quantum.testing",
                    "hamiltonian_H2_minus_ROHF_minimal_basis.json",
                )
            ),
            -0.6857403043904364,
            2,
        ),
    ],
)
def test_hf_rdm_energy(hamiltonian, ref_energy, nalpha):
    from orquestra.vqa.openfermion import hf_rdm

    rdm = hf_rdm(nalpha, 1, 2)
    assert np.isclose(ref_energy, rdm.expectation(hamiltonian))
