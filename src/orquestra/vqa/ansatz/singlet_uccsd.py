################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
from typing import Optional, Tuple

import numpy as np
import sympy
from openfermion import (
    FermionOperator,
    uccsd_singlet_generator,
    uccsd_singlet_paramsize,
)
from orquestra.quantum.circuits import Circuit
from overrides import overrides

from ..api.ansatz import Ansatz
from ..api.ansatz_utils import ansatz_property, invalidates_parametrized_circuit
from ..openfermion import build_hartree_fock_circuit, exponentiate_fermion_operator


class SingletUCCSDAnsatz(Ansatz):

    supports_parametrized_circuits = True
    transformation = ansatz_property("transformation")

    def __init__(
        self,
        number_of_spatial_orbitals: int,
        number_of_alpha_electrons: int,
        number_of_layers: int = 1,
        transformation: str = "Jordan-Wigner",
    ):
        """
        Ansatz class representing Singlet UCCSD Ansatz.

        Args:
            number_of_layers: number of layers of the ansatz. Since it's
                a UCCSD Ansatz, it can only be equal to 1.
            number_of_spatial_orbitals: number of spatial orbitals.
            number_of_alpha_electrons: number of alpha electrons.
            transformation: transformation used for translation between fermions
                and qubits.

        Attributes:
            number_of_beta_electrons: number of beta electrons
                (equal to number_of_alpha_electrons).
            number_of_electrons: total number of electrons (number_of_alpha_electrons
                + number_of_beta_electrons).
            number_of_qubits: number of qubits required for the ansatz circuit.
            number_of_params: number of the parameters that need to be set for
                the ansatz circuit.
        """
        super().__init__(number_of_layers=number_of_layers)
        self._number_of_layers = number_of_layers
        self._assert_number_of_layers()
        self._number_of_spatial_orbitals = number_of_spatial_orbitals
        self._number_of_alpha_electrons = number_of_alpha_electrons
        self._transformation = transformation
        self._assert_number_of_spatial_orbitals()

    @property
    def number_of_layers(self):
        return self._number_of_layers

    @invalidates_parametrized_circuit  # type: ignore
    @number_of_layers.setter
    def number_of_layers(self, new_number_of_layers):
        self._number_of_layers = new_number_of_layers
        self._assert_number_of_layers()

    @property
    def number_of_spatial_orbitals(self):
        return self._number_of_spatial_orbitals

    @invalidates_parametrized_circuit  # type: ignore
    @number_of_spatial_orbitals.setter
    def number_of_spatial_orbitals(self, new_number_of_spatial_orbitals):
        self._number_of_spatial_orbitals = new_number_of_spatial_orbitals
        self._assert_number_of_spatial_orbitals()

    @property
    def number_of_qubits(self):
        return self._number_of_spatial_orbitals * 2

    @property
    def number_of_alpha_electrons(self):
        return self._number_of_alpha_electrons

    @invalidates_parametrized_circuit  # type: ignore
    @number_of_alpha_electrons.setter
    def number_of_alpha_electrons(self, new_number_of_alpha_electrons):
        self._number_of_alpha_electrons = new_number_of_alpha_electrons
        self._assert_number_of_spatial_orbitals()

    @property
    def _number_of_beta_electrons(self):
        return self._number_of_alpha_electrons

    @property
    def number_of_electrons(self):
        return self._number_of_alpha_electrons + self._number_of_beta_electrons

    @property
    def number_of_params(self) -> int:
        """
        Returns number of parameters in the ansatz.
        """
        return uccsd_singlet_paramsize(
            n_qubits=self.number_of_qubits,
            n_electrons=self.number_of_electrons,
        )

    @staticmethod
    def screen_out_operator_terms_below_threshold(
        threshold: float, fermion_generator: FermionOperator, ignore_singles=False
    ) -> Tuple[np.ndarray, FermionOperator]:
        """Screen single and double excitation operators based on a guess
            for the amplitudes

        Args:
            threshold (float): threshold to select excitations. Only those with
                absolute amplitudes above the threshold are kept.
            fermion_generator (openfermion.FermionOperator): Fermion Operator
                containing the generators for the UCC ansatz
        Returns:
            amplitudes (np.array): screened amplitudes
            new_fermion_generator (openfermion.FermionOperator): screened
            Fermion Operator
        """

        new_fermion_generator = FermionOperator()
        amplitudes = []
        for op in fermion_generator.terms:
            if abs(fermion_generator.terms[op]) > threshold or (
                len(op) == 2 and ignore_singles
            ):
                new_fermion_generator += FermionOperator(
                    op, fermion_generator.terms[op]
                )
                amplitudes.append(fermion_generator.terms[op])
        amplitudes_array = np.asarray(amplitudes)
        return amplitudes_array, new_fermion_generator

    def compute_uccsd_vector_from_fermion_generator(
        self, raw_fermion_generator: FermionOperator, screening_threshold: float = 0.0
    ) -> np.ndarray:
        _, screened_mp2_operator = self.screen_out_operator_terms_below_threshold(
            screening_threshold, raw_fermion_generator
        )

        ansatz_operator = uccsd_singlet_generator(
            np.arange(1.0, self.number_of_params + 1),
            2 * self.number_of_spatial_orbitals,
            self.number_of_electrons,
            anti_hermitian=True,
        )

        params_vector = np.zeros(self.number_of_params)

        for term, coeff in screened_mp2_operator.terms.items():
            if term in ansatz_operator.terms.keys():
                params_vector[int(ansatz_operator.terms[term]) - 1] = coeff

        return params_vector

    def generate_circuit_from_fermion_generator(
        self, raw_fermion_generator: FermionOperator, screening_threshold: float = 0.0
    ) -> Circuit:
        params_vector = self.compute_uccsd_vector_from_fermion_generator(
            raw_fermion_generator, screening_threshold
        )

        return self.get_executable_circuit(params_vector)

    @overrides
    def _generate_circuit(self, params: Optional[np.ndarray] = None) -> Circuit:
        """
        Returns a parametrizable circuit represention of the ansatz.
        Args:
            params: parameters of the circuit.
        """
        circuit = build_hartree_fock_circuit(
            self.number_of_qubits,
            self.number_of_alpha_electrons,
            self._number_of_beta_electrons,
            self._transformation,
        )
        if params is None:
            params = np.asarray(
                [
                    sympy.Symbol("theta_" + str(i), real=True)
                    for i in range(self.number_of_params)
                ]
            )
        # Build UCCSD generator
        fermion_generator = uccsd_singlet_generator(
            params,
            self.number_of_qubits,
            self.number_of_electrons,
            anti_hermitian=True,
        )

        evolution_operator = exponentiate_fermion_operator(
            fermion_generator,
            self._transformation,
            self.number_of_qubits,
        )

        circuit += evolution_operator
        return circuit

    def _assert_number_of_spatial_orbitals(self):
        if self._number_of_spatial_orbitals < 2:
            raise (
                ValueError(
                    "Number of spatials orbitals must be greater "
                    "or equal 2 and is {0}.".format(self._number_of_spatial_orbitals)
                )
            )
        if self._number_of_spatial_orbitals <= self._number_of_alpha_electrons:
            raise (
                ValueError(
                    "Number of spatial orbitals must be greater than "
                    "number_of_alpha_electrons and is {0}".format(
                        self._number_of_spatial_orbitals
                    )
                )
            )

    def _assert_number_of_layers(self):
        if self._number_of_layers != 1:
            raise (
                ValueError(
                    "Number of layers must be equal to 1 for Singlet UCCSD Ansatz"
                )
            )
