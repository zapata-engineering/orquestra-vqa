from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.ansatz_utils import (
    ansatz_property,
    invalidates_parametrized_circuit,
)
from zquantum.core.circuit import Circuit
from openfermion.utils import uccsd_singlet_paramsize, uccsd_singlet_generator
from overrides import overrides
from .utils import exponentiate_fermion_operator, create_layer_of_gates
from pyquil import Program
from pyquil.gates import X
from typing import List, Optional
import sympy
import numpy as np


class SingletUCCSDAnsatz(Ansatz):

    supports_parametrized_circuits = False
    transformation = ansatz_property("transformation")

    def __init__(
        self,
        n_spatial_orbitals: int,
        n_alpha: int,
        n_layers: int,
        transformation: str = "Jordan-Wigner",
    ):
        """ 
        Ansatz class representing Singlet UCCSD Ansatz.

        Args:
            n_spatial_orbitals: number of spatial orbitals.
            n_alpha: number of alpha electrons.
            n_layers: number of layers of the ansatz.
            transformation: transformation used for translation between fermions and qubits.

        Attributes:
            n_beta: number of beta electrons (equal to n_alpha).
            n_electrons: total number of electrons (n_alpha + n_beta).
            n_qubits: number of qubits required for the ansatz circuit.
            number_of_params: number of the parameters that need to be set for the ansatz circuit.
        """
        super().__init__(n_layers)
        self._n_spatial_orbitals = n_spatial_orbitals
        self._n_alpha = n_alpha
        self._transformation = transformation
        self._assert_n_spatial_orbitals()

    @property
    def n_spatial_orbitals(self):
        return self._n_spatial_orbitals

    @invalidates_parametrized_circuit
    @n_spatial_orbitals.setter
    def n_spatial_orbitals(self, new_n_spatial_orbitals):
        self._n_spatial_orbitals = new_n_spatial_orbitals
        self._assert_n_spatial_orbitals()

    @property
    def n_qubits(self):
        return self._n_spatial_orbitals * 2

    @property
    def n_alpha(self):
        return self._n_alpha

    @invalidates_parametrized_circuit
    @n_alpha.setter
    def n_alpha(self, new_n_alpha):
        self._n_alpha = new_n_alpha
        self._assert_n_spatial_orbitals()

    @property
    def _n_beta(self):
        return self._n_alpha

    @property
    def n_electrons(self):
        return self._n_alpha + self._n_beta

    @property
    def number_of_params(self) -> int:
        """
        Returns number of parameters in the ansatz.
        """
        return uccsd_singlet_paramsize(
            n_qubits=self.n_qubits, n_electrons=self.n_electrons
        )

    @overrides
    def _generate_circuit(self, params: Optional[np.ndarray] = None) -> Circuit:
        """
        Returns a parametrizable circuit represention of the ansatz.
        Args:
            params: parameters of the circuit. 
        """
        circuit = Circuit()

        circuit += create_layer_of_gates(self.n_qubits, "X")

        # Build UCCSD generator
        fermion_generator = uccsd_singlet_generator(
            params, self.n_qubits, self.n_electrons, anti_hermitian=True,
        )

        evolution_operator = exponentiate_fermion_operator(
            fermion_generator, self._transformation
        )

        circuit += Circuit(evolution_operator)
        return circuit

    def _assert_n_spatial_orbitals(self):
        if self._n_spatial_orbitals < 2:
            raise (
                ValueError(
                    "Number of spatials orbitals must be greater or equal and is {0}.".format(
                        self._n_spatial_orbitals
                    )
                )
            )
        if self._n_spatial_orbitals <= self._n_alpha:
            raise (
                ValueError(
                    "Number of spatial orbitals must be greater than n_alpha and is {0}".format(
                        self._n_spatial_orbitals
                    )
                )
            )

