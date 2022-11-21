################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
"""This module contains utils related to openfermion created by Zapata."""


import itertools
from typing import Iterable, Optional

import numpy as np
from openfermion import (
    FermionOperator,
    InteractionOperator,
    InteractionRDM,
    PolynomialTensor,
    QubitOperator,
    count_qubits,
    freeze_orbitals,
    get_fermion_operator,
    get_interaction_operator,
    get_sparse_operator,
    jw_get_ground_state_at_particle_number,
    normal_ordered,
    number_operator,
)

from ._openfermion_adapter import openfermion_adapter


def get_fermion_number_operator(n_qubits, n_particles=None):
    """Return a FermionOperator representing the number operator for n qubits.
    If `n_particles` is specified, it can be used for creating constraint on the number
    of particles.

    Args:
        n_qubits (int): number of qubits in the system
        n_particles (int): number of particles in the system.
            If specified, it is subtracted from the number
            operator such as expectation value is zero.
    Returns:
         (openfermion.ops.FermionOperator): the number operator
    """
    operator = number_operator(n_qubits)
    if n_particles is not None:
        operator += FermionOperator("", -1.0 * float(n_particles))
    return get_interaction_operator(operator)


def get_diagonal_component(operator):
    if isinstance(operator, InteractionOperator):
        return _get_diagonal_component_interaction_operator(operator)
    elif isinstance(operator, PolynomialTensor):
        return _get_diagonal_component_polynomial_tensor(operator)
    else:
        raise TypeError(
            f"Getting diagonal component not supported for {0}".format(type(operator))
        )


def _get_diagonal_component_polynomial_tensor(polynomial_tensor):
    """Get the component of an interaction operator that is
    diagonal in the computational basis under Jordan-Wigner
    transformation (i.e., the terms that can be expressed
    as products of number operators).
    Args:
        interaction_operator (openfermion.ops.InteractionOperator): the operator

    Returns:
        tuple: two openfermion.ops.InteractionOperator objects. The first is the
            diagonal component, and the second is the remainder.
    """
    n_modes = count_qubits(polynomial_tensor)
    remainder_tensors = {}
    diagonal_tensors = {}

    diagonal_tensors[()] = polynomial_tensor.constant
    for key in polynomial_tensor.n_body_tensors:
        if key == ():
            continue
        remainder_tensors[key] = np.zeros((n_modes,) * len(key), complex)
        diagonal_tensors[key] = np.zeros((n_modes,) * len(key), complex)

        for indices in itertools.product(range(n_modes), repeat=len(key)):
            creation_counts = {}
            annihilation_counts = {}

            for meta_index, index in enumerate(indices):
                if key[meta_index] == 0:
                    if annihilation_counts.get(index) is None:
                        annihilation_counts[index] = 1
                    else:
                        annihilation_counts[index] += 1
                elif key[meta_index] == 1:
                    if creation_counts.get(index) is None:
                        creation_counts[index] = 1
                    else:
                        creation_counts[index] += 1

            term_is_diagonal = True
            for index in creation_counts:
                if creation_counts[index] != annihilation_counts.get(index):
                    term_is_diagonal = False
                    break
            if term_is_diagonal:
                for index in annihilation_counts:
                    if annihilation_counts[index] != creation_counts.get(index):
                        term_is_diagonal = False
                        break
            if term_is_diagonal:
                diagonal_tensors[key][indices] = polynomial_tensor.n_body_tensors[key][
                    indices
                ]
            else:
                remainder_tensors[key][indices] = polynomial_tensor.n_body_tensors[key][
                    indices
                ]

    return PolynomialTensor(diagonal_tensors), PolynomialTensor(remainder_tensors)


def _get_diagonal_component_interaction_operator(interaction_operator):
    """Get the component of an interaction operator that is
    diagonal in the computational basis under Jordan-Wigner
    transformation (i.e., the terms that can be expressed
    as products of number operators).
    Args:
        interaction_operator (openfermion.ops.InteractionOperator): the operator

    Returns:
        tuple: two openfermion.ops.InteractionOperator objects. The first is the
            diagonal component, and the second is the remainder.
    """

    one_body_tensor = np.zeros(
        interaction_operator.one_body_tensor.shape, dtype=complex
    )
    two_body_tensor = np.zeros(
        interaction_operator.two_body_tensor.shape, dtype=complex
    )
    diagonal_op = InteractionOperator(
        interaction_operator.constant, one_body_tensor, two_body_tensor
    )

    one_body_tensor = np.copy(interaction_operator.one_body_tensor).astype(complex)
    two_body_tensor = np.copy(interaction_operator.two_body_tensor).astype(complex)
    remainder_op = InteractionOperator(0.0, one_body_tensor, two_body_tensor)

    n_spin_orbitals = interaction_operator.two_body_tensor.shape[0]

    for p in range(n_spin_orbitals):
        for q in range(n_spin_orbitals):
            diagonal_op.two_body_tensor[
                p, q, p, q
            ] = interaction_operator.two_body_tensor[p, q, p, q]
            diagonal_op.two_body_tensor[
                p, q, q, p
            ] = interaction_operator.two_body_tensor[p, q, q, p]
            remainder_op.two_body_tensor[p, q, p, q] = 0.0
            remainder_op.two_body_tensor[p, q, q, p] = 0.0

    for p in range(n_spin_orbitals):
        diagonal_op.one_body_tensor[p, p] = interaction_operator.one_body_tensor[p, p]
        remainder_op.one_body_tensor[p, p] = 0.0

    return diagonal_op, remainder_op


def get_polynomial_tensor(fermion_operator, n_qubits=None):
    r"""Convert a fermionic operator to a Polynomial Tensor.

    Args:
        fermion_operator (openferion.ops.FermionOperator): The operator.
        n_qubits (int): The number of qubits to be included in the
            PolynomialTensor. Must be at least equal to the number of qubits
            that are acted on by fermion_operator. If None, then the number of
            qubits is inferred from fermion_operator.

    Returns:
        openfermion.ops.PolynomialTensor: The tensor representation of the
            operator.
    """
    if not isinstance(fermion_operator, FermionOperator):
        raise TypeError("Input must be a FermionOperator.")

    if n_qubits is None:
        n_qubits = count_qubits(fermion_operator)
    if n_qubits < count_qubits(fermion_operator):
        raise ValueError("Invalid number of qubits specified.")

    # Normal order the terms and initialize.
    fermion_operator = normal_ordered(fermion_operator)
    tensor_dict = {}

    # Loop through terms and assign to matrix.
    for term in fermion_operator.terms:
        coefficient = fermion_operator.terms[term]

        # Handle constant shift.
        if len(term) == 0:
            tensor_dict[()] = coefficient

        else:
            key = tuple([operator[1] for operator in term])
            if tensor_dict.get(key) is None:
                tensor_dict[key] = np.zeros((n_qubits,) * len(key), complex)

            indices = tuple([operator[0] for operator in term])
            tensor_dict[key][indices] = coefficient

    return PolynomialTensor(tensor_dict)


@openfermion_adapter()
def get_ground_state_rdm_from_qubit_op(
    qubit_operator: QubitOperator, n_particles: int
) -> InteractionRDM:
    """Diagonalize operator and compute the ground state 1- and 2-RDM

    Args:
        qubit_operator: The openfermion operator to diagonalize
        n_particles: number of particles in the target ground state

    Returns:
        rdm: interaction RDM of the ground state with the particle number n_particles
    """

    sparse_operator = get_sparse_operator(qubit_operator)
    e, ground_state_wf = jw_get_ground_state_at_particle_number(
        sparse_operator, n_particles
    )  # float/np.array pair
    n_qubits = count_qubits(qubit_operator)

    one_body_tensor_list = []
    for i in range(n_qubits):
        for j in range(n_qubits):
            idag_j = get_sparse_operator(
                FermionOperator(f"{i}^ {j}"), n_qubits=n_qubits
            )
            idag_j = idag_j.toarray()
            one_body_tensor_list.append(
                np.conjugate(ground_state_wf) @ idag_j @ ground_state_wf
            )

    one_body_tensor = np.array(one_body_tensor_list)
    one_body_tensor = one_body_tensor.reshape(n_qubits, n_qubits)

    two_body_tensor = np.zeros((n_qubits,) * 4, dtype=complex)
    for p in range(n_qubits):
        for q in range(0, p + 1):
            for r in range(n_qubits):
                for s in range(0, r + 1):
                    pdag_qdag_r_s = get_sparse_operator(
                        FermionOperator(f"{p}^ {q}^ {r} {s}"), n_qubits=n_qubits
                    )
                    pdag_qdag_r_s = pdag_qdag_r_s.toarray()
                    rdm_element = (
                        np.conjugate(ground_state_wf) @ pdag_qdag_r_s @ ground_state_wf
                    )
                    two_body_tensor[p, q, r, s] = rdm_element
                    two_body_tensor[q, p, r, s] = -rdm_element
                    two_body_tensor[q, p, s, r] = rdm_element
                    two_body_tensor[p, q, s, r] = -rdm_element

    return InteractionRDM(one_body_tensor, two_body_tensor)


def remove_inactive_orbitals(
    interaction_op: InteractionOperator, n_active: Optional[int] = None, n_core: int = 0
) -> InteractionOperator:
    """Remove orbitals not in the active space from an interaction operator.

    Args:
        interaction_op: the operator, assumed to be ordered with alternating spin-up and
            spin-down spin orbitals.
        n_active: the number of active molecular orbitals. If None, include all orbitals
            beyond n_core. Note that the number of active spin orbitals will be twice
            the number of active molecular orbitals.
        n_core: the number of core molecular orbitals to be frozen.

    Returns:
        The interaction operator with inactive orbitals removed, and the Hartree-Fock
            energy of the core orbitals added to the constant.
    """

    # This implementation is probably not very efficient, because it converts the
    # interaction operator into a fermion operator and then back to an interaction
    # operator.

    # Convert the InteractionOperator to a FermionOperator
    fermion_op = get_fermion_operator(interaction_op)

    # Determine which occupied spin-orbitals are to be frozen
    occupied = range(2 * n_core)

    unoccupied: Iterable
    # Determine which unoccupied spin-orbitals are to be frozen
    if n_active is not None:
        unoccupied = range(
            2 * n_core + 2 * n_active, interaction_op.one_body_tensor.shape[0]
        )
    else:
        unoccupied = []

    # Freeze the spin-orbitals
    frozen_fermion_op = freeze_orbitals(fermion_op, occupied, unoccupied)

    # Convert back to an interaction operator
    frozen_interaction_op = get_interaction_operator(frozen_fermion_op)

    return frozen_interaction_op


def hf_rdm(n_alpha: int, n_beta: int, n_orbitals: int) -> InteractionRDM:
    """Construct the RDM corresponding to a Hartree-Fock state.

    Args:
        n_alpha (int): number of spin-up electrons
        n_beta (int): number of spin-down electrons
        n_orbitals (int): number of spatial orbitals (not spin orbitals)

    Returns:
        openfermion.ops.InteractionRDM: the reduced density matrix
    """
    # Determine occupancy of each spin orbital
    occ = np.zeros(2 * n_orbitals)
    occ[: (2 * n_alpha) : 2] = 1
    occ[1 : (2 * n_beta + 1) : 2] = 1

    one_body_tensor = np.diag(occ)

    two_body_tensor = np.zeros([2 * n_orbitals for i in range(4)])
    for i in range(2 * n_orbitals):
        for j in range(2 * n_orbitals):
            if i != j and occ[i] and occ[j]:
                two_body_tensor[i, j, j, i] = 1
                two_body_tensor[i, j, i, j] = -1

    return InteractionRDM(one_body_tensor, two_body_tensor)
