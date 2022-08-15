################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
import operator
from functools import reduce
from typing import Iterable, Optional, Union

import numpy as np
import sympy
from openfermion import (
    FermionOperator,
    InteractionOperator,
    QubitOperator,
    bravyi_kitaev,
    get_fermion_operator,
    jordan_wigner,
)
from orquestra.quantum import circuits
from orquestra.quantum.circuits import CNOT, RX, RZ, Circuit, H, X


def exponentiate_fermion_operator(
    fermion_generator: Union[FermionOperator, InteractionOperator],
    transformation: str = "Jordan-Wigner",
    number_of_qubits: Optional[int] = None,
) -> Circuit:
    """Create a circuit corresponding to the exponentiation of an operator.
        Works only for antihermitian fermionic operators.

    Args:
        fermion_generator: fermionic generator.
        transformation: The name of the qubit-to-fermion transformation to use.
        number_of_qubits: This can be used to force the number of qubits in
            the resulting operator above the number that appears in the input operator.
            Defaults to None and the number of qubits in the resulting operator will
            match the number that appears in the input operator.
    """
    if transformation not in ["Jordan-Wigner", "Bravyi-Kitaev"]:
        raise RuntimeError(f"Unrecognized transformation {transformation}")

    # Transform generator to qubits
    if transformation == "Jordan-Wigner":
        qubit_generator = jordan_wigner(fermion_generator)
    else:
        if isinstance(fermion_generator, InteractionOperator):
            fermion_generator = get_fermion_operator(fermion_generator)
        qubit_generator = bravyi_kitaev(fermion_generator, n_qubits=number_of_qubits)

    for term in qubit_generator.terms:
        if isinstance(qubit_generator.terms[term], sympy.Expr):
            if sympy.re(qubit_generator.terms[term]) != 0:
                raise RuntimeError(
                    "Transformed fermion_generator is not anti-hermitian."
                )
            qubit_generator.terms[term] = sympy.im(qubit_generator.terms[term])
        else:
            if not np.isclose(qubit_generator.terms[term].real, 0.0):
                raise RuntimeError(
                    "Transformed fermion_generator is not anti-hermitian."
                )
            qubit_generator.terms[term] = float(qubit_generator.terms[term].imag)
    qubit_generator.compress()

    # Quantum circuit implementing the excitation operators
    circuit = _time_evolution_for_qubit_operator(
        qubit_generator, 1, method="Trotter", trotter_order=1
    )

    return circuit


def build_hartree_fock_circuit(
    number_of_qubits: int,
    number_of_alpha_electrons: int,
    number_of_beta_electrons: int,
    transformation: str,
    spin_ordering: str = "interleaved",
) -> Circuit:
    """Creates a circuit that prepares the Hartree-Fock state.

    Args:
        number_of_qubits: the number of qubits in the system.
        number_of_alpha_electrons: the number of alpha electrons in the system.
        number_of_beta_electrons: the number of beta electrons in the system.
        transformation: the Hamiltonian transformation to use.
        spin_ordering: the spin ordering convention to use. Defaults to "interleaved".

    Returns:
        orquestra.quantum.circuit.Circuit: a circuit that prepares Hartree-Fock state.
    """
    if spin_ordering != "interleaved":
        raise RuntimeError(
            f"{spin_ordering} is not supported at this time. Interleaved is the only"
            "supported spin-ordering."
        )
    circuit = Circuit(n_qubits=number_of_qubits)

    alpha_indexes = list(range(0, number_of_qubits, 2))
    beta_indexes = list(range(1, number_of_qubits, 2))
    index_list = []
    for index in alpha_indexes[:number_of_alpha_electrons]:
        index_list.append(index)
    for index in beta_indexes[:number_of_beta_electrons]:
        index_list.append(index)
    index_list.sort()
    op_list = [(x, 1) for x in index_list]
    fermion_op = FermionOperator(tuple(op_list), 1.0)
    if transformation == "Jordan-Wigner":
        transformed_op = jordan_wigner(fermion_op)
    elif transformation == "Bravyi-Kitaev":
        transformed_op = bravyi_kitaev(fermion_op, n_qubits=number_of_qubits)
    else:
        raise RuntimeError(
            f"{transformation} is not a supported transformation. Jordan-Wigner and "
            "Bravyi-Kitaev are supported at this time."
        )
    term = next(iter(transformed_op.terms.items()))
    for op in term[0]:
        if op[1] != "Z":
            circuit += X(op[0])
    return circuit


def _time_evolution_for_qubit_operator(
    hamiltonian: QubitOperator,
    time: Union[float, sympy.Expr],
    method: str = "Trotter",
    trotter_order: int = 1,
) -> circuits.Circuit:
    """This method is a duplicate of orquestra.quantum.evolution.time_evolution. It
    performs the same functionality but for a QubitOperator.

    This is needed here because singlet uccsd ansatz uses qubit operators with symbolic
    coefficients through the exponentiate_fermion_operator function. This is a temporary
    solution until we implement support for symbolic coefficients in the orquestra
    pauli operator classes.

    See docstring of orquestra.quantum.evolution.time_evolution for more information
    about time evolution.
    """
    if method != "Trotter":
        raise ValueError(f"Currently the method {method} is not supported.")

    terms: Iterable = list(hamiltonian.get_operators())

    return reduce(
        operator.add,
        (
            _time_evolution_for_qubit_operator_term(term, time / trotter_order)
            for _index_order in range(trotter_order)
            for term in terms
        ),
    )


def _time_evolution_for_qubit_operator_term(
    term: QubitOperator, time: Union[float, sympy.Expr]
) -> circuits.Circuit:
    """This method is a duplicate of orquestra.quantum.evolution.time_evolution_for_term
    It performs the same functionality but for a QubitOperator.

    See docstring of `_time_evolution_for_qubit_operator`
    """

    if len(term.terms) != 1:
        raise ValueError("This function works only on a single term.")
    term_components = list(term.terms.keys())[0]
    base_changes = []
    base_reversals = []
    cnot_gates = []
    central_gate: Optional[circuits.GateOperation] = None
    term_types = [component[1] for component in term_components]
    qubit_indices = [component[0] for component in term_components]
    coefficient = list(term.terms.values())[0]

    circuit = circuits.Circuit()

    # If constant term, return empty circuit.
    if not term_components:
        return circuit

    for i, (term_type, qubit_id) in enumerate(zip(term_types, qubit_indices)):
        if term_type == "X":
            base_changes.append(H(qubit_id))
            base_reversals.append(H(qubit_id))
        elif term_type == "Y":
            base_changes.append(RX(np.pi / 2)(qubit_id))
            base_reversals.append(RX(-np.pi / 2)(qubit_id))
        if i == len(term_components) - 1:
            central_gate = RZ(2 * time * coefficient)(qubit_id)
        else:
            cnot_gates.append(CNOT(qubit_id, qubit_indices[i + 1]))

    for gate in base_changes:
        circuit += gate

    for gate in cnot_gates:
        circuit += gate

    if central_gate is not None:
        circuit += central_gate

    for gate in reversed(cnot_gates):
        circuit += gate

    for gate in base_reversals:
        circuit += gate

    return circuit
