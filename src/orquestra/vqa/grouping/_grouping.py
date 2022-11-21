################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################

from typing import Iterable, List, Optional

import numpy as np
from orquestra.quantum.api.estimation import EstimationTask
from orquestra.quantum.measurements import ExpectationValues, expectation_values_to_real
from orquestra.quantum.operators import PauliRepresentation, PauliSum, PauliTerm


def group_individually(estimation_tasks: List[EstimationTask]) -> List[EstimationTask]:
    """
    Transforms list of estimation tasks by putting each term into a estimation task.

    Args:
        estimation_tasks: list of estimation tasks

    """
    output_estimation_tasks = []
    for estimation_task in estimation_tasks:
        for term in estimation_task.operator.terms:
            output_estimation_tasks.append(
                EstimationTask(
                    term, estimation_task.circuit, estimation_task.number_of_shots
                )
            )
    return output_estimation_tasks


def group_greedily(
    estimation_tasks: List[EstimationTask], sort_terms: bool = False
) -> List[EstimationTask]:
    """
    Transforms list of estimation tasks by performing greedy grouping and adding
    context selection logic to the circuits.

    Args:
        estimation_tasks: list of estimation tasks
    """
    if sort_terms:
        print("Greedy grouping with pre-sorting")
    else:
        print("Greedy grouping without pre-sorting")
    output_estimation_tasks = []
    for estimation_task in estimation_tasks:
        groups = group_comeasureable_terms_greedy(
            estimation_task.operator, sort_terms=sort_terms
        )
        for group in groups:
            group_estimation_task = EstimationTask(
                group, estimation_task.circuit, estimation_task.number_of_shots
            )
            output_estimation_tasks.append(group_estimation_task)
    return output_estimation_tasks


def is_comeasureable(term_1: PauliTerm, term_2: PauliTerm) -> bool:
    """Determine if two Pauli terms are co-measureable.

    Co-measureable means that
    for each qubit: if one term contains a Pauli operator acting on a qubit,
    then the other term cannot have a different Pauli operator acting on that
    qubit.

    Args:
        term_1: a Pauli term consisting of a product of Pauli operators
        term_2: a Pauli term consisting of a product of Pauli operators
    Returns:
        bool: True if the terms are co-measureable.
    """
    for qubit_1, operator_1 in term_1.operations:
        for qubit_2, operator_2 in term_2.operations:

            # Check if the two Pauli operators act on the same qubit
            if qubit_1 == qubit_2:

                # Check if the two Pauli operators are different
                if operator_1 != operator_2:
                    return False

    return True


def group_comeasureable_terms_greedy(
    qubit_operator: PauliRepresentation, sort_terms: bool = False
) -> List[PauliRepresentation]:
    """Group co-measurable terms in a qubit operator using a greedy algorithm.

    Adapted from PyQuil. Constant term is included as a separate group.

    Args:
        qubit_operator: the operator whose terms are to be grouped
        sort_terms: whether to sort terms by the absolute value of the coefficient when
            grouping.
        Returns:
        A list of qubit operators.
    """

    # List of pauli operators representing groups of co-measureable terms
    groups: List[PauliRepresentation] = []
    constant_term = None

    terms_iterator: Iterable[PauliTerm]
    if sort_terms:
        # Sort terms by the absolute value of the coefficient
        terms_iterator = sorted(
            qubit_operator.terms, key=lambda x: abs(x.coefficient), reverse=True
        )
    else:
        terms_iterator = qubit_operator.terms

    for term in terms_iterator:
        assigned = False  # True if the current term has been assigned to a group
        if term.is_constant:
            constant_term = PauliTerm.identity() * term.coefficient
            continue
        for i in range(len(groups)):
            if all(
                is_comeasureable(term, term_to_compare)
                for term_to_compare in groups[i].terms
            ):
                # Add the term to the group
                groups[i] += term.copy()
                assigned = True
                break

        # If term was not co-measureable with any group, it gets to start its own group!
        if not assigned:
            groups.append(term.copy())

    # Constant term is handled as separate term to make it easier to exclude it
    # from calculations or execution if that's needed.
    if constant_term is not None:
        groups.append(constant_term)

    return groups


def _calculate_variance_upper_bound(group: PauliRepresentation) -> float:
    coefficients = np.array([term.coefficient for term in group.terms])
    return np.sum(coefficients**2)


def _remove_constant_term_from_group(group: PauliRepresentation) -> PauliSum:
    return PauliSum([term for term in group.terms if not term.is_constant])


def compute_group_variances(
    groups: List[PauliRepresentation], expecval: Optional[ExpectationValues] = None
) -> np.ndarray:
    """Computes the variances of each frame in a grouped operator.

    If expectation values are provided, use variances from there,
    otherwise assume the upper bound for variances. Correlation information
    is ignored in the current implementation, covariances are assumed to be 0.

    Args:
        groups:  A list of Pauli terms that defines a (grouped) operator
        expecval: An ExpectationValues object containing the expectation
            values of each Pauli term. The term coefficients should be
            included in the expectation values, e.g. the expectation value of
            2*Z0 should be between -2 and 2.
    Returns:
        frame_variances: Computed variances for each frame.
    """

    if expecval is None:
        groups = [_remove_constant_term_from_group(group) for group in groups]
        frame_variances = [_calculate_variance_upper_bound(group) for group in groups]
    else:
        group_sizes = np.array([len(group.terms) for group in groups])
        if np.sum(group_sizes) != len(expecval.values):
            raise ValueError(
                "Number of expectation values should be the same as number of terms."
            )
        real_expecval = expectation_values_to_real(expecval)
        frame_variances = []
        for i, group in enumerate(groups):
            coeffs = np.array([term.coefficient for term in group.terms])
            offset = 0 if i == 0 else np.sum(group_sizes[:i])
            real_expecval_for_group = real_expecval.values[
                offset : offset + group_sizes[i]
            ]
            if (np.abs(real_expecval_for_group) > np.abs(coeffs)).any():
                raise ValueError(
                    "Absolute value of expectation value exceeds absolute value of"
                    "Pauli term coefficient."
                )

            frame_variances.append(np.sum(coeffs**2 - real_expecval_for_group**2))

    return np.array(frame_variances)
