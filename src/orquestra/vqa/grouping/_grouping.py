################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################

import copy
from typing import List, Tuple, cast

import numpy as np
from orquestra.quantum.api.estimation import EstimationTask
from orquestra.quantum.measurements import ExpectationValues, expectation_values_to_real
from orquestra.quantum.openfermion.ops import QubitOperator


def group_individually(estimation_tasks: List[EstimationTask]) -> List[EstimationTask]:
    """
    Transforms list of estimation tasks by putting each term into a estimation task.

    Args:
        estimation_tasks: list of estimation tasks

    """
    output_estimation_tasks = []
    for estimation_task in estimation_tasks:
        for term in estimation_task.operator.get_operators():
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
            cast(QubitOperator, estimation_task.operator), sort_terms=sort_terms
        )
        for group in groups:
            group_estimation_task = EstimationTask(
                group, estimation_task.circuit, estimation_task.number_of_shots
            )
            output_estimation_tasks.append(group_estimation_task)
    return output_estimation_tasks


def is_comeasureable(
    term_1: Tuple[Tuple[int, str], ...], term_2: Tuple[Tuple[int, str], ...]
) -> bool:
    """Determine if two Pauli terms are co-measureable.

    Co-measureable means that
    for each qubit: if one term contains a Pauli operator acting on a qubit,
    then the other term cannot have a different Pauli operator acting on that
    qubit.

    Args:
        term1: a product of Pauli operators represented in openfermion style
        term2: a product of Pauli operators represented in openfermion style
    Returns:
        bool: True if the terms are co-measureable.
    """

    for qubit_1, operator_1 in term_1:
        for qubit_2, operator_2 in term_2:

            # Check if the two Pauli operators act on the same qubit
            if qubit_1 == qubit_2:

                # Check if the two Pauli operators are different
                if operator_1 != operator_2:
                    return False

    return True


def group_comeasureable_terms_greedy(
    qubit_operator: QubitOperator, sort_terms: bool = False
) -> List[QubitOperator]:
    """Group co-measurable terms in a qubit operator using a greedy algorithm.

    Adapted from PyQuil. Constant term is included as a separate group.

    Args:
        qubit_operator: the operator whose terms are to be grouped
        sort_terms: whether to sort terms by the absolute value of the coefficient when
            grouping.
        Returns:
        A list of qubit operators.
    """

    groups: List[
        QubitOperator
    ] = []  # List of QubitOperators representing groups of co-measureable terms
    constant_term = None

    if sort_terms:
        terms_iterator = sorted(
            qubit_operator.terms.items(), key=lambda x: abs(x[1]), reverse=True
        )
    else:
        terms_iterator = qubit_operator.terms.items()

    for term, coefficient in terms_iterator:
        assigned = False  # True if the current term has been assigned to a group
        if term == ():
            constant_term = QubitOperator(term, coefficient)
            continue
        for group in groups:
            if all(
                is_comeasureable(term, term_to_compare)
                for term_to_compare in group.terms
            ):
                # Add the term to the group
                group += QubitOperator(term, coefficient)
                assigned = True
                break

        # If term was not co-measureable with any group, it gets to start its own group!
        if not assigned:
            groups.append(QubitOperator(term, qubit_operator.terms[term]))

    # Constant term is handled as separate term to make it easier to exclude it
    # from calculations or execution if that's needed.
    if constant_term is not None:
        groups.append(constant_term)

    return groups


def _group_comeasureable_terms_greedy_sorted(
    qubit_operator: QubitOperator,
) -> List[QubitOperator]:
    return group_comeasureable_terms_greedy(qubit_operator, True)


def _calculate_variance_upper_bound(group: QubitOperator) -> float:
    coefficients = np.array(list(group.terms.values()))
    return np.sum(coefficients**2)


def _remove_constant_term_from_group(group: QubitOperator) -> QubitOperator:
    new_group = copy.deepcopy(group)
    if new_group.terms.get(()):
        del new_group.terms[()]
    return new_group


def compute_group_variances(
    groups: List[QubitOperator], expecval: ExpectationValues = None
) -> np.ndarray:
    """Computes the variances of each frame in a grouped operator.

    If expectation values are provided, use variances from there,
    otherwise assume variances are 1 (upper bound). Correlation information
    is ignored in the current implementation, covariances are assumed to be 0.

    Args:
        groups:  A list of QubitOperators that defines a (grouped) operator
        expecval: An ExpectationValues object containing the expectation
            values of the operators.
    Returns:
        frame_variances: A Numpy array of the computed variances for each frame
    """

    if expecval is None:
        groups = [_remove_constant_term_from_group(group) for group in groups]
        frame_variances = [_calculate_variance_upper_bound(group) for group in groups]
    else:
        group_sizes = np.array([len(group.terms.keys()) for group in groups])
        if np.sum(group_sizes) != len(expecval.values):
            raise ValueError(
                "Number of expectation values should be the same as number of terms."
            )
        real_expecval = expectation_values_to_real(expecval)
        if not np.logical_and(
            real_expecval.values >= -1, real_expecval.values <= 1
        ).all():
            raise ValueError("Expectation values should have values between -1 and 1.")

        pauli_variances = 1.0 - real_expecval.values**2
        frame_variances = []
        for i, group in enumerate(groups):
            coeffs = np.array(list(group.terms.values()))
            offset = 0 if i == 0 else np.sum(group_sizes[:i])
            pauli_variances_for_group = pauli_variances[
                offset : offset + group_sizes[i]
            ]
            frame_variances.append(np.sum(coeffs**2 * pauli_variances_for_group))

    return np.array(frame_variances)
