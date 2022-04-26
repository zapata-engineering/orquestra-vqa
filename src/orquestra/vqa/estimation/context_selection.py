################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import List, Tuple, cast

import numpy as np
from orquestra.quantum.api.estimation import EstimationTask
from orquestra.quantum.circuits import RX, RY, Circuit
from orquestra.quantum.openfermion import IsingOperator, QubitOperator


def get_context_selection_circuit_for_group(
    qubit_operator: QubitOperator,
) -> Tuple[Circuit, IsingOperator]:
    """Get the context selection circuit for measuring the expectation value
    of a group of co-measurable Pauli terms.

    Args:
        qubit_operator: operator representing group of co-measurable Pauli term
    """
    context_selection_circuit = Circuit()
    transformed_operator = IsingOperator()
    context: List[Tuple[int, str]] = []

    for term in qubit_operator.terms:
        term_operator = IsingOperator(())
        for qubit, operator in term:
            for existing_qubit, existing_operator in context:
                if existing_qubit == qubit and existing_operator != operator:
                    raise ValueError("Terms are not co-measurable")
            if (qubit, operator) not in context:
                context.append((qubit, operator))
            term_operator *= IsingOperator((qubit, "Z"))
        transformed_operator += term_operator * qubit_operator.terms[term]

    for factor in context:
        if factor[1] == "X":
            context_selection_circuit += RY(-np.pi / 2)(factor[0])
        elif factor[1] == "Y":
            context_selection_circuit += RX(np.pi / 2)(factor[0])

    return context_selection_circuit, transformed_operator


def perform_context_selection(
    estimation_tasks: List[EstimationTask],
) -> List[EstimationTask]:
    """Changes the circuits in estimation tasks to involve context selection.

    Args:
        estimation_tasks: list of estimation tasks
    """
    output_estimation_tasks = []
    for estimation_task in estimation_tasks:
        (
            context_selection_circuit,
            frame_operator,
        ) = get_context_selection_circuit_for_group(
            cast(QubitOperator, estimation_task.operator)
        )
        frame_circuit = estimation_task.circuit + context_selection_circuit
        new_estimation_task = EstimationTask(
            frame_operator, frame_circuit, estimation_task.number_of_shots
        )
        output_estimation_tasks.append(new_estimation_task)
    return output_estimation_tasks
