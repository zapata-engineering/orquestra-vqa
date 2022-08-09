################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import List, Tuple

import numpy as np
from orquestra.quantum.api.estimation import EstimationTask
from orquestra.quantum.circuits import RX, RY, Circuit
from orquestra.quantum.operators import PauliRepresentation, PauliSum, PauliTerm


def get_context_selection_circuit_for_group(
    qubit_operator: PauliRepresentation,
) -> Tuple[Circuit, PauliSum]:
    """Get the context selection circuit for measuring the expectation value
    of a group of co-measurable Pauli terms.

    Args:
        qubit_operator: operator representing group of co-measurable Pauli term
    """
    context_selection_circuit = Circuit()
    transformed_operator = PauliSum([])
    context: List[Tuple[str, int]] = []

    for term in qubit_operator.terms:
        term_operator = PauliTerm.identity()
        for qubit, operator in term.operations:
            for existing_qubit, existing_operator in context:
                if existing_qubit == qubit and existing_operator != operator:
                    raise ValueError("Terms are not co-measurable")
            if (operator, qubit) not in context:
                context.append((operator, qubit))
            product = term_operator * PauliTerm({qubit: "Z"})
            assert isinstance(product, PauliTerm)
            term_operator = product
        transformed_operator += term_operator * term.coefficient

    for factor in context:
        if factor[0] == "X":
            context_selection_circuit += RY(-np.pi / 2)(factor[1])
        elif factor[0] == "Y":
            context_selection_circuit += RX(np.pi / 2)(factor[1])

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
        ) = get_context_selection_circuit_for_group(estimation_task.operator)
        frame_circuit = estimation_task.circuit + context_selection_circuit
        new_estimation_task = EstimationTask(
            frame_operator, frame_circuit, estimation_task.number_of_shots
        )
        output_estimation_tasks.append(new_estimation_task)
    return output_estimation_tasks
