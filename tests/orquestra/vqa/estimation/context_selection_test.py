################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import numpy as np
from orquestra.quantum.api.estimation import EstimationTask
from orquestra.quantum.circuits import RX, RY, Circuit, X
from orquestra.quantum.operators import PauliTerm, get_sparse_operator

from orquestra.vqa.estimation.context_selection import (
    get_context_selection_circuit_for_group,
    perform_context_selection,
)


class TestEstimatorUtils:
    def test_get_context_selection_circuit_for_group(self):
        group = PauliTerm("X0*Y1") - 0.5 * PauliTerm("Y1")
        circuit, ising_operator = get_context_selection_circuit_for_group(group)

        target_unitary = get_sparse_operator(group)
        transformed_unitary = (
            circuit.to_unitary().conj().T
            @ get_sparse_operator(ising_operator)
            @ circuit.to_unitary()
        )

        assert np.allclose(target_unitary.todense(), transformed_unitary)

    def test_perform_context_selection(self):
        target_operators = []
        target_operators.append(10.0 * PauliTerm("Z0"))
        target_operators.append(-3 * PauliTerm("Y0"))
        target_operators.append(1 * PauliTerm("X0"))
        target_operators.append(20 * PauliTerm.identity())

        expected_operators = []
        expected_operators.append(10.0 * PauliTerm("Z0"))
        expected_operators.append(-3 * PauliTerm("Z0"))
        expected_operators.append(1 * PauliTerm("Z0"))
        expected_operators.append(20 * PauliTerm.identity())

        base_circuit = Circuit([X(0)])
        x_term_circuit = Circuit([RY(-np.pi / 2)(0)])
        y_term_circuit = Circuit([RX(np.pi / 2)(0)])

        expected_circuits = [
            base_circuit,
            base_circuit + y_term_circuit,
            base_circuit + x_term_circuit,
            base_circuit,
        ]

        estimation_tasks = [
            EstimationTask(operator, base_circuit, None)
            for operator in target_operators
        ]

        tasks_with_context_selection = perform_context_selection(estimation_tasks)

        for task, expected_circuit, expected_operator in zip(
            tasks_with_context_selection, expected_circuits, expected_operators
        ):
            assert task.operator.terms == expected_operator.terms
            assert task.circuit == expected_circuit
