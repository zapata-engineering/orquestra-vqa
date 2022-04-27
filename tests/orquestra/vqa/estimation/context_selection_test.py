################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import numpy as np
from orquestra.quantum.api.estimation import EstimationTask
from orquestra.quantum.circuits import RX, RY, Circuit, X
from orquestra.quantum.openfermion import QubitOperator, qubit_operator_sparse
from orquestra.quantum.openfermion.zapata_utils._utils import change_operator_type

from orquestra.vqa.estimation.context_selection import (
    get_context_selection_circuit_for_group,
    perform_context_selection,
)


class TestEstimatorUtils:
    def test_get_context_selection_circuit_for_group(self):
        group = QubitOperator("X0 Y1") - 0.5 * QubitOperator((1, "Y"))
        circuit, ising_operator = get_context_selection_circuit_for_group(group)

        # Need to convert to QubitOperator in order to get matrix representation
        qubit_operator = change_operator_type(ising_operator, QubitOperator)

        target_unitary = qubit_operator_sparse(group)
        transformed_unitary = (
            circuit.to_unitary().conj().T
            @ qubit_operator_sparse(qubit_operator)
            @ circuit.to_unitary()
        )

        assert np.allclose(target_unitary.todense(), transformed_unitary)

    def test_perform_context_selection(self):
        target_operators = []
        target_operators.append(10.0 * QubitOperator("Z0"))
        target_operators.append(-3 * QubitOperator("Y0"))
        target_operators.append(1 * QubitOperator("X0"))
        target_operators.append(20 * QubitOperator(""))

        expected_operators = []
        expected_operators.append(10.0 * QubitOperator("Z0"))
        expected_operators.append(-3 * QubitOperator("Z0"))
        expected_operators.append(1 * QubitOperator("Z0"))
        expected_operators.append(20 * QubitOperator(""))

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
