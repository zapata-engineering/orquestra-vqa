################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
from orquestra.opt.gradients import finite_differences_gradient
from orquestra.quantum.distributions import (
    MeasurementOutcomeDistribution,
    compute_clipped_negative_log_likelihood,
    compute_mmd,
)
from orquestra.quantum.runners import SymbolicSimulator

from orquestra.vqa.ansatz.qcbm._qcbm import QCBMAnsatz
from orquestra.vqa.cost_function.qcbm_cost_function import create_QCBM_cost_function

number_of_layers = 1
number_of_qubits = 4
topology = "all"
ansatz = QCBMAnsatz(number_of_layers, number_of_qubits, topology)
target_distribution = MeasurementOutcomeDistribution(
    {
        "0000": 1.0,
        "0001": 0.0,
        "0010": 0.0,
        "0011": 1.0,
        "0100": 0.0,
        "0101": 1.0,
        "0110": 0.0,
        "0111": 0.0,
        "1000": 0.0,
        "1001": 0.0,
        "1010": 1.0,
        "1011": 0.0,
        "1100": 1.0,
        "1101": 0.0,
        "1110": 0.0,
        "1111": 1.0,
    }
)

runner = SymbolicSimulator()

n_samples = 1


class Test_create_QCBM_cost_function:
    @pytest.fixture(
        params=[
            {
                "distance_measure": compute_clipped_negative_log_likelihood,
                "distance_measure_parameters": {"epsilon": 1e-6},
            },
            {
                "distance_measure": compute_mmd,
                "distance_measure_parameters": {"sigma": 1},
            },
        ]
    )
    def distance_measure_kwargs(self, request):
        return request.param

    def test_evaluate_history(self, distance_measure_kwargs):
        # Given
        cost_function = create_QCBM_cost_function(
            ansatz,
            runner,
            n_samples,
            **distance_measure_kwargs,
            target_distribution=target_distribution,
        )
        from orquestra.opt.history.recorder import recorder

        cost_function = recorder(cost_function)

        params = np.array([0, 0, 0, 0])

        # When
        value_estimate = cost_function(params)
        history = cost_function.history

        # Then
        assert len(history) == 1
        np.testing.assert_array_equal(params, history[0].params)
        assert value_estimate == history[0].value

    @pytest.mark.parametrize(
        "gradient_kwargs, cf_factory",
        [
            (
                {"gradient_function": finite_differences_gradient},
                create_QCBM_cost_function,
            ),
        ],
    )
    def test_gradient(self, gradient_kwargs, cf_factory, distance_measure_kwargs):
        # Given
        cost_function = cf_factory(
            ansatz,
            runner,
            n_samples,
            **distance_measure_kwargs,
            target_distribution=target_distribution,
            **gradient_kwargs,
        )

        params = np.array([0, 0, 0, 0])

        # When
        gradient = cost_function.gradient(params)

        # Then
        assert len(params) == len(gradient)

    def test_error_raised_if_target_distribution_and_ansatz_are_for_differing_number_of_qubits(  # noqa E501
        self, distance_measure_kwargs
    ):
        # Given
        ansatz.number_of_qubits = 5

        # When/Then
        with pytest.raises(AssertionError):
            cost_function = (
                create_QCBM_cost_function(
                    ansatz,
                    runner,
                    n_samples,
                    **distance_measure_kwargs,
                    target_distribution=target_distribution,
                ),
            )
            params = np.array([0, 0, 0, 0])
            cost_function(params)
