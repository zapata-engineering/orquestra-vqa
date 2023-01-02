################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from numbers import Number
from typing import Callable, Optional

import numpy as np
from orquestra.opt.api.cost_function import CostFunction
from orquestra.opt.api.functions import StoreArtifact, function_with_gradient
from orquestra.opt.gradients import finite_differences_gradient
from orquestra.quantum.api import CircuitRunner
from orquestra.quantum.distributions import (
    MeasurementOutcomeDistribution,
    evaluate_distribution_distance,
)
from orquestra.quantum.utils import ValueEstimate

from ..api.ansatz import Ansatz

GradientFactory = Callable[[Callable], Callable[[np.ndarray], np.ndarray]]
DistanceMeasure = Callable[..., Number]


def create_QCBM_cost_function(
    ansatz: Ansatz,
    runner: CircuitRunner,
    n_samples: int,
    distance_measure: DistanceMeasure,
    distance_measure_parameters: dict,
    target_distribution: MeasurementOutcomeDistribution,
    gradient_function: GradientFactory = finite_differences_gradient,
) -> CostFunction:
    """Cost function used for evaluating QCBM.

    Args:
        ansatz: the ansatz used to construct the variational circuits
        runner: runner used for QCBM evaluation
        distance_measure: function used to calculate the distance measure
        distance_measure_parameters: dictionary containing the relevant parameters
            for the chosen distance measure
        target_distribution: bistring distribution which QCBM aims to learn
        gradient_function: a function which returns a function used to compute
            the gradient of the cost function
            (see orquestra.opt.gradients.finite_differences_gradient for reference)
    Returns:
        Callable CostFunction object that evaluates the parametrized circuit produced
        by the ansatz with the given parameters and returns the distance between
        the produced bitstring distribution and the target distribution
    """

    cost_function = _create_QCBM_cost_function(
        ansatz,
        runner,
        n_samples,
        distance_measure,
        distance_measure_parameters,
        target_distribution,
    )

    return function_with_gradient(cost_function, gradient_function(cost_function))


def _create_QCBM_cost_function(
    ansatz: Ansatz,
    runner: CircuitRunner,
    n_samples: int,
    distance_measure: DistanceMeasure,
    distance_measure_parameters: dict,
    target_distribution: MeasurementOutcomeDistribution,
):
    assert (
        int(target_distribution.get_number_of_subsystems()) == ansatz.number_of_qubits
    )

    def cost_function(
        parameters: np.ndarray, store_artifact: Optional[StoreArtifact] = None
    ) -> ValueEstimate:
        """Evaluates the value of the cost function for given parameters.

        Args:
            parameters: parameters for which the evaluation should occur.
            store_artifact: callable defining how the bitstring distributions
                should be stored.
        """
        # TODO: we use private method here due to performance reasons.
        # This should be fixed once better mechanism for handling
        # it will be implemented.
        # In case of questions ask mstechly.
        # circuit = ansatz.get_executable_circuit(parameters)
        circuit = ansatz._generate_circuit(parameters)
        distribution = runner.get_measurement_outcome_distribution(circuit, n_samples)
        value = evaluate_distribution_distance(
            target_distribution,
            distribution,
            distance_measure,
            distance_measure_parameters=distance_measure_parameters,
        )

        if store_artifact:
            store_artifact("measurement_outcome_distribution", distribution)

        return ValueEstimate(value)

    return cost_function
