################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
from orquestra.quantum.api.estimation import EstimateExpectationValues, EstimationTask
from orquestra.quantum.api.estimator_contract import (
    ESTIMATOR_CONTRACTS,
    _validate_expectation_value_includes_coefficients,
)
from orquestra.quantum.circuits import RX, Circuit, H, X
from orquestra.quantum.operators import PauliTerm
from orquestra.quantum.runners import SymbolicSimulator
from orquestra.quantum.testing import MockCircuitRunner

from orquestra.vqa.estimation.gibbs_objective import GibbsObjectiveEstimator


# The gibbs objective estimator exponentiates the expectation value of each outcome,
# sums the results, and takes the negative log of the sum. This does not scale linearly
# with the coefficient. We check expectation value includes coefficient differently
def _validate_expectation_value_includes_coefficients_for_gibbs_estimator(
    estimator: EstimateExpectationValues,
):
    runner = SymbolicSimulator(seed=1997)
    term_coefficient = 30
    estimation_tasks = [
        EstimationTask(PauliTerm("Z0"), Circuit([RX(np.pi / 3)(0)]), 10000),
        EstimationTask(
            PauliTerm("Z0", term_coefficient), Circuit([RX(np.pi / 3)(0)]), 10000
        ),
    ]

    expectation_values = estimator(
        runner=runner,
        estimation_tasks=estimation_tasks,
    )

    # For a sufficiently large coefficient, the exponential should become
    # much larger (greater than linear scaling)
    # Note that we take the negative exponential here because the gibbs
    # estimator takes the negative log
    return np.all(
        np.greater(
            np.exp(-expectation_values[1].values) / term_coefficient,
            np.exp(-expectation_values[0].values),
        )
    )


# Remove the test that assumes expectation value scales linearly with coefficient
# and add the new test defined above
ESTIMATOR_CONTRACTS.remove(_validate_expectation_value_includes_coefficients)
ESTIMATOR_CONTRACTS += [
    _validate_expectation_value_includes_coefficients_for_gibbs_estimator
]


@pytest.mark.parametrize("contract", ESTIMATOR_CONTRACTS)
def test_estimator_contract(contract):
    estimator = GibbsObjectiveEstimator(alpha=0.2)
    assert contract(estimator)


class TestGibbsEstimator:
    @pytest.fixture(params=[1.0, 0.8, 0.5, 0.2])
    def estimator(self, request):
        return GibbsObjectiveEstimator(alpha=request.param)

    @pytest.fixture()
    def circuit(self):
        return Circuit([X(0)])

    @pytest.fixture()
    def operator(self):
        return PauliTerm("Z0")

    @pytest.fixture()
    def estimation_tasks(self, operator, circuit):
        return [EstimationTask(operator, circuit, 10)]

    @pytest.fixture()
    def runner(self):
        return MockCircuitRunner()

    def test_raises_exception_if_operator_is_not_ising(
        self, estimator, runner, circuit
    ):
        # Given
        estimation_tasks = [EstimationTask(PauliTerm("X0"), circuit, 10)]
        with pytest.raises(TypeError):
            estimator(
                runner=runner,
                estimation_tasks=estimation_tasks,
            )

    @pytest.mark.parametrize("alpha", [-1, 0])
    def test_gibbs_estimator_raises_exception_if_alpha_less_than_or_equal_to_0(
        self, estimator, runner, estimation_tasks, alpha
    ):
        estimator.alpha = alpha
        with pytest.raises(ValueError):
            estimator(
                runner=runner,
                estimation_tasks=estimation_tasks,
            )

    def test_gibbs_estimator_returns_correct_values(self, estimator, runner, operator):
        # Given
        estimation_tasks = [EstimationTask(operator, Circuit([H(0)]), 10000)]

        expval_0 = np.exp(1 * -estimator.alpha)  # Expectation value of bitstring 0
        expval_1 = np.exp(-1 * -estimator.alpha)  # Expectation value of bitstring 1

        # Target value is the -log of the mean of the expectation values of the 2
        # bitstrings
        target_value = -np.log((expval_1 + expval_0) / 2)

        # When
        expectation_values = estimator(
            runner=runner,
            estimation_tasks=estimation_tasks,
        )

        # Then
        assert expectation_values[0].values == pytest.approx(target_value, abs=2e-2)
