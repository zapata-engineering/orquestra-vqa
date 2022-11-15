################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import pytest
from orquestra.quantum.api.estimation import EstimationTask
from orquestra.quantum.api.estimator_contract import ESTIMATOR_CONTRACTS
from orquestra.quantum.circuits import Circuit, H, X
from orquestra.quantum.operators import PauliTerm
from orquestra.quantum.runners import SymbolicSimulator
from orquestra.quantum.testing import MockCircuitRunner

from orquestra.vqa.estimation.cvar import CvarEstimator


@pytest.mark.parametrize("contract", ESTIMATOR_CONTRACTS)
def test_estimator_contract(contract):
    estimator = CvarEstimator(alpha=0.2)
    assert contract(estimator)


class TestCvarEstimator:
    @pytest.fixture(
        params=[
            {
                "alpha": 0.2,
                "use_exact_expectation_values": False,
            },
            {
                "alpha": 0.2,
                "use_exact_expectation_values": True,
            },
            {
                "alpha": 0.8,
                "use_exact_expectation_values": False,
            },
            {
                "alpha": 0.8,
                "use_exact_expectation_values": True,
            },
            {
                "alpha": 1.0,
                "use_exact_expectation_values": False,
            },
            {
                "alpha": 1.0,
                "use_exact_expectation_values": True,
            },
        ]
    )
    def estimator(self, request):
        return CvarEstimator(**request.param)

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
        return SymbolicSimulator()

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

    def test_cvar_estimator_raises_exception_if_alpha_less_than_0(
        self, estimator, runner, estimation_tasks
    ):
        estimator.alpha = -1
        with pytest.raises(ValueError):
            estimator(
                runner=runner,
                estimation_tasks=estimation_tasks,
            )

    def test_cvar_estimator_raises_exception_if_alpha_greater_than_1(
        self, estimator, runner, estimation_tasks
    ):
        estimator.alpha = 2
        with pytest.raises(ValueError):
            estimator(
                runner=runner,
                estimation_tasks=estimation_tasks,
            )

    def test_cvar_estimator_returns_correct_values(self, estimator, runner, operator):
        # Given
        estimation_tasks = [EstimationTask(operator, Circuit([H(0)]), 10000)]
        if estimator.alpha <= 0.5:
            target_value = -1
        else:
            target_value = (-1 * 0.5 + 1 * (estimator.alpha - 0.5)) / estimator.alpha

        # When
        expectation_values = estimator(
            runner=runner,
            estimation_tasks=estimation_tasks,
        )

        # Then
        assert expectation_values[0].values == pytest.approx(target_value, abs=2e-1)

    def test_raises_exception_if_uses_exact_expectations_without_simulator(
        self, circuit
    ):
        # Given
        runner = MockCircuitRunner()
        estimation_tasks = [EstimationTask(PauliTerm("X0"), circuit, 10)]
        estimator = CvarEstimator(use_exact_expectation_values=True, alpha=0.5)
        with pytest.raises(TypeError):
            estimator = estimator(
                runner=runner,
                estimation_tasks=estimation_tasks,
            )
