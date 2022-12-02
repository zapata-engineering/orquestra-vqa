import numpy as np
import pytest
from orquestra.opt.optimizers import ScipyOptimizer
from orquestra.quantum.distributions import MeasurementOutcomeDistribution
from orquestra.quantum.distributions.BAS_dataset import (
    get_bars_and_stripes_target_distribution,
)
from orquestra.quantum.runners import SymbolicSimulator

from orquestra.vqa.algorithms import QCBM
from orquestra.vqa.ansatz import QCBMAnsatz
from orquestra.vqa.estimation import CvarEstimator

N_LAYERS = 2
N_QUBITS = 4


@pytest.fixture()
def optimizer():
    return ScipyOptimizer(method="L-BFGS-B")


@pytest.fixture()
def simulator():
    return SymbolicSimulator()


@pytest.fixture()
def initial_params():
    return np.random.random(N_QUBITS)


@pytest.fixture
def target_distribution():
    return get_bars_and_stripes_target_distribution(int(N_QUBITS / 2), 2, 1.0, "zigzag")


@pytest.fixture()
def qcbm_object(target_distribution):
    return QCBM.default(target_distribution, N_LAYERS)


class TestQCBM:
    def test_default_optimizer_is_lbfgsb(self, target_distribution):
        qcbm = QCBM.default(target_distribution, N_LAYERS)
        assert qcbm.optimizer.method == "L-BFGS-B"

    def test_default_ansatz_is_farhi(self, target_distribution):
        qcbm = QCBM.default(target_distribution, N_LAYERS)
        assert isinstance(qcbm.ansatz, QCBMAnsatz)
        assert qcbm.ansatz.number_of_layers == N_LAYERS

    def test_default_estimation_is_calculate_exact_expectation_values(
        self, target_distribution
    ):
        qcbm = QCBM.default(target_distribution, N_LAYERS)
        assert qcbm.estimation_method.__name__ == "calculate_exact_expectation_values"
        assert qcbm._n_shots is None

    def test_default_estimation_changed_to_estimate_by_averaging(
        self, target_distribution
    ):
        n_shots = 1000
        qcbm = QCBM.default(
            target_distribution,
            n_layers=N_LAYERS,
            use_exact_expectation_values=False,
            n_shots=n_shots,
        )
        assert (
            qcbm.estimation_method.__name__
            == "estimate_expectation_values_by_averaging"
        )
        assert qcbm._n_shots == n_shots

    @pytest.mark.parametrize(
        "use_exact_expectation_values,n_shots", [(True, 1000), (False, None)]
    )
    def test_default_raises_exception_for_invalid_inputs(
        self, target_distribution, use_exact_expectation_values, n_shots
    ):
        with pytest.raises(ValueError):
            _ = QCBM.default(
                target_distribution,
                n_layers=N_LAYERS,
                use_exact_expectation_values=use_exact_expectation_values,
                n_shots=n_shots,
            )

    def test_init_works(self, target_distribution, optimizer):
        estimation_method = CvarEstimator(alpha=0.5)
        n_shots = 1000
        qcbm = QCBM(
            target_distribution,
            n_layers=N_LAYERS,
            optimizer=optimizer,
            estimation_method=estimation_method,
            n_shots=n_shots,
        )
        assert qcbm.optimizer is optimizer
        assert qcbm.estimation_method is estimation_method

    def test_n_qubits(self, qcbm_object):
        assert qcbm_object.n_qubits == N_QUBITS

    def test_replace_optimizer(self, qcbm_object):
        optimizer = ScipyOptimizer(method="COBYLA")

        new_qcbm_object = qcbm_object.replace_optimizer(optimizer)

        assert qcbm_object.optimizer is not optimizer
        assert new_qcbm_object.optimizer is optimizer

    def test_replace_estimation_method(self, qcbm_object):
        estimation_method = CvarEstimator(alpha=0.5)
        n_shots = 1001

        new_qcbm_object = qcbm_object.replace_estimation_method(
            estimation_method, n_shots=n_shots
        )

        assert (
            qcbm_object.estimation_method.__name__
            == "calculate_exact_expectation_values"
        )
        assert qcbm_object.estimation_method is not estimation_method
        assert new_qcbm_object.estimation_method is estimation_method
        assert qcbm_object._n_shots != new_qcbm_object._n_shots

    def test_replace_n_layers(self, qcbm_object):
        new_qcbm_object = qcbm_object.replace_n_layers(n_layers=5)

        assert new_qcbm_object._n_layers != qcbm_object._n_layers
        assert new_qcbm_object._n_layers == 5

    @pytest.mark.parametrize("initial_params", [None, np.random.random(14)])
    def test_find_optimal_params_doesnt_fail(
        self, qcbm_object, simulator, initial_params
    ):
        results = qcbm_object.find_optimal_params(simulator, initial_params)
        assert len(results.opt_params) == qcbm_object.ansatz.number_of_params

    def test_get_cost_function(self, qcbm_object, simulator):
        cost_function = qcbm_object.get_cost_function(simulator)
        assert cost_function(np.zeros(qcbm_object.ansatz.number_of_params)) >= 0

    def test_get_circuit(self, qcbm_object):
        params = np.random.random(qcbm_object.ansatz.number_of_params)
        circuit = qcbm_object.get_circuit(params)
        assert circuit.n_qubits == qcbm_object.ansatz.number_of_qubits
