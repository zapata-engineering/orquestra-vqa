import numpy as np
import pytest
from orquestra.opt.optimizers import ScipyOptimizer
from orquestra.quantum.runners.symbolic_simulator import SymbolicSimulator

from orquestra.vqa.algorithms.qcbm import QCBM
from orquestra.vqa.ansatz.qcbm import QCBMAnsatz
from orquestra.vqa.estimation.cvar import CvarEstimator


@pytest.fixture()
def optimizer():
    return ScipyOptimizer(method="L-BFGS-B")


N_LAYERS = 2
N_QUBITS = 4


@pytest.fixture()
def simulator():
    return SymbolicSimulator()


@pytest.fixture()
def qcbm_object():
    return QCBM.default(N_QUBITS, N_LAYERS)


@pytest.fixture()
def initial_params():
    return np.random.random(N_QUBITS)


class TestQCBM:
    def test_default_optimizer_is_lbfgsb(self):
        qcbm = QCBM.default(N_QUBITS, N_LAYERS)
        assert qcbm.optimizer.method == "L-BFGS-B"

    def test_default_ansatz_is_farhi(self):
        qcbm = QCBM.default(N_QUBITS, N_LAYERS)
        assert isinstance(qcbm.ansatz, QCBMAnsatz)
        assert qcbm.ansatz.number_of_layers == N_LAYERS

    def test_default_estimation_is_calculate_exact_expectation_values(self):
        qcbm = QCBM.default(N_QUBITS, N_LAYERS)
        assert qcbm.estimation_method.__name__ == "calculate_exact_expectation_values"
        assert qcbm._n_shots is None

    def test_default_estimation_changed_to_estimate_by_averaging(self):
        n_shots = 1000
        qcbm = QCBM.default(
            n_qubits=N_QUBITS,
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
        self, use_exact_expectation_values, n_shots
    ):
        with pytest.raises(ValueError):
            _ = QCBM.default(
                n_qubits=N_QUBITS,
                n_layers=N_LAYERS,
                use_exact_expectation_values=use_exact_expectation_values,
                n_shots=n_shots,
            )

    def test_init_works(self, optimizer):
        estimation_method = CvarEstimator(alpha=0.5)
        n_shots = 1000
        qcbm = QCBM(
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            optimizer=optimizer,
            estimation_method=estimation_method,
            n_shots=n_shots,
        )
        assert qcbm.optimizer is optimizer
        assert qcbm.estimation_method is estimation_method

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

    def test_replace_n_qubits(self, qcbm_object):
        new_qcbm_object = qcbm_object.replace_n_qubits(n_qubits=20)

        assert new_qcbm_object._n_qubits != qcbm_object._n_qubits
        assert new_qcbm_object._n_qubits == 20

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

    # def test_get_cost_function(self, qcbm_object, simulator):
    #     cost_function = qcbm_object.get_cost_function(simulator)
    #     assert cost_function(np.zeros(qcbm_object.ansatz.number_of_params)) >= 0

    def test_get_circuit(self, qcbm_object):
        params = np.random.random(qcbm_object.ansatz.number_of_params)
        circuit = qcbm_object.get_circuit(params)
        assert circuit.n_qubits == qcbm_object.ansatz.number_of_qubits
