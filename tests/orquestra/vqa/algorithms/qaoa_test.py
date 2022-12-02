import numpy as np
import pytest
from orquestra.opt.optimizers import ScipyOptimizer
from orquestra.quantum.operators import PauliTerm
from orquestra.quantum.runners import SymbolicSimulator

from orquestra.vqa.algorithms import QAOA
from orquestra.vqa.ansatz import QAOAFarhiAnsatz, WarmStartQAOAAnsatz
from orquestra.vqa.estimation import CvarEstimator


@pytest.fixture()
def hamiltonian():
    return PauliTerm("Z0") + PauliTerm("Z1")


@pytest.fixture()
def optimizer():
    return ScipyOptimizer(method="L-BFGS-B")


N_LAYERS = 2


@pytest.fixture()
def simulator():
    return SymbolicSimulator()


@pytest.fixture()
def initial_params(hamiltonian):
    return np.random.random(hamiltonian.n_qubits)


@pytest.fixture()
def qaoa_object(hamiltonian, optimizer):
    return QAOA.default(cost_hamiltonian=hamiltonian, n_layers=N_LAYERS)


class TestQaoa:
    def test_default_optimizer_is_lbfgsb(self, hamiltonian):
        qaoa = QAOA.default(cost_hamiltonian=hamiltonian, n_layers=N_LAYERS)
        assert qaoa.optimizer.method == "L-BFGS-B"

    def test_default_ansatz_is_farhi(self, hamiltonian):
        qaoa = QAOA.default(cost_hamiltonian=hamiltonian, n_layers=N_LAYERS)
        assert isinstance(qaoa.ansatz, QAOAFarhiAnsatz)
        assert qaoa.ansatz.number_of_layers == N_LAYERS

    def test_default_estimation_is_calculate_exact_expectation_values(
        self, hamiltonian
    ):
        qaoa = QAOA.default(
            cost_hamiltonian=hamiltonian,
            n_layers=N_LAYERS,
        )
        assert qaoa.estimation_method.__name__ == "calculate_exact_expectation_values"
        assert qaoa._n_shots is None

    def test_default_estimation_changed_to_estimate_by_averaging(self, hamiltonian):
        n_shots = 1000
        qaoa = QAOA.default(
            cost_hamiltonian=hamiltonian,
            n_layers=N_LAYERS,
            use_exact_expectation_values=False,
            n_shots=n_shots,
        )
        assert (
            qaoa.estimation_method.__name__
            == "estimate_expectation_values_by_averaging"
        )
        assert qaoa._n_shots == n_shots

    @pytest.mark.parametrize(
        "use_exact_expectation_values,n_shots", [(True, 1000), (False, None)]
    )
    def test_default_raises_exception_for_invalid_inputs(
        self, hamiltonian, use_exact_expectation_values, n_shots
    ):
        with pytest.raises(ValueError):
            _ = QAOA.default(
                cost_hamiltonian=hamiltonian,
                n_layers=N_LAYERS,
                use_exact_expectation_values=use_exact_expectation_values,
                n_shots=n_shots,
            )

    def test_init_works(self, hamiltonian, optimizer):
        ansatz = WarmStartQAOAAnsatz(
            2, hamiltonian, thetas=np.random.random(hamiltonian.n_qubits)
        )
        estimation_method = CvarEstimator(alpha=0.5)
        n_shots = 1000
        qaoa = QAOA(
            cost_hamiltonian=hamiltonian,
            optimizer=optimizer,
            ansatz=ansatz,
            estimation_method=estimation_method,
            n_shots=n_shots,
        )
        assert qaoa.ansatz is ansatz
        assert qaoa.estimation_method is estimation_method

    def test_replace_ansatz(self, qaoa_object, hamiltonian):
        ansatz = WarmStartQAOAAnsatz(
            2, hamiltonian, thetas=np.random.random(hamiltonian.n_qubits)
        )

        new_qaoa_object = qaoa_object.replace_ansatz(ansatz)

        assert qaoa_object.ansatz is not ansatz
        assert new_qaoa_object.ansatz is ansatz

    def test_replace_optimizer(self, qaoa_object):
        optimizer = ScipyOptimizer(method="COBYLA")

        new_qaoa_object = qaoa_object.replace_optimizer(optimizer)

        assert qaoa_object.optimizer is not optimizer
        assert new_qaoa_object.optimizer is optimizer

    def test_replace_estimation_method(self, qaoa_object):
        estimation_method = CvarEstimator(alpha=0.5)
        n_shots = 1001

        new_qaoa_object = qaoa_object.replace_estimation_method(
            estimation_method, n_shots=n_shots
        )

        assert (
            qaoa_object.estimation_method.__name__
            == "calculate_exact_expectation_values"
        )
        assert qaoa_object.estimation_method is not estimation_method
        assert new_qaoa_object.estimation_method is estimation_method
        assert qaoa_object._n_shots != new_qaoa_object._n_shots

    @pytest.mark.parametrize("initial_params", [None, np.random.random(4)])
    def test_find_optimal_params_doesnt_fail(
        self, qaoa_object, simulator, initial_params
    ):
        results = qaoa_object.find_optimal_params(simulator, initial_params)
        assert len(results.opt_params) == qaoa_object.ansatz.number_of_params

    def test_get_cost_function(self, qaoa_object, simulator):
        cost_function = qaoa_object.get_cost_function(simulator)
        assert np.isclose(
            cost_function(np.zeros(qaoa_object.ansatz.number_of_params)), 0
        )

    def test_get_circuit(self, qaoa_object):
        params = np.random.random(qaoa_object.ansatz.number_of_params)
        circuit = qaoa_object.get_circuit(params)
        assert circuit.n_qubits == qaoa_object.ansatz.number_of_qubits
