import numpy as np
import pytest
from orquestra.opt.optimizers import ScipyOptimizer
from orquestra.quantum.backends.symbolic_simulator import SymbolicSimulator
from orquestra.quantum.operators import PauliTerm

from orquestra.vqa.algorithms.qaoa import QAOA
from orquestra.vqa.ansatz.qaoa_farhi import QAOAFarhiAnsatz
from orquestra.vqa.ansatz.qaoa_warm_start import WarmStartQAOAAnsatz
from orquestra.vqa.estimation.cvar import CvarEstimator

# def test_if_runs():
#     hamiltonian = PauliTerm("Z0") + PauliTerm("Z1")
#     optimizer = ScipyOptimizer(method="L-BFGS-B")
#     backend = SymbolicSimulator()
#     qaoa = QAOA.default(cost_hamiltonian=hamiltonian, optimizer=optimizer, n_layers=2)
#     results = qaoa.find_optimal_params(backend=backend)
#     breakpoint()


@pytest.fixture()
def hamiltonian():
    return PauliTerm("Z0") + PauliTerm("Z1")


@pytest.fixture()
def optimizer():
    return ScipyOptimizer(method="L-BFGS-B")


@pytest.fixture()
def n_layers():
    return 2


@pytest.fixture()
def backend():
    return SymbolicSimulator()


@pytest.fixture()
def initial_params(hamiltonian):
    return np.random.random(hamiltonian.n_qubits)


@pytest.fixture()
def qaoa_object(hamiltonian, optimizer, n_layers):
    return QAOA.default(cost_hamiltonian=hamiltonian, n_layers=n_layers)


class TestQaoa:
    def test_default_optimizer_init(self, hamiltonian, n_layers):
        qaoa = QAOA.default(cost_hamiltonian=hamiltonian, n_layers=n_layers)
        assert qaoa.optimizer.method == "L-BFGS-B"

    def test_default_ansatz_init(self, hamiltonian, n_layers):
        qaoa = QAOA.default(cost_hamiltonian=hamiltonian, n_layers=n_layers)
        assert isinstance(qaoa.ansatz, QAOAFarhiAnsatz)
        assert qaoa.ansatz.number_of_layers == n_layers

    def test_default_estimation_init_default_value(self, hamiltonian, n_layers):
        qaoa = QAOA.default(
            cost_hamiltonian=hamiltonian,
            n_layers=n_layers,
        )
        assert qaoa.estimation_method.__name__ == "calculate_exact_expectation_values"
        assert qaoa._n_shots is None

    def test_default_estimation_init_exact(self, hamiltonian, n_layers):
        qaoa = QAOA.default(
            cost_hamiltonian=hamiltonian,
            n_layers=n_layers,
            use_exact_expectation_values=True,
        )
        assert qaoa.estimation_method.__name__ == "calculate_exact_expectation_values"
        assert qaoa._n_shots is None

    def test_default_estimation_init_averaging(self, hamiltonian, n_layers):
        n_shots = 1000
        qaoa = QAOA.default(
            cost_hamiltonian=hamiltonian,
            n_layers=n_layers,
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
        self, hamiltonian, n_layers, use_exact_expectation_values, n_shots
    ):
        with pytest.raises(ValueError):
            _ = QAOA.default(
                cost_hamiltonian=hamiltonian,
                n_layers=n_layers,
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
        assert isinstance(qaoa.ansatz, WarmStartQAOAAnsatz)
        assert isinstance(qaoa.estimation_method, CvarEstimator)

    def test_replace_ansatz(self, qaoa_object, hamiltonian):
        ansatz = WarmStartQAOAAnsatz(
            2, hamiltonian, thetas=np.random.random(hamiltonian.n_qubits)
        )

        new_qaoa_object = qaoa_object.replace_ansatz(ansatz)

        assert isinstance(qaoa_object.ansatz, QAOAFarhiAnsatz)
        assert isinstance(new_qaoa_object.ansatz, WarmStartQAOAAnsatz)

    def test_replace_optimizer(self, qaoa_object):
        optimizer = ScipyOptimizer(method="COBYLA")

        new_qaoa_object = qaoa_object.replace_optimizer(optimizer)

        assert qaoa_object.optimizer.method == "L-BFGS-B"
        assert new_qaoa_object.optimizer.method == "COBYLA"

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
        assert isinstance(new_qaoa_object.estimation_method, CvarEstimator)
        assert qaoa_object._n_shots != new_qaoa_object._n_shots

    @pytest.mark.parametrize("initial_params", [None, np.random.random(4)])
    def test_find_optimal_params_doesnt_fail(
        self, qaoa_object, backend, initial_params
    ):
        results = qaoa_object.find_optimal_params(backend, initial_params)
        assert len(results.opt_params) == qaoa_object.ansatz.number_of_params

    def test_get_cost_function(self, qaoa_object, backend):
        cost_function = qaoa_object.get_cost_function(backend)
        assert np.isclose(
            cost_function(np.zeros(qaoa_object.ansatz.number_of_params)), 0
        )

    def test_get_circuit(self, qaoa_object):
        params = np.random.random(qaoa_object.ansatz.number_of_params)
        circuit = qaoa_object.get_circuit(params)
        assert circuit.n_qubits == qaoa_object.ansatz.number_of_qubits
