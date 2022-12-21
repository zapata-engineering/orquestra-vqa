import numpy as np
import pytest
from orquestra.opt.optimizers import ScipyOptimizer
from orquestra.quantum.operators import PauliTerm
from orquestra.quantum.runners import SymbolicSimulator

from orquestra.vqa.algorithms import VQE
from orquestra.vqa.ansatz import HEAQuantumCompilingAnsatz
from orquestra.vqa.estimation import CvarEstimator
from orquestra.vqa.grouping import group_greedily, group_individually
from orquestra.vqa.shot_allocation import (
    allocate_shots_proportionally,
    allocate_shots_uniformly,
)


@pytest.fixture()
def hamiltonian():
    return PauliTerm("Z0") * PauliTerm("Z0") + PauliTerm("X0") + PauliTerm("X1")


@pytest.fixture()
def optimizer():
    return ScipyOptimizer(method="L-BFGS-B")


@pytest.fixture()
def simulator():
    return SymbolicSimulator()


@pytest.fixture()
def ansatz(hamiltonian):
    return HEAQuantumCompilingAnsatz(1, hamiltonian.n_qubits)


@pytest.fixture()
def initial_params(hamiltonian):
    return np.random.random(hamiltonian.n_qubits)


@pytest.fixture()
def vqe_object(hamiltonian, ansatz):
    return VQE.default(hamiltonian=hamiltonian, ansatz=ansatz)


@pytest.fixture()
def vqe_object_with_shots(hamiltonian, ansatz):
    return VQE.default(
        hamiltonian=hamiltonian,
        ansatz=ansatz,
        use_exact_expectation_values=False,
        grouping="greedy",
        n_shots=1000,
    )


class TestDefaultVQE:
    def test_default_optimizer_is_lbfgsb(self, vqe_object):
        assert vqe_object.optimizer.method == "L-BFGS-B"

    def test_default_estimation_is_calculate_exact_expectation_values(self, vqe_object):
        assert (
            vqe_object.estimation_method.__name__
            == "calculate_exact_expectation_values"
        )
        assert vqe_object._n_shots is None

    def test_default_estimation_changed_to_estimate_by_averaging(self, hamiltonian):
        n_shots = 1000
        vqe = VQE.default(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            use_exact_expectation_values=False,
            n_shots=n_shots,
        )
        assert (
            vqe.estimation_method.__name__ == "estimate_expectation_values_by_averaging"
        )
        assert vqe._n_shots == n_shots

    def test_default_grouping_is_None(self, hamiltonian, ansatz):
        vqe = VQE.default(hamiltonian=hamiltonian, ansatz=ansatz)
        assert vqe.grouping is None

    def test_default_shots_allocation_is_proportional(self, hamiltonian, ansatz):
        vqe = VQE.default(hamiltonian=hamiltonian, ansatz=ansatz)
        assert vqe.shots_allocation.__name__ == "allocate_shots_proportionally"

    @pytest.mark.parametrize(
        "use_exact_expectation_values,n_shots", [(True, 1000), (False, None)]
    )
    def test_default_raises_exception_for_invalid_inputs(
        self, hamiltonian, ansatz, use_exact_expectation_values, n_shots
    ):
        with pytest.raises(ValueError):
            _ = VQE.default(
                hamiltonian=hamiltonian,
                ansatz=ansatz,
                use_exact_expectation_values=use_exact_expectation_values,
                n_shots=n_shots,
            )

    def test_init_works(self, hamiltonian, optimizer, ansatz):
        estimation_method = CvarEstimator(alpha=0.5)
        grouping = group_greedily
        shots_allocation = allocate_shots_proportionally
        n_shots = 1000
        vqe = VQE(
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            ansatz=ansatz,
            estimation_method=estimation_method,
            grouping=grouping,
            shots_allocation=shots_allocation,
            n_shots=n_shots,
        )
        assert vqe.ansatz is ansatz
        assert vqe.estimation_method is estimation_method
        assert vqe.shots_allocation is shots_allocation
        assert vqe.grouping is grouping
        assert vqe._n_shots == 1000

    def test_replace_ansatz(self, vqe_object, hamiltonian):
        ansatz_2_layer = HEAQuantumCompilingAnsatz(2, hamiltonian.n_qubits)

        new_vqe_object = vqe_object.replace_ansatz(ansatz)

        assert vqe_object.ansatz is not ansatz_2_layer
        assert new_vqe_object.ansatz is ansatz

    def test_replace_optimizer(self, vqe_object):
        optimizer = ScipyOptimizer(method="COBYLA")

        new_vqe_object = vqe_object.replace_optimizer(optimizer)

        assert vqe_object.optimizer is not optimizer
        assert new_vqe_object.optimizer is optimizer

    def test_replace_estimation_method(self, vqe_object):
        estimation_method = CvarEstimator(alpha=0.5)
        n_shots = 1001

        new_vqe_object = vqe_object.replace_estimation_method(
            estimation_method, n_shots=n_shots
        )

        assert (
            vqe_object.estimation_method.__name__
            == "calculate_exact_expectation_values"
        )
        assert vqe_object.estimation_method is not estimation_method
        assert new_vqe_object.estimation_method is estimation_method
        assert vqe_object._n_shots != new_vqe_object._n_shots

    def test_replace_grouping(self, vqe_object):
        grouping = group_individually

        new_vqe_object = vqe_object.replace_grouping(grouping)

        assert vqe_object.grouping is not grouping
        assert new_vqe_object.grouping is grouping

    def test_replace_shots_allocation(self, vqe_object):
        shots_allocation = allocate_shots_uniformly
        new_vqe_object = vqe_object.replace_shots_allocation(shots_allocation, 1000)

        assert vqe_object.shots_allocation is not shots_allocation
        assert new_vqe_object.shots_allocation is shots_allocation

    @pytest.mark.parametrize("initial_params", [None, np.random.random(12)])
    def test_find_optimal_params_doesnt_fail(
        self, vqe_object, simulator, initial_params
    ):
        results = vqe_object.find_optimal_params(simulator, initial_params)
        assert len(results.opt_params) == vqe_object.ansatz.number_of_params

    def test_get_cost_function(self, vqe_object, simulator):
        cost_function = vqe_object.get_cost_function(simulator)
        assert np.isclose(
            cost_function(np.zeros(vqe_object.ansatz.number_of_params)), 1
        )

    def test_get_cost_function_with_shots(self, vqe_object_with_shots, simulator):
        vqe_object_with_shots.get_cost_function(simulator)

    def test_get_circuit(self, vqe_object):
        params = np.random.random(vqe_object.ansatz.number_of_params)
        circuit = vqe_object.get_circuit(params)
        assert circuit.n_qubits == vqe_object.ansatz.number_of_qubits
