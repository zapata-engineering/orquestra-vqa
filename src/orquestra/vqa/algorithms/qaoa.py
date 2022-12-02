from functools import partial
from typing import List, Optional, cast

import numpy as np
from orquestra.opt.api import CostFunction, Optimizer
from orquestra.opt.optimizers import ScipyOptimizer
from orquestra.quantum.api.circuit_runner import CircuitRunner
from orquestra.quantum.api.estimation import EstimateExpectationValues, EstimationTask
from orquestra.quantum.circuits import Circuit
from orquestra.quantum.estimation import (
    calculate_exact_expectation_values,
    estimate_expectation_values_by_averaging,
)
from orquestra.quantum.operators import PauliRepresentation
from scipy.optimize import OptimizeResult

from orquestra.vqa.ansatz import QAOAFarhiAnsatz
from orquestra.vqa.api.ansatz import Ansatz
from orquestra.vqa.cost_function import (
    create_cost_function,
    substitution_based_estimation_tasks_factory,
)
from orquestra.vqa.shot_allocation import allocate_shots_uniformly


class QAOA:
    def __init__(
        self,
        cost_hamiltonian: PauliRepresentation,
        optimizer: Optimizer,
        ansatz: Ansatz,
        estimation_method: EstimateExpectationValues,
        n_shots: Optional[int] = None,
    ) -> None:
        """Class providing an easy interface to work with QAOA.

        For new users, usage of "default" method is recommended.

        Args:
            cost_hamiltonian: Cost Hamiltonian defining the problem.
            optimizer: Optimizer used to find optimal parameters
            ansatz: Ansatz defining what circuit will be used.
            estimation_method: Method used for calculating expectation values of
                the Hamiltonian.
            n_shots: number of shots to be used for evaluation of expectation values.
                For simulation with exact expectation value it should be None.
        """
        self.cost_hamiltonian = cost_hamiltonian
        self.optimizer = optimizer
        self.ansatz = ansatz
        self.estimation_method = estimation_method
        self._n_shots = n_shots

    @classmethod
    def default(
        cls,
        cost_hamiltonian: PauliRepresentation,
        n_layers: int,
        use_exact_expectation_values: bool = True,
        n_shots: Optional[int] = None,
    ) -> "QAOA":
        """Creates a QAOA object with some default settings:

        - optimizer: L-BFGS-B optimizer (scipy implementation)
        - ansatz: standard ansatz as proposed by Farhi
            in https://arxiv.org/abs/1411.4028.
        - estimation method: either using exact expectation values
            (only for simulation) or standard method of calculating expectation
            values through averaging the results of measurements.

        These can be later replaced using one of the `replace_*` methods.

        Args:
            cost_hamiltonian: Cost Hamiltonian defining the problem.
            n_layers: Number of layers for the ansatz.
            use_exact_expectation_values: A flag indicating whether to use exact
                calculation of the expectation values. This is possible only when
                running on a simulator. Defaults to True.
            n_shots: If non-exact method for calculating expectation values is used,
                this argument specifies number of shots per expectation value.

        Raises:
            ValueError: if wrong combination of "use_exact_expectation_values" and
                "n_shots" is provided.

        """
        optimizer = ScipyOptimizer(method="L-BFGS-B")
        ansatz = QAOAFarhiAnsatz(
            number_of_layers=n_layers,
            cost_hamiltonian=cost_hamiltonian,
        )
        if use_exact_expectation_values and n_shots is None:
            estimation_method = cast(
                EstimateExpectationValues, calculate_exact_expectation_values
            )
        elif not use_exact_expectation_values and n_shots is not None:
            estimation_method = estimate_expectation_values_by_averaging
        else:
            raise ValueError(
                f"Invalid n_shots={n_shots} for \
                    use_exact_expectation_values={use_exact_expectation_values}."
            )

        return cls(cost_hamiltonian, optimizer, ansatz, estimation_method, n_shots)

    def replace_optimizer(self, optimizer: Optimizer) -> "QAOA":
        """Creates a new QAOA object with a provided optimizer.

        Args:
            optimizer: new optimizer to be used.
        """
        return QAOA(
            self.cost_hamiltonian,
            optimizer,
            self.ansatz,
            self.estimation_method,
            self._n_shots,
        )

    def replace_ansatz(self, ansatz: Ansatz) -> "QAOA":
        """Creates a new QAOA object with a provided ansatz.

        Args:
            ansatz: new ansatz to be used.
        """
        return QAOA(
            self.cost_hamiltonian,
            self.optimizer,
            ansatz,
            self.estimation_method,
            self._n_shots,
        )

    def replace_estimation_method(
        self, estimation_method: EstimateExpectationValues, n_shots: Optional[int]
    ) -> "QAOA":
        """Creates a new QAOA object with a provided estimation method.

        It requires providing both new estimation method and number of shots.

        Args:
            estimation_method: new estimation method to be used.
            n_shots: number of shots for the new estimation method.
        """
        return QAOA(
            self.cost_hamiltonian,
            self.optimizer,
            self.ansatz,
            estimation_method,
            n_shots,
        )

    def find_optimal_params(
        self, runner: CircuitRunner, initial_params: Optional[np.ndarray] = None
    ) -> OptimizeResult:
        """Optimizes the paramaters of QAOA ansatz using provided runner.

        Args:
            runner: runner used for running quantum circuits.
            initial_params: Initial parameters for the optimizer. If None provided,
                will create random parameters from [0, pi]. Defaults to None.
        """
        cost_function = self.get_cost_function(runner)
        if initial_params is None:
            initial_params = np.random.random(self.ansatz.number_of_params) * np.pi

        return self.optimizer.minimize(cost_function, initial_params)

    def get_cost_function(self, runner: CircuitRunner) -> CostFunction:
        """Returns cost function associated with given QAOA instance.

        Args:
            runner: runner used for running quantum circuits.
        """
        estimation_preprocessors = []
        if self._n_shots is not None:

            def shot_allocation(estimation_tasks: List[EstimationTask]):
                return allocate_shots_uniformly(
                    estimation_tasks, number_of_shots=self._n_shots  # type: ignore
                )

            estimation_preprocessors.append(shot_allocation)

        estimation_task_factory = substitution_based_estimation_tasks_factory(
            self.cost_hamiltonian, self.ansatz, estimation_preprocessors
        )

        return create_cost_function(
            runner,
            estimation_task_factory,
            self.estimation_method,
        )

    def get_circuit(self, params: np.ndarray) -> Circuit:
        """Returns a circuit associated with give QAOA instance.

        Args:
            params: ansatz parameters.
        """
        return self.ansatz.get_executable_circuit(params)

    @property
    def n_qubits(self):
        return self.ansatz.number_of_qubits
