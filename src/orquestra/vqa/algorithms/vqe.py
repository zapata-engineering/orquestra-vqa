from functools import partial
from typing import List, Optional, cast

import numpy as np
from orquestra.opt.api import CostFunction, Optimizer
from orquestra.opt.optimizers import ScipyOptimizer
from orquestra.quantum.api.backend import QuantumBackend
from orquestra.quantum.api.estimation import (
    EstimateExpectationValues,
    EstimationPreprocessor,
)
from orquestra.quantum.circuits import Circuit
from orquestra.quantum.estimation import (
    calculate_exact_expectation_values,
    estimate_expectation_values_by_averaging,
)
from orquestra.quantum.operators import PauliRepresentation
from scipy.optimize import OptimizeResult

from orquestra.vqa.api.ansatz import Ansatz
from orquestra.vqa.cost_function.cost_function import (
    create_cost_function,
    substitution_based_estimation_tasks_factory,
)
from orquestra.vqa.estimation.context_selection import perform_context_selection
from orquestra.vqa.grouping import group_greedily, group_individually
from orquestra.vqa.shot_allocation import (
    allocate_shots_proportionally,
    allocate_shots_uniformly,
)

# https://github.com/zapatacomputing/orquestra-core/blob/main/tests/fh_vqe_test.py


class VQE:
    def __init__(
        self,
        hamiltonian: PauliRepresentation,
        optimizer: Optimizer,
        ansatz: Ansatz,
        estimation_method: EstimateExpectationValues,
        grouping: EstimationPreprocessor,
        shots_allocation: EstimationPreprocessor,
        n_shots: Optional[int] = None,
    ) -> None:
        """Class providing an easy interface to work with VQE.

        For new users, usage of "default" method is recommended.

        Args:
            hamiltonian: Cost Hamiltonian defining the problem.
            optimizer: Optimizer used to find optimal parameters
            ansatz: Ansatz defining what circuit will be used.
            estimation_method: Method used for calculating expectation values of
                the Hamiltonian.
            n_shots: number of shots to be used for evaluation of expectation values.
                For simulation with exact expectation value it should be None.
        """
        self.hamiltonian = hamiltonian
        self.optimizer = optimizer
        self.ansatz = ansatz
        self.estimation_method = estimation_method
        self._n_shots = n_shots
        self.grouping = grouping
        self.shots_allocation = shots_allocation

    @classmethod
    def default(
        cls,
        hamiltonian: PauliRepresentation,
        ansatz: Ansatz,
        use_exact_expectation_values: bool = True,
        grouping: str = "greedy",
        shots_allocation: str = "proportional",
        n_shots: Optional[int] = None,
    ) -> "VQE":
        """Creates a VQE object with some default settings:
              #TODO: use the VQE paper

        - optimizer: L-BFGS-B optimizer (scipy implementation)

        - ansatz: ansatz used for
        - estimation method: either using exact expectation values
            (only for simulation) or standard method of calculating expectation
            values through averaging the results of measurements.

        These can be later replaced using one of the `replace_*` methods.

        Args:
            hamiltonian: Cost Hamiltonian defining the problem.
            use_exact_expectation_values: A flag indicating whether to use exact
                calculation of the expectation values. This is possible only when
                running on a simulator. Defaults to True.
            n_shots: If non-exact method for calculating expectation values is used,
                this argument specifies number of shots per expectation value.

        Raises:
            ValueError: if wrong combination of "use_exact_expectation_values" and
                "n_shots" is provided.

        """

        grouping_object = _get_grouping(grouping)
        shots_allocation_object = _get_shots_allocation(shots_allocation)

        optimizer = ScipyOptimizer(method="L-BFGS-B")

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

        return cls(
            hamiltonian,
            optimizer,
            ansatz,
            estimation_method,
            grouping_object,
            shots_allocation_object,
            n_shots,
        )

    def replace_optimizer(self, optimizer: Optimizer) -> "VQE":
        """Creates a new VQE object with a provided optimizer.

        Args:
            optimizer: new optimizer to be used.
        """
        return VQE(
            self.hamiltonian,
            optimizer,
            self.ansatz,
            self.estimation_method,
            self.grouping,
            self.shots_allocation,
            self._n_shots,
        )

    def replace_hamiltonian(self, hamiltonian: PauliRepresentation) -> "VQE":
        """Creates a new VQE object with a provided hamiltonian

        Args:
            hamiltonioan: new hamiltonian to be used
        """

        return VQE(
            hamiltonian,
            self.optimizer,
            self.ansatz,
            self.estimation_method,
            self.grouping,
            self.shots_allocation,
            self._n_shots,
        )

    def replace_ansatz(self, ansatz: Ansatz) -> "VQE":
        """Creates a new VQE object with a provided ansatz.

        Args:
            ansatz: new ansatz to be used.
        """
        return VQE(
            self.hamiltonian,
            self.optimizer,
            ansatz,
            self.estimation_method,
            self.grouping,
            self.shots_allocation,
            self._n_shots,
        )

    def replace_estimation_method(
        self, estimation_method: EstimateExpectationValues, n_shots: Optional[int]
    ) -> "VQE":
        """Creates a new VQE object with a provided estimation method.

        It requires providing both new estimation method and number of shots.

        Args:
            estimation_method: new estimation method to be used.
            n_shots: number of shots for the new estimation method.
        """
        return VQE(
            self.hamiltonian,
            self.optimizer,
            self.ansatz,
            estimation_method,
            self.grouping,
            self.shots_allocation,
            n_shots,
        )

    def replace_grouping(self, grouping: EstimationPreprocessor) -> "VQE":
        # TODO: redo docstrings
        """Creates a new VQE object with a provided estimation method.

        It requires providing both new estimation method and number of shots.

        Args:
            estimation_method: new estimation method to be used.
            n_shots: number of shots for the new estimation method.
        """
        return VQE(
            self.hamiltonian,
            self.optimizer,
            self.ansatz,
            self.estimation_method,
            grouping,
            self.shots_allocation,
            self._n_shots,
        )

    def replace_shots_allocation(
        self, shots_allocation: EstimationPreprocessor, n_shots: int
    ) -> "VQE":
        # TODO: redo docstrings
        """Creates a new VQE object with a provided estimation method.

        It requires providing both new estimation method and number of shots.

        Args:
            estimation_method: new estimation method to be used.
            n_shots: number of shots for the new estimation method.
        """
        # QUESTION: shots allocation and number of shots

        return VQE(
            self.hamiltonian,
            self.optimizer,
            self.ansatz,
            self.estimation_method,
            self.grouping,
            shots_allocation,
            self._n_shots,
        )

    def find_optimal_params(
        self, backend: QuantumBackend, initial_params: Optional[np.ndarray] = None
    ) -> OptimizeResult:
        """Optimizes the paramaters of VQE ansatz using provided backend.

        Args:
            backend: backend used for running quantum circuits.
            initial_params: Initial parameters for the optimizer. If None provided,
                will create random parameters from [0, pi]. Defaults to None.
        """
        cost_function = self.get_cost_function(backend)
        if initial_params is None:
            initial_params = np.random.random(self.ansatz.number_of_params) * np.pi

        return self.optimizer.minimize(cost_function, initial_params)

    def get_cost_function(self, backend: QuantumBackend) -> CostFunction:
        """Returns cost function associated with given VQE instance.

        Args:
            backend: backend used for running quantum circuits.
        """

        shots_allocation = partial(self.shots_allocation, self._n_shots)

        if self._n_shots is None:
            estimation_preprocessors = []
        else:

            estimation_preprocessors = [
                self.grouping,
                perform_context_selection,
                shots_allocation,
            ]

        estimation_task_factory = substitution_based_estimation_tasks_factory(
            self.hamiltonian, self.ansatz, estimation_preprocessors
        )

        return create_cost_function(
            backend,
            estimation_task_factory,
            self.estimation_method,
        )

    def get_circuit(self, params: np.ndarray) -> Circuit:
        """Returns a circuit associated with give VQE instance.

        Args:
            params: ansatz parameters.
        """
        return self.ansatz.get_executable_circuit(params)


def _get_grouping(grouping) -> EstimationPreprocessor:
    if grouping is not None:
        if grouping == "greedy":
            grouping = group_greedily
        elif grouping == "individual":
            grouping = group_individually
        else:
            raise ValueError(
                'Grouping provided is not recognized by the "default" method'
                "For custom grouping, please use VQE class init method"
                'instead of "default" method'
            )

    return grouping


def _get_shots_allocation(shot_allocation) -> EstimationPreprocessor:
    if shot_allocation == "proportional":
        shot_allocation = allocate_shots_proportionally
    elif shot_allocation == "uniform":
        shot_allocation == allocate_shots_uniformly
    else:
        raise ValueError(
            "Only proportional and uniform shots allocations are accepted"
            "For using custom shots allocation, please use VQE class init method"
            'instead of "default" method'
        )
    return shot_allocation
