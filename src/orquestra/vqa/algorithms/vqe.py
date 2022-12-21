from functools import partial
from typing import List, Optional, cast
from warnings import warn

import numpy as np
from orquestra.opt.api import CostFunction, Optimizer
from orquestra.opt.optimizers import ScipyOptimizer
from orquestra.quantum.api.circuit_runner import CircuitRunner
from orquestra.quantum.api.estimation import (
    EstimateExpectationValues,
    EstimationPreprocessor,
    EstimationTask,
)
from orquestra.quantum.circuits import Circuit
from orquestra.quantum.estimation import (
    calculate_exact_expectation_values,
    estimate_expectation_values_by_averaging,
)
from orquestra.quantum.operators import PauliRepresentation
from scipy.optimize import OptimizeResult

from orquestra.vqa.api.ansatz import Ansatz
from orquestra.vqa.cost_function import (
    create_cost_function,
    substitution_based_estimation_tasks_factory,
)
from orquestra.vqa.estimation.context_selection import perform_context_selection
from orquestra.vqa.grouping import group_greedily, group_individually
from orquestra.vqa.shot_allocation import (
    allocate_shots_proportionally,
    allocate_shots_uniformly,
)


class VQE:
    def __init__(
        self,
        hamiltonian: PauliRepresentation,
        optimizer: Optimizer,
        ansatz: Ansatz,
        estimation_method: EstimateExpectationValues,
        grouping: Optional[EstimationPreprocessor],
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
            grouping: Transforms list of estimation tasks by grouping and adding
            context selection logic to the circuits
            shots_allocation: Function allocate the shots for each task
            n_shots: Number of shots to be used for evaluation of expectation values.
                For simulation with exact expectation value it should be None.
        """
        self.hamiltonian = hamiltonian
        self.optimizer = optimizer
        self.ansatz = ansatz
        self.estimation_method = estimation_method
        self.grouping = grouping
        self.shots_allocation = shots_allocation
        self._n_shots = n_shots

    @classmethod
    def default(
        cls,
        hamiltonian: PauliRepresentation,
        ansatz: Ansatz,
        use_exact_expectation_values: bool = True,
        grouping: Optional[str] = None,
        shots_allocation: str = "proportional",
        n_shots: Optional[int] = None,
    ) -> "VQE":
        """Creates a VQE object with some default settings:

            - optimizer: L-BFGS-B optimizer (scipy implementation)
            - estimation method: either using exact expectation values
                (only for simulation) or standard method of calculating expectation
                values through averaging the results of measurements.
            - grouping: The default value is None, therefore everything is co-measurable
            - shots_allocation: proportional to the weights of the Pauli terms

        These can be later replaced using one of the `replace_*` methods.

        Args:
            hamiltonian: Cost Hamiltonian defining the problem.
            use_exact_expectation_values: A flag indicating whether to use exact
                calculation of the expectation values. This is possible only when
                running on a simulator. Defaults to True.
            grouping: name of the grouping function provided as a string. It only
            accepts "greedy", "individual" or None as an argument.
            shots_allocation: name of the shots allocation function provided as a
            string. It only accepts "proportional" and "individual" as an argument.
            n_shots: Specifies number of shots to be used for a given shots allocation
            method. If exact_expectation_values is true, it should be equal to None.

        Raises:
            ValueError: if wrong combination of "use_exact_expectation_values" and
                "n_shots" is provided.

        """

        optimizer = ScipyOptimizer(method="L-BFGS-B")

        if use_exact_expectation_values and n_shots is None:
            estimation_method = cast(
                EstimateExpectationValues, calculate_exact_expectation_values
            )
            if grouping is not None:
                warn(
                    "Since we are using use_exact expectation values,"
                    "grouping is changed to None"
                )
                grouping = None
        elif not use_exact_expectation_values and n_shots is not None:
            estimation_method = estimate_expectation_values_by_averaging
        else:
            raise ValueError(
                f"Invalid n_shots={n_shots} for \
                    use_exact_expectation_values={use_exact_expectation_values}."
            )

        grouping_object = _get_grouping(grouping)
        shots_allocation_object = _get_shots_allocation(shots_allocation)

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
        """Creates a new VQE object with a provided grouping method.

        Args:
            grouping: new grouping method to be used.
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

        """Creates a new VQE object with a provided shots allocation.

        It requires providing both new estimation method and number of shots.

        Args:
            shots_allocation: new estimation method to be used.
            n_shots: number of shots for the new estimation method.
        """

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
        self, runner: CircuitRunner, initial_params: Optional[np.ndarray] = None
    ) -> OptimizeResult:
        """Optimizes the paramaters of VQE ansatz using provided runner.

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
        """Returns cost function associated with given VQE instance.

        Args:
            runner: runner used for running quantum circuits.
        """

        def shots_allocation(estimation_tasks):
            return self.shots_allocation(estimation_tasks, self._n_shots)

        estimation_preprocessors: List[EstimationPreprocessor] = []

        if self._n_shots is not None:

            estimation_preprocessors = [
                self.grouping,  # type: ignore
                perform_context_selection,
                shots_allocation,
            ]

        estimation_task_factory = substitution_based_estimation_tasks_factory(
            self.hamiltonian, self.ansatz, estimation_preprocessors
        )

        return create_cost_function(
            runner,
            estimation_task_factory,
            self.estimation_method,
        )

    def get_circuit(self, params: np.ndarray) -> Circuit:
        """Returns a circuit associated with give VQE instance.

        Args:
            params: ansatz parameters.
        """
        return self.ansatz.get_executable_circuit(params)

    @property
    def n_qubits(self):
        return self.ansatz.number_of_qubits


def _get_grouping(
    grouping: Optional[str] = None,
) -> Optional[EstimationPreprocessor]:
    if grouping is not None:
        if grouping == "greedy":
            grouping_object = group_greedily
        elif grouping == "individual":
            grouping_object = group_individually  # type: ignore
        else:
            raise ValueError(
                'Grouping provided is not recognized by the "default" method'
                "For custom grouping, please use VQE class init method"
                'instead of "default" method'
            )
    else:
        grouping_object = None

    return grouping_object


def _get_shots_allocation(
    shot_allocation: str,
) -> EstimationPreprocessor:
    if shot_allocation == "proportional":
        shot_allocation_object = allocate_shots_proportionally
    elif shot_allocation == "uniform":
        shot_allocation_object = allocate_shots_uniformly  # type: ignore
    else:
        raise ValueError(
            "Only proportional and uniform shots allocations are accepted"
            "For using custom shots allocation, please use VQE class init method"
            'instead of "default" method'
        )
    return shot_allocation_object  # type: ignore
