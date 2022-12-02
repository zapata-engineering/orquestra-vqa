from numbers import Number
from typing import Callable, Optional, cast

import numpy as np
from orquestra.opt.api import CostFunction, Optimizer
from orquestra.opt.optimizers import ScipyOptimizer
from orquestra.quantum.api.circuit_runner import CircuitRunner
from orquestra.quantum.api.estimation import EstimateExpectationValues
from orquestra.quantum.circuits import Circuit
from orquestra.quantum.distributions import (
    MeasurementOutcomeDistribution,
    compute_clipped_negative_log_likelihood,
)
from orquestra.quantum.estimation import (
    calculate_exact_expectation_values,
    estimate_expectation_values_by_averaging,
)
from scipy.optimize import OptimizeResult

from orquestra.vqa.ansatz import QCBMAnsatz
from orquestra.vqa.cost_function.qcbm_cost_function import create_QCBM_cost_function


class QCBM:
    def __init__(
        self,
        target_distribution: MeasurementOutcomeDistribution,
        n_layers: int,
        optimizer: Optimizer,
        estimation_method: EstimateExpectationValues,
        n_shots: Optional[int] = None,
    ) -> None:
        """Class providing an easy interface to work with Quantum Circuit Born Machine (QCBM).

        For new users, usage of "default" method is recommended.

        Args:
            target_distribution: bitstring distribution which QCBM aims to learn
            n_layers: Number of layers for the ansatz.
            optimizer: Optimizer used to find optimal parameters
            estimation_method: Method used for calculating expectation values of
                the Hamiltonian.
            n_shots: number of shots to be used for evaluation of expectation values.
                For simulation with exact expectation value it should be None.
        """

        self.target_distribution = target_distribution
        self._n_layers = n_layers
        self.optimizer = optimizer
        self.estimation_method = estimation_method
        self._n_shots = n_shots

    @classmethod
    def default(
        cls,
        target_distribution: MeasurementOutcomeDistribution,
        n_layers: int,
        use_exact_expectation_values: bool = True,
        n_shots: Optional[int] = None,
    ) -> "QCBM":
        """Creates a QCBM object with some default settings:

        - optimizer: L-BFGS-B optimizer (scipy implementation)
        - estimation method: either using exact expectation values
            (only for simulation) or standard method of calculating expectation
            values through averaging the results of measurements.
        - topology: topology for ansatz is set to 'all'

        These can be later replaced using one of the `replace_*` methods.

        Args:
            target_distribution: bitstring distribution which QCBM aims to learn
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
            target_distribution,
            n_layers,
            optimizer,
            estimation_method,
            n_shots,
        )

    def replace_n_layers(self, n_layers: int) -> "QCBM":
        """Creates a new QCBM object with a provided number of layers.

        Args:
            n_layers: new number of layers to be used.
        """
        return QCBM(
            self.target_distribution,
            n_layers,
            self.optimizer,
            self.estimation_method,
            self._n_shots,
        )

    def replace_optimizer(self, optimizer: Optimizer) -> "QCBM":
        """Creates a new QCBM object with a provided optimizer.

        Args:
            optimizer: new optimizer to be used.
        """
        return QCBM(
            self.target_distribution,
            self._n_layers,
            optimizer,
            self.estimation_method,
            self._n_shots,
        )

    def replace_estimation_method(
        self, estimation_method: EstimateExpectationValues, n_shots: Optional[int]
    ) -> "QCBM":
        """Creates a new QCBM object with a provided estimation method.

        It requires providing both new estimation method and number of shots.

        Args:
            estimation_method: new estimation method to be used.
            n_shots: number of shots for the new estimation method.
        """
        return QCBM(
            self.target_distribution,
            self._n_layers,
            self.optimizer,
            estimation_method,
            n_shots,
        )

    def find_optimal_params(
        self, runner: CircuitRunner, initial_params: Optional[np.ndarray] = None
    ) -> OptimizeResult:
        """Optimizes the paramaters of QCBM ansatz using provided runner.

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
        """Returns cost function associated with given QCBM instance.

        Args:
            runner: runner used for running quantum circuits.
        """

        return create_QCBM_cost_function(
            ansatz=self.ansatz,
            runner=runner,
            n_samples=self._n_shots,  # type: ignore
            distance_measure=cast(
                Callable[..., Number], compute_clipped_negative_log_likelihood
            ),
            distance_measure_parameters={"epsilon": 1e-6},
            target_distribution=self.target_distribution,
        )

    def get_circuit(self, params: np.ndarray) -> Circuit:
        """Returns a circuit associated with give QCBM instance.

        Args:
            params: ansatz parameters.
        """
        return self.ansatz.get_executable_circuit(params)

    @property
    def n_qubits(self):
        key_value = [key for key in self.target_distribution.distribution_dict.keys()][
            0
        ]
        return len(key_value)

    @property
    def ansatz(self):
        return QCBMAnsatz(self._n_layers, self.n_qubits, "all")
