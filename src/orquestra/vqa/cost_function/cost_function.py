################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
from typing import Callable, Iterable, List, Optional, Union

import numpy as np
import sympy
from orquestra.opt.api.cost_function import CostFunction, ParameterPreprocessor
from orquestra.opt.api.functions import function_with_gradient
from orquestra.opt.gradients import finite_differences_gradient
from orquestra.quantum.api.circuit_runner import CircuitRunner
from orquestra.quantum.api.estimation import (
    EstimateExpectationValues,
    EstimationPreprocessor,
    EstimationTask,
    EstimationTasksFactory,
)
from orquestra.quantum.circuits import Circuit
from orquestra.quantum.estimation import (
    estimate_expectation_values_by_averaging,
    evaluate_estimation_circuits,
)
from orquestra.quantum.measurements import (
    ExpectationValues,
    concatenate_expectation_values,
    expectation_values_to_real,
)
from orquestra.quantum.operators import PauliRepresentation
from orquestra.quantum.typing import SupportsLessThan
from orquestra.quantum.utils import ValueEstimate, create_symbols_map

from ..api.ansatz import Ansatz
from ..api.ansatz_utils import combine_ansatz_params

GradientFactory = Callable[[Callable], Callable[[np.ndarray], np.ndarray]]
SymbolsSortKey = Callable[[sympy.Symbol], SupportsLessThan]


def _get_sorted_set_of_circuit_symbols(
    estimation_tasks: List[EstimationTask], key: SymbolsSortKey = str
) -> List[sympy.Symbol]:

    return sorted(
        list(
            {param for task in estimation_tasks for param in task.circuit.free_symbols}
        ),
        key=key,
    )


_by_averaging = estimate_expectation_values_by_averaging


def sum_expectation_values(expectation_values: ExpectationValues) -> ValueEstimate:
    """Compute the sum of expectation values.

    If correlations are available, the precision of the sum is computed as

    \\epsilon = \\sqrt{\\sum_k \\sigma^2_k}

    where the sum runs over frames and \\sigma^2_k is the estimated variance of
    the estimated contribution of frame k to the total. This is calculated as

    \\sigma^2_k = \\sum_{i,j} Cov(o_{k,i}, o_{k, j})

    where Cov(o_{k,i}, o_{k, j}) is the estimated covariance in the estimated
    expectation values of operators i and j of frame k.

    Args:
        expectation_values: The expectation values to sum.

    Returns:
        The value of the sum, including a precision if the expectation values
            included covariances.
    """

    value = np.sum(expectation_values.values)

    precision = None

    if expectation_values.estimator_covariances:
        estimator_variance = 0.0
        for frame_covariance in expectation_values.estimator_covariances:
            estimator_variance += float(np.sum(frame_covariance, (0, 1)))
        precision = np.sqrt(estimator_variance)
    return ValueEstimate(value, precision)


def fix_parameters(fixed_parameters: np.ndarray) -> ParameterPreprocessor:
    """Preprocessor appending fixed parameters.

    Args:
        fixed_parameters: parameters to be appended to the ones being preprocessed.
    Returns:
        preprocessor
    """

    def _preprocess(parameters: np.ndarray) -> np.ndarray:
        return combine_ansatz_params(fixed_parameters, parameters)

    return _preprocess


def add_normal_noise(
    parameter_precision, parameter_precision_seed
) -> ParameterPreprocessor:
    """Preprocessor adding noise to the parameters.

    The added noise is iid normal with mean=0.0 and stdev=`parameter_precision`.

    Args:
        parameter_precision: stddev of the noise distribution
        parameter_precision_seed: seed for random number generator. The generator
          is seeded during preprocessor creation (not during each preprocessor call).
    Returns:
        preprocessor
    """
    rng = np.random.default_rng(parameter_precision_seed)

    def _preprocess(parameters: np.ndarray) -> np.ndarray:
        noise = rng.normal(0.0, parameter_precision, len(parameters))
        return parameters + noise

    return _preprocess


def create_cost_function(
    runner: CircuitRunner,
    estimation_tasks_factory: EstimationTasksFactory,
    estimation_method: EstimateExpectationValues = _by_averaging,
    parameter_preprocessors: Optional[Iterable[ParameterPreprocessor]] = None,
    gradient_function: GradientFactory = finite_differences_gradient,
) -> CostFunction:
    """This function can be used to generate callable cost functions for parametric
    circuits. This function is the main entry to use other functions in this module.

    Args:
        runner: quantum runner used for evaluation.
        estimation_tasks_factory: function that produces estimation tasks from
            parameters. See example use case below for clarification.
        estimation_method: the estimator used to compute expectation value of target
            operator.
        parameter_preprocessors: a list of callable functions that are applied to
            parameters prior to estimation task evaluation. These functions have to
            adhere to the ParameterPreprocessor protocol.
        gradient_function: a function which returns a function used to compute the
            gradient of the cost function (see
            orquestra.opt.gradients.finite_differences_gradient for reference)

    Returns:
        A callable CostFunction object.

    Example use case:

    .. code:: python

        target_operator = ...
        ansatz = ...

        estimation_factory = substitution_based_estimation_tasks_factory(
            target_operator, ansatz
        )
        noise_preprocessor = add_normal_noise(1e-5, seed=1234)

        cost_function = create_cost_function(
            runner,
            estimation_factory,
            parameter_preprocessors=[noise_preprocessor]
        )

        optimizer = ...
        initial_params = ...

        opt_results = optimizer.minimize(cost_function, initial_params)

    """

    def _cost_function(parameters: np.ndarray) -> float:

        for preprocessor in (
            [] if parameter_preprocessors is None else parameter_preprocessors
        ):
            parameters = preprocessor(parameters)

        estimation_tasks = estimation_tasks_factory(parameters)
        expectation_values_list = estimation_method(runner, estimation_tasks)

        sum_of_expectation_values = np.concatenate(
            [expectation_value.values for expectation_value in expectation_values_list]
        ).sum()

        return sum_of_expectation_values

    return function_with_gradient(_cost_function, gradient_function(_cost_function))


def expectation_value_estimation_tasks_factory(
    target_operator: PauliRepresentation,
    parametrized_circuit: Circuit,
    estimation_preprocessors: Optional[List[EstimationPreprocessor]] = None,
    symbols_sort_key: SymbolsSortKey = str,
) -> EstimationTasksFactory:
    """Creates a EstimationTasksFactory object that can be used to create
    estimation tasks that returns the estimated expectation value of the input
    target operator with respect to the state prepared by the parameterized
    quantum circuit when evaluated to the input parameters.

    To be used with `create_cost_function` to create ground state cost functions.
    See `create_cost_function` docstring for an example use case.

    Args:
        target_operator: operator to be evaluated
        parametrized_circuit: parameterized circuit to prepare quantum states
        estimation_preprocessors: A list of callable functions used to create the
            estimation tasks. Each function must adhere to the EstimationPreprocessor
            protocol.
        symbols_sort_key: key defining ordering on parametrized_circuits free symbols.
            If s1,...,sN are all free symbols in parametrized_circuit, and cost function
            is called with `parameters` then the following binding occurs:
            parameters[i] -> sorted([s1,...,sN], key=symbols_sort_key)[i]
    Returns:
        An EstimationTasksFactory object.

    """
    if estimation_preprocessors is None:
        estimation_preprocessors = []

    estimation_tasks = [
        EstimationTask(
            operator=target_operator,
            circuit=parametrized_circuit,
            number_of_shots=None,
        )
    ]

    for preprocessor in estimation_preprocessors:
        estimation_tasks = preprocessor(estimation_tasks)

    circuit_symbols = _get_sorted_set_of_circuit_symbols(
        estimation_tasks, symbols_sort_key
    )

    def _tasks_factory(parameters: np.ndarray) -> List[EstimationTask]:
        symbols_map = create_symbols_map(circuit_symbols, parameters)
        return evaluate_estimation_circuits(
            estimation_tasks, [symbols_map for _ in estimation_tasks]
        )

    return _tasks_factory


def substitution_based_estimation_tasks_factory(
    target_operator: PauliRepresentation,
    ansatz: Ansatz,
    estimation_preprocessors: Optional[List[EstimationPreprocessor]] = None,
) -> EstimationTasksFactory:
    """Creates a EstimationTasksFactory object that can be used to create
    estimation tasks dynamically with parameters provided on the fly. These
    tasks will evaluate the parametric circuit of an ansatz, using a symbol-
    parameter map. Wow, a factory for factories! This is so meta.

    To be used with `create_cost_function`. See `create_cost_function` docstring
    for an example use case.

    Args:
        target_operator: operator to be evaluated
        ansatz: ansatz used to evaluate cost function
        estimation_preprocessors: A list of callable functions used to create the
            estimation tasks. Each function must adhere to the EstimationPreprocessor
            protocol.

    Returns:
        An EstimationTasksFactory object.

    """
    return expectation_value_estimation_tasks_factory(
        target_operator,
        ansatz.parametrized_circuit,
        estimation_preprocessors,
        ansatz.symbols_sort_key,
    )


def dynamic_circuit_estimation_tasks_factory(
    target_operator: PauliRepresentation,
    ansatz: Ansatz,
    estimation_preprocessors: Optional[List[EstimationPreprocessor]] = None,
) -> EstimationTasksFactory:
    """Creates a EstimationTasksFactory object that can be used to create
    estimation tasks dynamically with parameters provided on the fly. These
    tasks will evaluate the parametric circuit of an ansatz, without using
    a symbol-parameter map. Wow, a factory for factories!

    To be used with `create_cost_function`. See `create_cost_function` docstring
    for an example use case.

    Args:
        target_operator: operator to be evaluated
        ansatz: ansatz used to evaluate cost function
        estimation_preprocessors: A list of callable functions used to create the
            estimation tasks. Each function must adhere to the EstimationPreprocessor
            protocol.

    Returns:
        An EstimationTasksFactory object.
    """

    def _tasks_factory(parameters: np.ndarray) -> List[EstimationTask]:

        # TODO: In some ansatzes, `ansatz._generate_circuit(parameters)` does not
        # produce an executable circuit, but rather, they ignore the parameters and
        # returns a parametrized circuit with sympy symbols.
        # (Ex. see ansatzes in orquestra-vqa)
        #
        # Combined with how this is a private method, we will probably have to somewhat
        # refactor the ansatz class.

        circuit = ansatz._generate_circuit(parameters)

        estimation_tasks = [
            EstimationTask(
                operator=target_operator, circuit=circuit, number_of_shots=None
            )
        ]

        for preprocessor in (
            [] if estimation_preprocessors is None else estimation_preprocessors
        ):
            estimation_tasks = preprocessor(estimation_tasks)

        return estimation_tasks

    return _tasks_factory
