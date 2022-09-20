################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import copy
from collections import defaultdict
from typing import Callable, Dict, cast

import numpy as np
from orquestra.opt.api import CostFunction, NestedOptimizer, Optimizer
from orquestra.opt.api.optimizer import extend_histories
from orquestra.opt.history.recorder import AnyRecorder, RecorderFactory
from orquestra.opt.history.recorder import recorder as _recorder
from scipy.optimize import OptimizeResult

from orquestra.vqa.api.ansatz import Ansatz


def append_random_params(target_size: int, params: np.ndarray) -> np.ndarray:
    """
    Adds new random parameters to the `params` so that the size
    of the output is `target_size`.
    New parameters are sampled from a uniform distribution over [-pi, pi].

    Args:
        target_size: target number of parameters
        params: params that we want to extend
    """
    assert len(params) < target_size
    new_params = np.random.uniform(-np.pi, np.pi, target_size - len(params))
    return np.concatenate([params, new_params])


class LayerwiseAnsatzOptimizer(NestedOptimizer):
    @property
    def inner_optimizer(self) -> Optimizer:
        return self._inner_optimizer

    @property
    def recorder(self) -> RecorderFactory:
        return self._recorder

    def __init__(
        self,
        ansatz: Ansatz,
        inner_optimizer: Optimizer,
        min_layer: int,
        max_layer: int,
        n_layers_per_iteration: int = 1,
        parameters_initializer: Callable[
            [int, np.ndarray], np.ndarray
        ] = append_random_params,
        recorder: RecorderFactory = _recorder,
    ) -> None:
        """
        LayerwiseAnsatzOptimizer is an optimizer for optimizing ansatz parameters
        for ansatzes with layered structure.
        In each iteration it adds specific number of new layers and initializes
        their parameters using `parameters_initializer`.
        The idea behind this method is to start from a less complex problem
        (i.e. small number of layers) and gradually increase its difficulty using
        parameters obtained in the previous iteration as good starting points
        for the following iteration.

        To make it work it requires using a cost function factory that takes
        `Ansatz` object as input
        to generate the cost function (see `_minimize` method).

        Args:
            ansatz: ansatz that will be used for creating the cost function.
            inner_optimizer: optimizer used for optimization of parameters
                after adding a new layer to the ansatz.
            min_layer: starting number of layers.
            max_layer: maximum number of layers, at which optimization should stop.
            n_layers_per_iteration: number of layers added for each iteration.
            parameters_initializer: method for initializing parameters of the added
            layers. See append_new_random_params for example of an implementation.
        """

        assert 0 <= min_layer <= max_layer
        assert n_layers_per_iteration > 0
        self._ansatz = ansatz
        self._inner_optimizer = inner_optimizer
        self._min_layer = min_layer
        self._max_layer = max_layer
        self._n_layers_per_iteration = n_layers_per_iteration
        self._parameters_initializer = parameters_initializer
        self._recorder = recorder

    def _minimize(
        self,
        cost_function_factory: Callable[[Ansatz], CostFunction],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """
        Finds the parameters that minimize the value of the cost function created using
        `cost_function_factory`. In each iteration the number of layers of ansatz are
        increased, and therefore new cost function is created and the size of
        the parameter vector increases.

        NOTE:
            - The size of `initial_params` should correspond to the number of
            parameters of the ansatz with number of layers specified by `min_layer`.
            - The optimal parameters should minimize the value of the cost function
            for the ansatz with number of layers specified by `max_layer`.

        Args:
            cost_function_factory: a function that returns a cost function that
                depends on the provided ansatz.
            inital_params: initial parameters for the cost function,
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.

        """
        ansatz = copy.deepcopy(self._ansatz)
        ansatz.number_of_layers = self._min_layer

        nit = 0
        nfev = 0
        histories: Dict = defaultdict(list)
        histories["history"] = []
        initial_params_per_iteration = initial_params
        optimal_params: np.ndarray = np.array([])
        for i in range(
            self._min_layer, self._max_layer + 1, self._n_layers_per_iteration
        ):
            assert ansatz.number_of_layers == i

            if i != self._min_layer:
                initial_params_per_iteration = self._parameters_initializer(
                    ansatz.number_of_params, optimal_params
                )

            cost_function = cost_function_factory(ansatz=ansatz)  # type: ignore

            if keep_history:
                cost_function = self.recorder(cost_function)
            layer_results = self.inner_optimizer.minimize(
                cost_function, initial_params_per_iteration, keep_history=False
            )

            optimal_params = layer_results.opt_params
            ansatz.number_of_layers += self._n_layers_per_iteration

            nfev += layer_results.nfev
            nit += layer_results.nit

            if keep_history:
                histories = extend_histories(
                    cast(AnyRecorder, cost_function), histories
                )

        del layer_results["history"]
        del layer_results["nit"]
        del layer_results["nfev"]

        return OptimizeResult(**layer_results, **histories, nfev=nfev, nit=nit)
