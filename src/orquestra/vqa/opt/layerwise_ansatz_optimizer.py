from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.interfaces.optimizer import Optimizer, construct_history_info
from zquantum.core.interfaces.functions import CallableStoringArtifacts
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.functions import CallableWithGradient
from zquantum.core.typing import RecorderFactory
from scipy.optimize import OptimizeResult
from typing import Dict, Optional, List, Union, Callable
import numpy as np
from functools import partial
import copy


class LayerwiseAnsatzOptimizer:
    def __init__(
        self,
        inner_optimizer: Optimizer,
        min_layer: int,
        max_layer: int,
        n_layers_per_iteration: int = 1,
        parameters_initializer: Optional[Callable] = None,
        recorder: RecorderFactory = _recorder,
    ):
        """
        Args:
            inner_optimizer: optimizer used for optimization at each layer.
            min_layer: starting number of layers.
            max_layer: maximum number of layers, at which optimization should stop.
            recorder: recorder object which defines how to store the optimization history.
        """
        assert 0 <= min_layer <= max_layer
        assert n_layers_per_iteration > 0
        self.inner_optimizer = inner_optimizer
        self.min_layer = min_layer
        self.max_layer = max_layer
        self.n_layers_per_iteration = n_layers_per_iteration
        if parameters_initializer is None:
            self.parameters_initializer = partial(np.random.uniform, -np.pi, np.pi)
        else:
            self.parameters_initializer = parameters_initializer
        self.recorder = recorder

    def minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying all the parameters from the grid.

        Note:
            Cost function needs to have `ansatz` property

        Args:
            cost_function(zquantum.core.interfaces.cost_function.CostFunction): object representing cost function we want to minimize
            inital_params (np.ndarray): initial parameters for the cost function

        """
        # Since this optimizer modifies the ansatz which is a part of the input cost function
        # we use copy of it instead.
        cost_function = copy.deepcopy(cost_function)

        if not hasattr(cost_function, "ansatz"):
            raise ValueError("Provided cost function needs to have ansatz property.")
        if keep_history:
            cost_function = self.recorder(cost_function)

        cost_function.ansatz.number_of_layers = self.min_layer

        number_of_params = cost_function.ansatz.number_of_params
        if initial_params is None:
            initial_params = self.parameters_initializer(number_of_params)

        for i in range(self.min_layer, self.max_layer + 1, self.n_layers_per_iteration):
            # keep_history is set to False, as the cost function is already being recorded
            # if keep_history is specified.
            if i != self.min_layer:
                new_layer_params = self.parameters_initializer(
                    cost_function.ansatz.number_of_params - len(optimal_params)
                )
                initial_params = np.concatenate([optimal_params, new_layer_params])

            layer_results = self.inner_optimizer.minimize(
                cost_function, initial_params, keep_history=False
            )
            optimal_params = layer_results.opt_params
            cost_function.ansatz.number_of_layers += self.n_layers_per_iteration

        # layer_results["history"] will be empty as inner_optimizer was used with
        # keep_history false.
        del layer_results["history"]

        return OptimizeResult(
            **layer_results, **construct_history_info(cost_function, keep_history)
        )