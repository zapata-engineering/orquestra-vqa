import numpy as np
from zquantum.optimizers.layerwise_ansatz_optimizer import LayerwiseAnsatzOptimizer
from zquantum.optimizers.scipy_optimizer import ScipyOptimizer
from zquantum.core.interfaces.mock_objects import MockAnsatz
from zquantum.core.interfaces.optimizer_test import OptimizerTests
from zquantum.core.history.recorder import recorder

import pytest
from functools import partial


@pytest.fixture(
    params=[
        {"inner_optimizer": ScipyOptimizer("L-BFGS-B"), "min_layer": 1, "max_layer": 1}
    ]
)
def optimizer(request):
    return LayerwiseAnsatzOptimizer(**request.param)


@pytest.fixture(params=[True, False])
def keep_history(request):
    return request.param


class TestLayerwiseOptimizer(OptimizerTests):
    @pytest.fixture
    def sum_x_squared(self):
        class my_fun:
            def __init__(self, ansatz):
                self.ansatz = ansatz

            def __call__(self, x):
                return sum(x ** 2)

        return my_fun(MockAnsatz(1, 5))

    @pytest.fixture
    def rosenbrock_function(self):
        def _rosenbrock_function(x):
            return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

        _rosenbrock_function.ansatz = MockAnsatz(1, 5)

        return _rosenbrock_function

    def test_gradients_history_is_recorded_if_keep_history_is_true(
        self, optimizer, sum_x_squared
    ):
        pytest.xfail(
            """This test fail since LayerwiseAnsatzOptimizer is deepcopying 
                input cost_function, which is intended behaviour."""
        )

    def test_ansatz_is_not_modified_outside_of_minimize(self, sum_x_squared):
        optimizer = LayerwiseAnsatzOptimizer(
            inner_optimizer=ScipyOptimizer("L-BFGS-B"), min_layer=2, max_layer=4
        )
        cost_function = sum_x_squared
        initial_number_of_layers = cost_function.ansatz.number_of_layers
        _ = optimizer.minimize(cost_function, initial_params=[0, 0])
        assert cost_function.ansatz.number_of_layers == initial_number_of_layers

    @pytest.mark.parametrize("max_layer", [2, 3, 4, 5])
    def test_length_of_parameters_in_history_increases(self, sum_x_squared, max_layer):
        min_layer = 1
        optimizer = LayerwiseAnsatzOptimizer(
            inner_optimizer=ScipyOptimizer("L-BFGS-B"),
            min_layer=min_layer,
            max_layer=max_layer,
        )
        opt_results = optimizer.minimize(
            sum_x_squared, initial_params=np.ones(min_layer)
        )
        assert len(opt_results.opt_params) == max_layer

    def test_fails_if_cost_function_does_not_have_ansatz(self, optimizer):
        cost_function_without_ansatz = lambda x: sum(x ** 2)
        with pytest.raises(ValueError):
            _ = optimizer.minimize(cost_function_without_ansatz, initial_params=[0, 0])

    @pytest.mark.parametrize("min_layer,max_layer", [[1, 2], [1, 5], [100, 120]])
    def test_parameters_are_properly_initialized_for_each_layer(
        self, sum_x_squared, optimizer, min_layer, max_layer
    ):
        parameters_initializer = recorder(partial(np.random.uniform, -np.pi, np.pi))
        optimizer = LayerwiseAnsatzOptimizer(
            inner_optimizer=ScipyOptimizer("L-BFGS-B"),
            min_layer=min_layer,
            max_layer=max_layer,
            parameters_initializer=parameters_initializer,
        )

        _ = optimizer.minimize(sum_x_squared, initial_params=np.ones(min_layer))
        assert parameters_initializer.call_number == max_layer - min_layer
        for entry in parameters_initializer.history:
            assert len(entry.value) == 1

    @pytest.mark.parametrize("min_layer,max_layer", [[-1, 2], [3, 2], [-5, -1]])
    def test_fails_for_invalid_min_max_layer(self, min_layer, max_layer):
        with pytest.raises(AssertionError):
            LayerwiseAnsatzOptimizer(
                inner_optimizer=ScipyOptimizer("L-BFGS-B"),
                min_layer=min_layer,
                max_layer=max_layer,
            )

    def test_fails_if_cost_function_does_not_have_ansatz(self, optimizer):
        cost_function_without_ansatz = lambda x: sum(x ** 2)
        with pytest.raises(ValueError):
            _ = optimizer.minimize(cost_function_without_ansatz, initial_params=[0, 0])