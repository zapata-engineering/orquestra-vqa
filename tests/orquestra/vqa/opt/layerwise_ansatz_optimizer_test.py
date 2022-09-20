################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from unittest import mock

import numpy as np
import pytest
from orquestra.opt.api.optimizer_test import NESTED_OPTIMIZER_CONTRACTS
from orquestra.opt.optimizers.scipy_optimizer import ScipyOptimizer

from orquestra.vqa.opt.layerwise_ansatz_optimizer import (
    LayerwiseAnsatzOptimizer,
    append_random_params,
)
from orquestra.vqa.testing.mock_objects import MockAnsatz


@pytest.fixture
def ansatz():
    return MockAnsatz(1, 5)


@pytest.fixture
def initial_params():
    return np.array([1])


def cost_function_factory(ansatz):
    def cost_function(x):
        return sum(x**2) * ansatz.number_of_layers

    return cost_function


class TestLayerwiseAnsatzOptimizer:
    @pytest.mark.parametrize("contract", NESTED_OPTIMIZER_CONTRACTS)
    def test_if_satisfies_contracts(self, contract, ansatz, initial_params):
        optimizer = LayerwiseAnsatzOptimizer(
            ansatz=ansatz,
            inner_optimizer=ScipyOptimizer("L-BFGS-B"),
            min_layer=1,
            max_layer=3,
        )

        assert contract(optimizer, cost_function_factory, initial_params)

    def test_ansatz_is_not_modified_outside_of_minimize(self, ansatz, initial_params):
        initial_number_of_layers = ansatz.number_of_layers
        optimizer = LayerwiseAnsatzOptimizer(
            ansatz=ansatz,
            inner_optimizer=ScipyOptimizer("L-BFGS-B"),
            min_layer=1,
            max_layer=3,
        )
        _ = optimizer.minimize(cost_function_factory, initial_params=initial_params)
        assert ansatz.number_of_layers == initial_number_of_layers

    @pytest.mark.parametrize("max_layer", [2, 3, 4, 5])
    def test_dimension_of_solution_increases(self, max_layer, ansatz):
        min_layer = 1
        optimizer = LayerwiseAnsatzOptimizer(
            ansatz=ansatz,
            inner_optimizer=ScipyOptimizer("L-BFGS-B"),
            min_layer=min_layer,
            max_layer=max_layer,
        )
        opt_results = optimizer.minimize(
            cost_function_factory, initial_params=np.ones(min_layer), keep_history=True
        )
        assert len(opt_results.opt_params) == max_layer

    @pytest.mark.parametrize(
        "min_layer,max_layer,n_layers_per_iteration",
        [[1, 2, 1], [1, 5, 1], [100, 120, 1], [1, 5, 2], [1, 10, 4], [1, 10, 20]],
    )
    def test_parameters_are_properly_initialized_for_each_layer(
        self, min_layer, max_layer, n_layers_per_iteration, ansatz
    ):
        def parameters_initializer(number_of_params, old_params):
            return np.random.uniform(-np.pi, np.pi, number_of_params)

        parameters_initializer = mock.Mock(wraps=parameters_initializer)
        optimizer = LayerwiseAnsatzOptimizer(
            ansatz=ansatz,
            inner_optimizer=ScipyOptimizer("L-BFGS-B"),
            min_layer=min_layer,
            max_layer=max_layer,
            n_layers_per_iteration=n_layers_per_iteration,
            parameters_initializer=parameters_initializer,
        )

        _ = optimizer.minimize(cost_function_factory, initial_params=np.ones(min_layer))
        assert (
            parameters_initializer.call_count
            == (max_layer - min_layer) // n_layers_per_iteration
        )

        for ((args, _kwrgs), i) in zip(
            parameters_initializer.call_args_list,
            range(
                min_layer + n_layers_per_iteration, max_layer, n_layers_per_iteration
            ),
        ):
            number_of_params, old_params = args
            assert number_of_params == i

    @pytest.mark.parametrize("min_layer,max_layer", [[-1, 2], [3, 2], [-5, -1]])
    def test_fails_for_invalid_min_max_layer(self, min_layer, max_layer, ansatz):
        with pytest.raises(AssertionError):
            LayerwiseAnsatzOptimizer(
                ansatz=ansatz,
                inner_optimizer=ScipyOptimizer("L-BFGS-B"),
                min_layer=min_layer,
                max_layer=max_layer,
            )


@pytest.mark.parametrize(
    "target_size,params",
    [[4, np.array([0, 1])], [100, np.ones(99)], [5, np.array([])]],
)
def test_append_random_params(target_size, params):
    new_params = append_random_params(target_size, params)
    assert len(new_params) == target_size

    np.testing.assert_array_equal(params, new_params[: len(params)])


@pytest.mark.parametrize(
    "target_size,params",
    [[-5, np.array([0, 1])], [15, np.ones(99)], [10, np.ones(10)]],
)
def test_append_random_params_fails_for_wrong_input(target_size, params):
    with pytest.raises(AssertionError):
        _ = append_random_params(target_size, params)
