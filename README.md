# orquestra-vqa

## What is it?

`orquestra-vqa` is a library with core functionalities for implementing variational quantum algorithms developed by [Zapata](https://www.zapatacomputing.com) for our [Orquestra](https://www.zapatacomputing.com/orquestra/) platform.

`orquestra-vqa` provides:

- interfaces for implementing ansatzes including qaoa and qcbm.
- optimizers and cost functions tailored to vqa
- misc functions such as grouping, qaoa interpolation, and estimators

## Installation

Even though it's intended to be used with Orquestra, `orquestra-quantum` can be also used as a Python module.
To install it, make to install its dependencies: `orquestra-quantum` and `orquestra-opt`. Then you just need to run `pip install .` from the main directory.

## Usage

Here's an example of how to use methods from `orquestra-vqa` to create a cost function for qcbm and optimize it using scipy optimizer.

```python
from orquestra.vqa.cost_function.qcbm_cost_function import create_QCBM_cost_function
from orquestra.vqa.ansatz.qcbm import QCBMAnsatz
from orquestra.opt.history.recorder import recorder
from orquestra.quantum.symbolic_simulator import SymbolicSimulator


def orquestra_vqa_example_function()
    ansatz = QCBMAnsatz(1, 4, "all")
    backend = SymbolicSimulator()
    distance_measure_kwargs = {
                "distance_measure": compute_mmd,
                "distance_measure_parameters": {"sigma": 1},
            }
    cost_function = create_QCBM_cost_function(
        ansatz,
        backend,
        10,
        **distance_measure_kwargs,
    )

    optimizer = ScipyOptimizer(method="L-BFGS-B")
    initial_params = np.ones(ansatz.number_of_params) / 5
    opt_results = optimizer.minimize(cost_function, initial_params)

    return opt_results
```

## Development and Contribution

You can find the development guidelines in the [`orquestra-quantum` repository](https://github.com/zapatacomputing/orquestra-quantum).