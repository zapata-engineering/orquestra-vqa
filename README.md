# orquestra-vqa

## What is it?

`orquestra-vqa` is a library with core functionalities for implementing variational quantum algorithms developed by [Zapata](https://www.zapatacomputing.com) for our [Orquestra](https://www.zapatacomputing.com/orquestra/) platform.

`orquestra-vqa` provides:

-   interfaces for implementing ansatzes including qaoa and qcbm.
-   optimizers and cost functions tailored to vqa
-   misc functions such as grouping, qaoa interpolation, and estimators

## Installation

Even though it's intended to be used with Orquestra, `orquestra-vqa` can be also used as a Python module.
To install it, make to install its dependencies: `orquestra-quantum`, `orquestra-opt` and `orquestra-cirq`. Then you just need to run `pip install .` from the main directory.

## Usage

Here's an example of how to use methods from `orquestra-vqa` to create a cost function for qcbm and optimize it using scipy optimizer.

```python
from orquestra.vqa.cost_function.qcbm_cost_function import create_QCBM_cost_function
from orquestra.vqa.ansatz.qcbm import QCBMAnsatz
from orquestra.opt.history.recorder import recorder
from orquestra.quantum.symbolic_simulator import SymbolicSimulator
from orquestra.quantum.distributions import compute_mmd
from orquestra.quantum.distributions import MeasurementOutcomeDistribution
from orquestra.opt.optimizers.scipy_optimizer import ScipyOptimizer
import numpy as np

target_distribution = MeasurementOutcomeDistribution(
    {
        "0000": 1.0,
        "0001": 0.0,
        "0010": 0.0,
        "0011": 1.0,
        "0100": 0.0,
        "0101": 1.0,
        "0110": 0.0,
        "0111": 0.0,
        "1000": 0.0,
        "1001": 0.0,
        "1010": 1.0,
        "1011": 0.0,
        "1100": 1.0,
        "1101": 0.0,
        "1110": 0.0,
        "1111": 1.0,
    }
)

def orquestra_vqa_example_function():
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
        target_distribution=target_distribution
    )

    optimizer = ScipyOptimizer(method="L-BFGS-B")
    initial_params = np.ones(ansatz.number_of_params) / 5
    opt_results = optimizer.minimize(cost_function, initial_params)

    return opt_results

orquestra_vqa_example_function()
```

## Development and Contribution

You can find the development guidelines in the [`orquestra-quantum` repository](https://github.com/zapatacomputing/orquestra-quantum).
