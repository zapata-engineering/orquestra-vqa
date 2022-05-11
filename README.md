# orquestra-vqa

## What is it?

`orquestra-vqa` is a core library of the scientific code for [Orquestra](https://www.zapatacomputing.com/orquestra/) – the platform developed by [Zapata Computing](https://www.zapatacomputing.com) for performing variational quantum algorithms.

`orquestra-vqa` provides:

- interfaces for implementing ansatzes including qaoa and qcbm.
- optimizers and cost functions tailored to vqa
- misc functions such as grouping, qaoa interpolation, and estimators

## Usage

### Workflows

Here's an example of how to use methods from `orquestra-vqa` to run a workflow. This workflow runs a circuit with a single Hadamard gate 100 times and returns the results:

```python
from orquestra.vqa.cost_function.qcbm_cost_function import create_QCBM_cost_function
from orquestra.vqa.ansatz.qcbm import QCBMAnsatz
from orquestra.opt.history.recorder import recorder
from orquestra.quantum.symbolic_simulator import SymbolicSimulator

@sdk.task(
    source_import=sdk.GitImport(repo_url="git@github.com:my_username/my_repository.git", git_ref="main"),
    dependency_imports=[sdk.GitImport(repo_url="git@github.com:zapatacomputing/orquestra-vqa.git", git_ref="main"),
    sdk.GitImport(repo_url="git@github.com:zapatacomputing/orquestra-quantum.git", git_ref="main"),
    sdk.GitImport(repo_url="git@github.com:zapatacomputing/orquestra-opt.git", git_ref="main")]
)
def orquestra_vqa_example_task()
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


@sdk.workflow()
def orquestra_vqa_example_workflow()
    results = orquestra_vqa_example_task()
    return [results]
```

### Python

Even though it's intended to be used with Orquestra, `orquestra-vqa` can be also used as a standalone Python module.
To install it, you just need to run `pip install -e .` from the main directory.

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