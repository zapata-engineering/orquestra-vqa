################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
from typing import Optional

import numpy as np
import sympy
from orquestra.quantum.circuits import RX, Circuit
from orquestra.quantum.utils import create_symbols_map
from overrides import overrides

from ..api.ansatz import Ansatz
from ..api.ansatz_utils import ansatz_property


class MockAnsatz(Ansatz):
    supports_parametrized_circuits = True
    problem_size = ansatz_property("problem_size")

    def __init__(self, number_of_layers: int, problem_size: int):
        super().__init__(number_of_layers)
        self.number_of_layers = number_of_layers
        self.problem_size = problem_size

    @property
    def number_of_qubits(self) -> int:
        return self.problem_size

    @overrides
    def _generate_circuit(self, params: Optional[np.ndarray] = None) -> Circuit:
        circuit = Circuit()
        symbols = [
            sympy.Symbol(f"theta_{layer_index}")
            for layer_index in range(self.number_of_layers)
        ]
        for theta in symbols:
            for qubit_index in range(self.number_of_qubits):
                circuit += RX(theta)(qubit_index)
        if params is not None:
            symbols_map = create_symbols_map(symbols, params)
            circuit = circuit.bind(symbols_map)
        return circuit
