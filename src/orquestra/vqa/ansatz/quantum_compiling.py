################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from typing import List, Optional

import numpy as np
import sympy
from orquestra.quantum.circuits import CNOT, RX, RZ, Circuit
from overrides import overrides

from ..api.ansatz import Ansatz
from ..api.ansatz_utils import ansatz_property


class HEAQuantumCompilingAnsatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")

    def __init__(self, number_of_layers: int, number_of_qubits: int):
        """An ansatz implementation for the Hardware Efficient Quantum Compiling Ansatz
            used in https://arxiv.org/pdf/2011.12245.pdf

        Args:
            number_of_layers: number of layers in the circuit.
            number_of_qubits: number of qubits in the circuit.

        Attributes:
            number_of_qubits: See Args
            number_of_layers: See Args
        """
        if number_of_layers <= 0:
            raise ValueError("number_of_layers must be a positive integer")
        super().__init__(number_of_layers)
        assert number_of_qubits % 2 == 0
        self._number_of_qubits = number_of_qubits

    def _build_rotational_subcircuit(
        self, circuit: Circuit, parameters: np.ndarray
    ) -> Circuit:
        """Add the subcircuit which includes several rotation gates acting on each qubit

        Args:
            circuit: The circuit to append to
            parameters: The variational parameters (or symbolic parameters)

        Returns:
            circuit with added rotational sub-layer
        """
        # Add RZ(theta) RX(pi/2) RZ(theta') RX(pi/2) RZ(theta'')
        for qubit_index in range(self.number_of_qubits):

            qubit_parameters = parameters[qubit_index * 3 : (qubit_index + 1) * 3]

            circuit += RZ(qubit_parameters[0])(qubit_index)
            circuit += RX(np.pi / 2)(qubit_index)
            circuit += RZ(qubit_parameters[1])(qubit_index)
            circuit += RX(np.pi / 2)(qubit_index)
            circuit += RZ(qubit_parameters[2])(qubit_index)

        return circuit

    def _build_circuit_layer(self, parameters: np.ndarray) -> Circuit:
        """Build circuit layer for the hardware efficient quantum compiling ansatz

        Args:
            parameters: The variational parameters (or symbolic parameters)

        Returns:
            Circuit containing a single layer of the Hardware Efficient Quantum
            Compiling Ansatz
        """
        circuit_layer = Circuit()

        # Add RZ(theta) RX(pi/2) RZ(theta') RX(pi/2) RZ(theta'')
        circuit_layer = self._build_rotational_subcircuit(
            circuit_layer, parameters[: 3 * self.number_of_qubits]
        )

        qubit_ids = list(range(self.number_of_qubits))
        # Add CNOT(x, x+1) for x in even(qubits)
        for control, target in zip(
            qubit_ids[::2], qubit_ids[1::2]
        ):  # loop over qubits 0, 2, 4...
            circuit_layer += CNOT(control, target)

        # Add RZ(theta) RX(pi/2) RZ(theta') RX(pi/2) RZ(theta'')
        circuit_layer = self._build_rotational_subcircuit(
            circuit_layer,
            parameters[3 * self.number_of_qubits : 6 * self.number_of_qubits],
        )

        # Add CNOT layer working "inside -> out", skipping every other qubit

        for qubit_index in qubit_ids[: int(self.number_of_qubits / 2)][::-1][::2]:
            control = qubit_index
            target = self.number_of_qubits - qubit_index - 1
            circuit_layer += CNOT(control, target)

            if qubit_index != 0 or self.number_of_qubits % 4 == 0:
                control = self.number_of_qubits - qubit_index
                target = qubit_index - 1
                circuit_layer += CNOT(control, target)

        return circuit_layer

    @overrides
    def _generate_circuit(self, params: Optional[np.ndarray] = None) -> Circuit:
        """Builds the ansatz circuit (based on: 2011.12245, Fig. 1)

        Args:
            params (numpy.ndarray): input parameters of the circuit (1d array).

        Returns:
            Circuit
        """
        if params is None:
            params = np.asarray(self.symbols, dtype=object)

        assert len(params) == self.number_of_params

        circuit = Circuit()
        for layer_index in range(self.number_of_layers):
            circuit += self._build_circuit_layer(
                params[
                    layer_index
                    * self.number_of_params_per_layer : (layer_index + 1)
                    * self.number_of_params_per_layer
                ]
            )
        return circuit

    @property
    def number_of_params(self) -> int:
        """
        Returns number of parameters in the ansatz.
        """
        return self.number_of_params_per_layer * self.number_of_layers

    @property
    def number_of_params_per_layer(self) -> int:
        """
        Returns number of parameters in the ansatz.
        """
        return 3 * self.number_of_qubits * 2

    @property
    def symbols(self) -> List[sympy.Symbol]:
        """
        Returns a list of symbolic parameters used for creating the ansatz.
        The order of the symbols should match the order in which parameters
        should be passed for creating executable circuit.
        """
        return [
            sympy.Symbol("theta_{}".format(i)) for i in range(self.number_of_params)
        ]
