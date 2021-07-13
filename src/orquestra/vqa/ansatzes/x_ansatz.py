from zquantum.core.interfaces.ansatz import Ansatz, ansatz_property

from zquantum.core.circuits import Circuit
from zquantum.core.evolution import time_evolution
from zquantum.core.openfermion import change_operator_type

from openfermion import QubitOperator, IsingOperator
from openfermion.utils import count_qubits
from typing import Tuple, Union, Optional, List
import numpy as np
import sympy
from overrides import overrides
from itertools import combinations


class QAOAXAnsatz(Ansatz):

    supports_parametrized_circuits = True
    cost_hamiltonian = ansatz_property("cost_hamiltonian")

    def __init__(
        self,
        number_of_layers: int,
        cost_hamiltonian: Union[QubitOperator, IsingOperator],
    ):
        """This is implementation of the X Ansatz from https://arxiv.org/abs/2105.01114

        Args:
            number_of_layers: k-body depth (the maximum number of qubits entangled at one time) as described in https://arxiv.org/abs/2105.01114. Cannot be greater than the number of qubits.
            cost_hamiltonian: Hamiltonian representing the cost function

        Attributes:
            number_of_qubits: number of qubits required for the ansatz circuit.
            number_of_params: number of the parameters that need to be set for the ansatz circuit.
        """

        super().__init__(number_of_layers)
        self._cost_hamiltonian = cost_hamiltonian

        assert number_of_layers <= self.number_of_qubits

    @property
    def number_of_qubits(self):
        """Returns number of qubits used for the ansatz circuit."""
        return count_qubits(change_operator_type(self._cost_hamiltonian, QubitOperator))

    @property
    def number_of_params(self) -> int:
        """Returns number of parameters in the ansatz."""
        # return sigma(self.number_of_layers self.number_of_qubits choose k
        sum = 0
        for i in range(1, self.number_of_layers + 1):
            sum += ncr(self.number_of_qubits, i)
        return sum

    @overrides  # bc this method is in super
    def _generate_circuit(self, params: Optional[np.ndarray] = None) -> Circuit:
        """Returns a parametrizable circuit represention of the ansatz.
        Args:
            params: parameters of the circuit.
        """
        if params is not None:
            Warning(
                "This method retuns a parametrizable circuit, params will be ignored."
            )
        circuit = Circuit()

        # Add time evolution layers
        j = 0
        edges = get_edges_from_cost_hamiltonian(self._cost_hamiltonian)
        for k in range(self.number_of_layers):

            A = combinations(range(0, self.number_of_qubits), k + 1)

            for S in list(A):
                H_j = QubitOperator(" ".join([f"X{i}" for i in S])) * cost_of_cut(
                    S, edges
                ) / 2

                circuit += time_evolution(H_j, sympy.Symbol(f"theta_{j}"))
                j += 1

        return circuit


def create_x_qaoa_circuits(
    hamiltonians: List[QubitOperator], number_of_layers: Union[int, List[int]]
):
    """Creates parameterizable quantum circuits based on the farhi qaoa ansatz for each
    hamiltonian in the input list using the set number of layers.
    Args:
        hamiltonians (List[QubitOperator]): List of hamiltonians for constructing the
            circuits
        number_of_layers (Union[int, List[int]]): The k-body depth of the ansatz in the circuit.
            If an int is passed in, the same k-body depth is used for every ansatz circuit, however,
            if a list of ints is passed in, the k-body depth used for the hamiltonian at index i of the hamiltonians
            list is the integer at index i of the number_of_layers list.
    Returns:
        List of zquantum.core.circuit.Circuit
    """
    if isinstance(number_of_layers, int):
        number_of_layers = [number_of_layers for _ in range(len(hamiltonians))]
    number_of_layers_list = number_of_layers
    assert len(number_of_layers_list) == len(hamiltonians)

    circuitset = []
    for number_of_layers, hamiltonian in zip(number_of_layers_list, hamiltonians):
        ansatz = QAOAXAnsatz(number_of_layers, hamiltonian)
        circuitset.append(ansatz.parametrized_circuit)
    return circuitset


import operator as op
from functools import reduce


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def cost_of_cut(cut: Tuple, edges: List):
    # Edge is in the form of [node1, node2, weight]
    total_cost = 0
    for edge in edges:
        if len(edge) == 3:
            a = edge[0] in cut
            b = edge[1] in cut
            if a != b:
                total_cost += edge[2]
    return total_cost


def get_edges_from_cost_hamiltonian(cost_hamiltonian: QubitOperator):
    # Edge is in the form of [node1, node2, weight]
    edges = []
    for term in cost_hamiltonian.terms:
        temp = []
        for operator in term:
            temp.append(operator[0])
        if temp:
            edges.append(temp)

    a = 0
    for value in cost_hamiltonian.terms.values():
        if len(edges) > a:  # only works when constant term is last term.
            edges[a].append(value * 2) # Multiply by 2 because when converting maxcut graph to hamiltonian the edges get divided by 2
        a += 1
    return edges
