import unittest
from zquantum.core.circuit import Circuit, Gate, Qubit
from .utils import exponentiate_fermion_operator, create_layer_of_gates


class TestUCCSDUtils(unittest.TestCase):
    def test_create_layer_of_gates(self):
        # Given
        number_of_qubits = 4
        gate_name = "X"
        qubits = [Qubit(i) for i in range(0, number_of_qubits)]
        gate_0 = Gate(gate_name, qubits=[qubits[0]])
        gate_1 = Gate(gate_name, qubits=[qubits[1]])
        gate_2 = Gate(gate_name, qubits=[qubits[2]])
        gate_3 = Gate(gate_name, qubits=[qubits[3]])
        target_circuit = Circuit()
        target_circuit.qubits = qubits
        target_circuit.gates = [gate_0, gate_1, gate_2, gate_3]

        # When
        layer_of_x = create_layer_of_gates(number_of_qubits, gate_name)

        # Then
        self.assertEqual(layer_of_x, target_circuit)

