################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from openfermion import IsingOperator, QubitOperator
from orquestra.quantum.operators import PauliSum, PauliTerm

from orquestra.vqa.openfermion import openfermion_adapter


@openfermion_adapter()
def function(op):
    assert isinstance(op, QubitOperator)
    return op


@openfermion_adapter(operatorType=IsingOperator)
def function_ising(op):
    assert isinstance(op, IsingOperator)
    return op


def test_adapter_converts_function_input_and_output():
    # op an orquestra pauli operator but the adapter converts it to an openfermion
    # operator before passing it into the function.
    op = PauliTerm("X0*Z1", 0.5)
    result = function(op)

    op_ising = PauliSum("Z0*Z1 + Z2*Z1")
    result_ising = function_ising(op_ising)

    # The adapter converts the output of the function back into the original
    # orquestra type
    assert isinstance(result, PauliSum)
    assert isinstance(result_ising, PauliSum)

    # Check that the result is the same as the original input
    assert result == op
    assert result_ising == op_ising
