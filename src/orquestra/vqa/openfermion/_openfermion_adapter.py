################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import functools
from typing import Sequence

import openfermion
from orquestra.integrations.cirq.conversions import from_openfermion, to_openfermion
from orquestra.quantum.operators import PauliSum, PauliTerm

relevant_openfermion_types = [openfermion.QubitOperator, openfermion.IsingOperator]

relevant_orquestra_types = [PauliSum, PauliTerm]


def _extract_relevant_objects(args, relevant_types):
    res = []

    for i, arg in enumerate(args):
        if isinstance(arg, tuple(relevant_types)):
            res.append((arg, i))

    return res


def openfermion_adapter(operatorType=openfermion.QubitOperator):
    def adapter(func):
        """Wrapper around openfermion functions that contain QubitOperator or IsingOperator,
        to make them compatible with orquestra PauliSum and PauliTerm.
        """

        @functools.wraps(func)
        def wrapper_decorator(*args, **kwargs):
            # cast orquestra objects to openfermion objects
            orquestra_objects_to_cast = _extract_relevant_objects(
                args, relevant_orquestra_types
            )

            args = list(args)

            for object, idx in orquestra_objects_to_cast:
                cast_object = to_openfermion(object, operatorType)
                args[idx] = cast_object

            value = func(*args, **kwargs)

            is_not_originally_seq = False
            if not isinstance(value, Sequence):
                value = [value]
                is_not_originally_seq = True
            else:
                seq_type = type(value)
                value = list(value)

            cloned_objects_to_cast = _extract_relevant_objects(
                value, relevant_openfermion_types
            )

            for object, idx in cloned_objects_to_cast:
                cast_object = from_openfermion(object)
                value[idx] = cast_object

            return value[0] if is_not_originally_seq else seq_type(value)

        return wrapper_decorator

    return adapter
