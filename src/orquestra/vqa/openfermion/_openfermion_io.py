################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from typing import Any, Dict

import numpy as np
import rapidjson as json
from openfermion import InteractionOperator, InteractionRDM
from orquestra.quantum.typing import AnyPath, LoadSource
from orquestra.quantum.utils import convert_array_to_dict, convert_dict_to_array


def convert_interaction_op_to_dict(op: InteractionOperator) -> Dict[str, Any]:
    """Convert an InteractionOperator to a dictionary.
    Args:
        op: the operator
    Returns:
        dictionary: the dictionary representation
    """

    dictionary: Dict[str, Any] = {}
    dictionary["constant"] = convert_array_to_dict(np.array(op.constant))
    dictionary["one_body_tensor"] = convert_array_to_dict(np.array(op.one_body_tensor))
    dictionary["two_body_tensor"] = convert_array_to_dict(np.array(op.two_body_tensor))

    return dictionary


def convert_dict_to_interaction_op(dictionary: dict) -> InteractionOperator:
    """Get an InteractionOperator from a dictionary.
    Args:
        dictionary: the dictionary representation
    Returns:
        op: the operator
    """

    # The tolist method is used to convert the constant from a zero-dimensional array to
    # a float/complex
    constant = convert_dict_to_array(dictionary["constant"]).tolist()

    one_body_tensor = convert_dict_to_array(dictionary["one_body_tensor"])
    two_body_tensor = convert_dict_to_array(dictionary["two_body_tensor"])

    return InteractionOperator(constant, one_body_tensor, two_body_tensor)


def load_interaction_operator(file: LoadSource) -> InteractionOperator:
    """Load an interaction operator object from a file.
    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        op: the operator.
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return convert_dict_to_interaction_op(data)


def save_interaction_operator(
    interaction_operator: InteractionOperator, filename: AnyPath
) -> None:
    """Save an interaction operator to file.
    Args:
        interaction_operator: the operator to be saved
        filename: the name of the file
    """

    with open(filename, "w") as f:
        f.write(
            json.dumps(convert_interaction_op_to_dict(interaction_operator), indent=2)
        )


def convert_interaction_rdm_to_dict(op):
    """Convert an InteractionRDM to a dictionary.
    Args:
        op (openfermion.ops.InteractionRDM): the operator
    Returns:
        dictionary (dict): the dictionary representation
    """

    dictionary = {}
    dictionary["one_body_tensor"] = convert_array_to_dict(np.array(op.one_body_tensor))
    dictionary["two_body_tensor"] = convert_array_to_dict(np.array(op.two_body_tensor))

    return dictionary


def convert_dict_to_interaction_rdm(dictionary):
    """Get an InteractionRDM from a dictionary.
    Args:
        dictionary (dict): the dictionary representation
    Returns:
        op (openfermion.ops.InteractionRDM): the operator
    """

    one_body_tensor = convert_dict_to_array(dictionary["one_body_tensor"])
    two_body_tensor = convert_dict_to_array(dictionary["two_body_tensor"])

    return InteractionRDM(one_body_tensor, two_body_tensor)


def load_interaction_rdm(file: LoadSource) -> InteractionRDM:
    """Load an interaction RDM object from a file.
    Args:
        file: a file-like object to load the interaction RDM from.

    Returns:
        The interaction RDM.
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return convert_dict_to_interaction_rdm(data)


def save_interaction_rdm(interaction_rdm: InteractionRDM, filename: AnyPath) -> None:
    """Save an interaction operator to file.
    Args:
        interaction_operator: the operator to be saved
        filename: the name of the file
    """

    with open(filename, "w") as f:
        f.write(json.dumps(convert_interaction_rdm_to_dict(interaction_rdm), indent=2))
