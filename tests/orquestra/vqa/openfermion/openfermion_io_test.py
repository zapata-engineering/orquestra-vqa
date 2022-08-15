################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import os
import unittest

import numpy as np
from openfermion import (
    FermionOperator,
    InteractionRDM,
    get_interaction_operator,
    hermitian_conjugated,
)
from orquestra.quantum.utils import convert_dict_to_array

from orquestra.vqa.openfermion._openfermion_io import (
    convert_dict_to_interaction_op,
    convert_dict_to_interaction_rdm,
    convert_interaction_op_to_dict,
    convert_interaction_rdm_to_dict,
    load_interaction_operator,
    load_interaction_rdm,
    save_interaction_operator,
    save_interaction_rdm,
)


class TestQubitOperator(unittest.TestCase):
    def setUp(self):
        n_modes = 2
        np.random.seed(0)
        one_body_tensor = np.random.rand(*(n_modes,) * 2)
        two_body_tensor = np.random.rand(*(n_modes,) * 4)
        self.interaction_rdm = InteractionRDM(one_body_tensor, two_body_tensor)

    def test_interaction_op_to_dict_io(self):
        # Given
        test_op = FermionOperator("1^ 2^ 3 4")
        test_op += hermitian_conjugated(test_op)
        interaction_op = get_interaction_operator(test_op)
        interaction_op.constant = 0.0

        # When
        interaction_op_dict = convert_interaction_op_to_dict(interaction_op)
        recreated_interaction_op = convert_dict_to_interaction_op(interaction_op_dict)

        # Then
        self.assertEqual(recreated_interaction_op, interaction_op)

    def test_interaction_operator_io(self):
        # Given
        test_op = FermionOperator("1^ 2^ 3 4")
        test_op += hermitian_conjugated(test_op)
        interaction_op = get_interaction_operator(test_op)
        interaction_op.constant = 0.0

        # When
        save_interaction_operator(interaction_op, "interaction_op.json")
        loaded_op = load_interaction_operator("interaction_op.json")

        # Then
        self.assertEqual(interaction_op, loaded_op)
        os.remove("interaction_op.json")

    def test_interaction_rdm_io(self):
        # Given

        # When
        save_interaction_rdm(self.interaction_rdm, "interaction_rdm.json")
        loaded_interaction_rdm = load_interaction_rdm("interaction_rdm.json")

        # Then
        self.assertEqual(self.interaction_rdm, loaded_interaction_rdm)
        os.remove("interaction_rdm.json")

    def test_convert_interaction_rdm_to_dict(self):
        rdm_dict = convert_interaction_rdm_to_dict(self.interaction_rdm)

        self.assertTrue(
            np.allclose(
                convert_dict_to_array(rdm_dict["one_body_tensor"]),
                self.interaction_rdm.one_body_tensor,
            )
        )
        self.assertTrue(
            np.allclose(
                convert_dict_to_array(rdm_dict["two_body_tensor"]),
                self.interaction_rdm.two_body_tensor,
            )
        )

    def test_convert_dict_to_interaction_rdm(self):
        rdm_dict = convert_interaction_rdm_to_dict(self.interaction_rdm)
        converted_interaction_rdm = convert_dict_to_interaction_rdm(rdm_dict)

        self.assertEqual(self.interaction_rdm, converted_interaction_rdm)
