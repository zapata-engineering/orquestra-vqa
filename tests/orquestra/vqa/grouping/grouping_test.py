################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import math

import numpy as np
import pytest
from orquestra.quantum.measurements import ExpectationValues
from orquestra.quantum.openfermion import InteractionRDM, QubitOperator

from orquestra.vqa.grouping import (
    compute_group_variances,
    group_comeasureable_terms_greedy,
    is_comeasureable,
)

h2_hamiltonian = QubitOperator(
    """-0.0420789769629383 [] +
-0.04475014401986127 [X0 X1 Y2 Y3] +
0.04475014401986127 [X0 Y1 Y2 X3] +
0.04475014401986127 [Y0 X1 X2 Y3] +
-0.04475014401986127 [Y0 Y1 X2 X3] +
0.17771287459806312 [Z0] +
0.1705973832722407 [Z0 Z1] +
0.12293305054268083 [Z0 Z2] +
0.1676831945625421 [Z0 Z3] +
0.17771287459806312 [Z1] +
0.1676831945625421 [Z1 Z2] +
0.12293305054268083 [Z1 Z3] +
-0.24274280496459985 [Z2] +
0.17627640802761105 [Z2 Z3] +
-0.24274280496459985 [Z3]"""
)

rdm1 = np.array(
    [
        [0.98904311, 0.0, 0.0, 0.0],
        [0.0, 0.98904311, 0.0, 0.0],
        [0.0, 0.0, 0.01095689, 0.0],
        [0.0, 0.0, 0.0, 0.01095689],
    ]
)

rdm2 = np.array(
    [
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, -0.98904311, 0.0, -0.0],
                [0.98904311, 0.0, 0.0, 0.0],
                [0.0, -0.0, 0.0, 0.10410015],
                [0.0, 0.0, -0.10410015, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, -0.0, 0.0, -0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, -0.0, 0.0, -0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
        [
            [
                [0.0, 0.98904311, 0.0, 0.0],
                [-0.98904311, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, -0.10410015],
                [-0.0, 0.0, 0.10410015, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [-0.0, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-0.0, 0.0, -0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, -0.0, 0.0, -0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, -0.0, 0.0, -0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.10410015, 0.0, -0.0],
                [-0.10410015, 0.0, 0.0, 0.0],
                [0.0, -0.0, 0.0, -0.01095689],
                [0.0, 0.0, 0.01095689, 0.0],
            ],
        ],
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [-0.0, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-0.0, 0.0, -0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, -0.10410015, 0.0, 0.0],
                [0.10410015, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, 0.01095689],
                [-0.0, 0.0, -0.01095689, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
    ]
)

rdms = InteractionRDM(rdm1, rdm2)


@pytest.mark.parametrize(
    "term1,term2,expected_result",
    [
        (((0, "Y"),), ((0, "X"),), False),
        (((0, "Y"),), ((1, "X"),), True),
        (((0, "Y"), (1, "X")), (), True),
    ],
)
def test_is_comeasureable(term1, term2, expected_result):
    assert is_comeasureable(term1, term2) == expected_result


@pytest.mark.parametrize(
    "qubit_operator,expected_groups",
    [
        (
            QubitOperator("2[Z0 Z1] + [X0 X1] + [Z0] + [X0]"),
            [QubitOperator("2[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")],
        ),
        (
            QubitOperator("[Z0 Z1] + [X0 X1] + [Z0] + []"),
            [
                QubitOperator("[Z0 Z1] + [Z0]"),
                QubitOperator("[X0 X1]"),
                QubitOperator("[]"),
            ],
        ),
        (
            QubitOperator("-3[]"),
            [QubitOperator("-3[]")],
        ),
    ],
)
def test_group_comeasureable_terms_greedy(qubit_operator, expected_groups):
    groups = group_comeasureable_terms_greedy(qubit_operator)
    assert groups == expected_groups


@pytest.mark.parametrize(
    "qubit_operator,sort_terms,expected_groups",
    [
        (
            QubitOperator("[Z0 Z1] + [X0 X1] + [Z0] + [X0]"),
            True,
            [QubitOperator("[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")],
        ),
        (
            QubitOperator("[X0] + 2 [X0 Y1] + 3 [X0 Z1]"),
            True,
            [QubitOperator("[X0] + 3 [X0 Z1]"), QubitOperator("2 [X0 Y1]")],
        ),
        (
            QubitOperator("[X0] + 2 [X0 Y1] + 3 [X0 Z1]"),
            False,
            [QubitOperator("[X0] + 2 [X0 Y1]"), QubitOperator("3 [X0 Z1]")],
        ),
    ],
)
def test_group_comeasureable_terms_greedy_sorted(
    qubit_operator, sort_terms, expected_groups
):
    groups = group_comeasureable_terms_greedy(qubit_operator, sort_terms=sort_terms)
    assert groups == expected_groups


@pytest.mark.parametrize(
    "groups, expecval, variances",
    [
        (
            [QubitOperator("[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")],
            None,
            np.array([2.0, 2.0]),
        ),
        (
            [
                QubitOperator("[Z0 Z1] + [Z0] + []"),
                QubitOperator("[X0 X1] + [X0] + 5 []"),
            ],
            None,
            np.array([2.0, 2.0]),
        ),
        (
            [QubitOperator("[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")],
            ExpectationValues(np.zeros(4)),
            np.array([2.0, 2.0]),
        ),
        (
            [QubitOperator("2 [Z0 Z1] + 3 [Z0]"), QubitOperator("[X0 X1] + [X0]")],
            ExpectationValues(np.zeros(4)),
            np.array([13.0, 2.0]),
        ),
        (
            [QubitOperator("2[]")],
            ExpectationValues(np.array([1])),
            np.array([0.0]),
        ),
        (
            [
                QubitOperator("2 [Z0 Z1] + 3 [Z0] + 8[]"),
                QubitOperator("[X0 X1] + [X0] + []"),
            ],
            ExpectationValues(np.asarray([0.0, 0.0, 1.0, 0, 0, 1.0])),
            np.array([13.0, 2.0]),
        ),
    ],
)
def test_compute_group_variances_with_ref(groups, expecval, variances):
    test_variances = compute_group_variances(groups, expecval)
    np.testing.assert_allclose(test_variances, variances)


@pytest.mark.parametrize(
    "groups, expecval, variances",
    [
        (
            [QubitOperator("[Z0 Z1] + [Z0]"), QubitOperator("[X0 X1] + [X0]")],
            ExpectationValues(np.array([-1.5, 0, 0, 0])),
            np.array([2.0, 2.0]),
        ),
        (
            [QubitOperator("2 [Z0 Z1] + 3 [Z0]"), QubitOperator("[X0 X1] + [X0]")],
            ExpectationValues(np.array([1.5, 0, 0, 0])),
            np.array([13.0, 2.0]),
        ),
        (
            [QubitOperator("2 [Z0 Z1] + 3 [Z0]"), QubitOperator("[X0 X1] + [X0]")],
            ExpectationValues(np.array([1.5, 0, 0, 0])),
            np.array([13.0, 2.0]),
        ),
    ],
)
def test_compute_group_variances_fails_for_invalid_refs(groups, expecval, variances):
    with pytest.raises(ValueError):
        _ = compute_group_variances(groups, expecval)


@pytest.mark.parametrize(
    "groups, expecval",
    [
        (
            group_comeasureable_terms_greedy(h2_hamiltonian, False),
            ExpectationValues(np.ones(15)),
        ),
        (
            group_comeasureable_terms_greedy(h2_hamiltonian, False),
            ExpectationValues(np.zeros(15)),
        ),
        (
            group_comeasureable_terms_greedy(h2_hamiltonian, False),
            ExpectationValues(np.repeat(0.5, 15)),
        ),
    ],
)
def test_compute_group_variances_without_ref(groups, expecval):
    test_variances = compute_group_variances(groups, expecval)
    test_ham_variance = np.sum(test_variances)
    # Assemble H and compute its variances independently
    ham = QubitOperator()
    for g in groups:
        ham += g
    ham_coeff = np.array(list(ham.terms.values()))
    pauli_var = 1.0 - expecval.values**2
    ref_ham_variance = np.sum(ham_coeff**2 * pauli_var)
    assert math.isclose(
        test_ham_variance, ref_ham_variance
    )  # this is true as long as the groups do not overlap
