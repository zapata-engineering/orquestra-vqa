import numpy as np

from forestopenfermion import exponentiate
from zquantum.core.circuit import Circuit
from openfermion.utils import uccsd_singlet_generator
from openfermion.ops import QubitOperator
from openfermion import jordan_wigner, bravyi_kitaev

from pyquil.gates import X
from pyquil.quil import Program
from pyquil.paulis import exponentiate as pyquil_exponentiate

def exponentiate_fermion_operator(fermion_generator, transformation='Jordan-Wigner'):
    """
    Create a pyQuil circuit corresponding to the exponentiation of an operator.

    Args:
        fermion_generator (openfermion.FermionOperator or 
            openfermion.InteractionOperator): fermionic generator.
        fermion_transform (str): The name of the qubit to fermion transformation
            to use.

    Returns:
        pyquil.quil.Program: Circuit corresponding to the exponentiation of the
            transformed operator. 
    """
    if transformation not in ['Jordan-Wigner', 'Bravyi-Kitaev']:
        raise RuntimeError(f'Unrecognized transformation {transformation}')

    # Transform generator to qubits
    if transformation == 'Jordan-Wigner':
        transformation = jordan_wigner
    elif transformation == 'Bravyi-Kitaev':
        transformation = bravyi_kitaev

    qubit_generator = transformation(fermion_generator)

    for term in qubit_generator.terms:
        qubit_generator.terms[term] = float(qubit_generator.terms[term].imag)
    qubit_generator.compress()

    # Quantum circuit implementing the excitation operators
    circuit = exponentiate(qubit_generator)

    return circuit

def build_singlet_uccsd_circuit(parameters, n_mo, n_electrons, transformation):
    """Constructs the circuit for a singlet UCCSD ansatz with HF initial state.

    Args:
        parameters (numpy.ndarray): Vector of coupled-cluster amplitudes
        n_mo (int): number of molecular orbitals
        n_electrons (int): number of electrons

    Returns:
        qprogram (pyquil.quil.Program): a program for simulating the ansatz
    """
    qprogram = Program()

    # Set initial state with correct number of electrons
    for i in range(n_electrons):
        qubit_index = n_electrons-i-1
        qprogram += X(qubit_index)

    # Build UCCSD generator
    fermion_generator = uccsd_singlet_generator(parameters,
                                                2*n_mo,
                                                n_electrons,
                                                anti_hermitian=True)
    evolution_operator = exponentiate_fermion_operator(fermion_generator,
                                                       transformation)

    qprogram += evolution_operator
    return Circuit(qprogram)
