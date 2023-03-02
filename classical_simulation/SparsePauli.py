import numpy as np
from scipy.sparse import coo_matrix, kron

def SparsePauli(L):
    Szloc = all_pauli_z_matrices(L)
    Sxloc = all_pauli_x_matrices(L)
    Syloc = all_pauli_y_matrices(L)
    return Sxloc, Syloc, Szloc

def pauli_z(n, i):
    """
    Returns the Pauli Z matrix on the i-th qubit of n qubits.
    """
    Z = coo_matrix([[1, 0], [0, -1]], dtype=np.complex128)
    I = coo_matrix([[1, 0], [0, 1]], dtype=np.complex128)
    qubit_ops = [I] * n
    qubit_ops[i] = Z
    result = qubit_ops[0]
    for i in range(1, len(qubit_ops)):
        result = kron(result, qubit_ops[i])
    return result

def all_pauli_z_matrices(n):
    """
    Returns a list of all Pauli Z matrices on n qubits.
    """
    return [pauli_z(n, i) for i in range(n)]

def pauli_x(n, i):
    """
    Returns the Pauli X matrix on the i-th qubit of n qubits.
    """
    X = coo_matrix([[0, 1], [1, 0]], dtype=np.complex128)
    I = coo_matrix([[1, 0], [0, 1]], dtype=np.complex128)
    qubit_ops = [I] * n
    qubit_ops[i] = X
    result = qubit_ops[0]
    for i in range(1, len(qubit_ops)):
        result = kron(result, qubit_ops[i])
    return result

def all_pauli_x_matrices(n):
    """
    Returns a list of all Pauli X matrices on n qubits.
    """
    return [pauli_x(n, i) for i in range(n)]

def pauli_y(n, i):
    """
    Returns the Pauli Y matrix on the i-th qubit of n qubits.
    """
    Y = coo_matrix([[0, -1j], [1j, 0]], dtype=np.complex128)
    I = coo_matrix([[1, 0], [0, 1]], dtype=np.complex128)
    qubit_ops = [I] * n
    qubit_ops[i] = Y
    result = qubit_ops[0]
    for i in range(1, len(qubit_ops)):
        result = kron(result, qubit_ops[i])
    return result

def all_pauli_y_matrices(n):
    """
    Returns a list of all Pauli Y matrices on n qubits.
    """
    return [pauli_y(n, i) for i in range(n)]