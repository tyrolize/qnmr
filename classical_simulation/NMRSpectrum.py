import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from SparsePauli import SparsePauli

def NMRSpectrum(k):
    # Compute the NMR spectrum for molecule number k in the dataset
    # Input: k, the label of the molecule 
    # Output: -frequency w and spectral weight A

    # Get quantum operators
    N = 4
    DH = 2**N
    Sx, Sy, Sz = SparsePauli(N)

    # Load spin matrix
    name = f"../Molecules/matrix{k}.csv"
    J = np.loadtxt(name, delimiter=',')
    h = np.diag(J)
    h = h - np.mean(h)  # Get chemical shifts and remove the mean
    wext = 40  # External magnetic field strength in Mhz
    h = h * wext  # Convert from ppm to Mhz
    J = J - np.diag(np.diag(J))  # Remove diagonal

    # Generate Hamiltonian
    H = np.zeros((DH, DH), dtype=np.complex128)
    n = np.zeros((DH, DH), dtype=np.complex128)
    X = np.zeros((DH, DH), dtype=np.complex128)
    for i in range(N-1):
        for j in range(i+1, N):
            H += J[i, j]/4 * (np.kron(Sz[i], Sz[j]) + np.kron(Sx[i], Sx[j]) + np.kron(Sy[i], Sy[j]))
    for i in range(N):
        n += h[i] * Sz[i]
        X += Sx[i]
    H = H + n

    # Diagonalize and construct spectrum
    eigvals, eigvecs = eigsh(H, k=DH-1, which='SA')
    xe = eigvecs.conj().T @ X @ eigvecs
    de = np.abs(eigvals[:, np.newaxis] - eigvals)

    xr = np.array([], dtype=np.float64)
    dr = np.array([], dtype=np.float64)
    for i in range(1, 2**N):
        xr = np.hstack((xr, np.diag(xe, i), np.diag(xe, -i)))
        dr = np.hstack((dr, np.diag(de, i), np.diag(de, -i)))
    BW = np.max(dr)  # Bandwidth

    gamma = 1  # Decoherence rate in Hz
    w = np.linspace(-1.2*BW, 1.2*BW, 10000)
    f = gamma / (gamma**2 + (w[:, np.newaxis] - dr[np.newaxis, :])**2) / np.pi / DH
    A = f @ (xr**2)  # Spectral function

    return A
