
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mpmath import zetazero
from scipy.spatial.distance import euclidean

# Define parameters
N = 1000
L = 10.0
x = np.linspace(-L, L, N)
dx = x[1] - x[0]
num_levels = 100

# Laplacian (kinetic energy operator)
main_diag = -2.0 * np.ones(N)
off_diag = np.ones(N - 1)
laplacian = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csr') / dx**2

# Reference Riemann zeta zeros
zeta_zeros = [zetazero(n).imag for n in range(1, num_levels + 1)]
zeta_array = np.array(zeta_zeros)
zeta_ref = (zeta_array - np.mean(zeta_array)) / np.std(zeta_array)

# Define a test chaotic potential: V(x) = x^2 + 1.5*sin(7x)
a, b = 1.5, 7.0
V_test = sp.diags(x**2 + a * np.sin(b * x), 0, format='csr')
H_test = -laplacian + V_test

# Compute eigenvalues
eigvals, _ = spla.eigsh(H_test, k=num_levels, which='SM')
norm_eigvals = (eigvals - np.mean(eigvals)) / np.std(eigvals)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(norm_eigvals, label='Chaotic Quantum System Eigenvalues', marker='o', linestyle='None', alpha=0.6)
plt.plot(zeta_ref, label='Zeta Zeros (Imaginary Parts)', marker='x', linestyle='None', alpha=0.6)
plt.title("Chaotic Quantum System vs. Riemann Zeta Zeros")
plt.xlabel("Index")
plt.ylabel("Normalized Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute and print Euclidean distance
error = euclidean(norm_eigvals, zeta_ref)
print(f"Spectral distance (Euclidean): {error}")
