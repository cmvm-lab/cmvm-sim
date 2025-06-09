"""
CMVM Cell – SU(2) Bosonic Matrix‑Model Prototype
================================================
This *starter script* implements the minimal workflow we discussed:
  1.  Build the truncated bosonic Hilbert space for an SU(2) matrix model.
  2.  Add the gauge‑penalty term so only singlet states survive.
  3.  Diagonalise the Hamiltonian and record the ground‑state energy.
  4.  Sweep the Fock‑space cutoff to show exponential convergence.

The code is intentionally compact and commented so you can quickly
augment it with extra fields, higher N, or CMVM‑specific interactions.

Dependencies
------------
  * numpy  ≥ 1.21
  * qutip  ≥ 5.0  (conda/pip install qutip)
  * matplotlib for the convergence plot

Usage
-----
Run directly with Python ≥ 3.9:
    python cmvm_cell_su2_matrix_model.py
It will print a small table of ground‑state energies vs cutoff and pop up
a convergence plot.

Next Steps / TODO
-----------------
* Replace the toy “harmonic‑oscillator” mass term with the full
  BFSS‑style interaction ½ Tr P² + ¼ g² Tr [X_i,X_j]².
* Introduce additional spatial matrices (d = 3, 9, …) and fermions.
* Port the same workflow into a QuTiP‑powered Jupyter notebook for richer
  visualisation or export the Hamiltonian to a VQE circuit.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def build_fock_basis(n_cut: int) -> list[qt.Qobj]:
    """Return creation/annihilation operators *a, a†* for a single bosonic mode
    truncated at *n_cut* quanta.  QuTiP labels: destroy(n), create(n).
    """
    a = qt.destroy(n_cut)
    adag = a.dag()
    return [a, adag]


def su2_structure_constants() -> dict[str, np.ndarray]:
    """Return SU(2) structure constants *f^{abc}* (totally antisymmetric) as a
    Python dictionary keyed by (a,b,c) tuples.  For SU(2) we only have one
    non‑zero value up to permutations: f^{123} = 1.
    """
    f = np.zeros((3, 3, 3))
    f[0, 1, 2] = f[1, 2, 0] = f[2, 0, 1] = 1.0
    f[0, 2, 1] = f[2, 1, 0] = f[1, 0, 2] = -1.0
    return {"f": f}


def gauge_generators(a_ops: list[qt.Qobj], adag_ops: list[qt.Qobj]) -> list[qt.Qobj]:
    """Construct SU(2) gauge generators G^a = i ε^{abc} a†_b a_c in the
    truncated Fock space.  Here we treat each colour index as an *independent*
    oscillator.  For clarity we use three separate modes ⇒ Hilbert space dims
    = n_cut³, manageable for n_cut ≲ 6 on a laptop.
    """
    f = su2_structure_constants()["f"]
    G = []
    for a in range(3):
        G_a = 0
        for b in range(3):
            for c in range(3):
                coeff = 1j * f[a, b, c]
                G_a += coeff * adag_ops[b] * a_ops[c]
        G.append(G_a)
    return G


def su2_hamiltonian(n_cut: int, omega: float = 1.0, c_pen: float = 10.0) -> qt.Qobj:
    """Toy SU(2) bosonic Hamiltonian:
         H = ω Σ_i a†_i a_i  +  c_pen Σ_a G_a²

    ▸ First term: simple harmonic oscillator mass (placeholder).
    ▸ Second term: gauge‑penalty that projects onto singlet sector as
      c_pen → ∞ (we choose a large but finite value).
    """
    # Build three independent oscillators (colour index = 0,1,2)
    a_ops = []
    adag_ops = []
    for _ in range(3):
        a, adag = build_fock_basis(n_cut)
        a_ops.append(a)
        adag_ops.append(adag)

    # Tensor‑product Hilbert space: colour 0 ⊗ colour 1 ⊗ colour 2
    # QuTiP uses * for tensor product.
    a_ops = [qt.tensor(*([a_ops[i] if i == col else qt.qeye(n_cut) for col in range(3)])) for i in range(3)]
    adag_ops = [op.dag() for op in a_ops]

    # Harmonic term
    h_ho = omega * sum(adag_ops[i] * a_ops[i] for i in range(3))

    # Gauge penalty term
    G_list = gauge_generators(a_ops, adag_ops)
    h_penalty = c_pen * sum(G * G for G in G_list)

    return h_ho + h_penalty


# -----------------------------------------------------------------------------
#  Main: sweep cutoff and diagonalise
# -----------------------------------------------------------------------------

def sweep_ground_state(max_cut: int = 10):
    cuts = np.arange(2, max_cut + 1, 2)  # 2,4,6,…
    energies = []

    for n in cuts:
        H = su2_hamiltonian(n_cut=n)
        # Compute lowest eigenvalue using QuTiP's *eigenstates* (dense for small dims)
        e0 = H.eigenenergies(eigvals=1)[0]
        energies.append(e0.real)
        print(f"Cutoff n={n:2d} →  E0 = {e0.real:.6f}")

    # Plot convergence
    plt.figure()
    plt.plot(cuts, energies, marker="o")
    plt.title("Ground‑state energy vs Fock cutoff (toy SU(2) model)")
    plt.xlabel("n_cut (quanta per colour)")
    plt.ylabel("E0 [ω units]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sweep_ground_state(max_cut=10)
