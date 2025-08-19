# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
#
# Module: spintransport.physics.st
# Brief : Core numerical operators (derivatives, Gaussian packets) and Crank–Nicolson time stepper.
# Project: spintransport-suite
# Authors: Jorge Luis Beltran Diaz and Leovildo Diago Cisneros
"""
Numerical kernels used by the simulator (finite differences, evolution operator, solver).
"""


from __future__ import annotations

import numpy as np
from scipy import linalg as spla
from typing import overload


# ===========================
# Atomic units and converters
# ===========================
#: ħ in atomic units
HBAR_AU: float = 1.0

#: eV -> Hartree (atomic units of energy)
EV_TO_AU: float = 1.0 / 27.2114

#: Å -> Bohr (atomic units of length)
ANGSTROM_TO_BOHR: float = 1.89

#: fs -> atomic units of time
FS_TO_AU: float = 41.35


@overload
def eV(x: float) -> float: ...
@overload
def eV(x: np.ndarray) -> np.ndarray: ...
def eV(x):  # type: ignore[override]
    return x * EV_TO_AU


@overload
def angstrom(x: float) -> float: ...
@overload
def angstrom(x: np.ndarray) -> np.ndarray: ...
def angstrom(x):  # type: ignore[override]
    return x * ANGSTROM_TO_BOHR


@overload
def femtosecond(x: float) -> float: ...
@overload
def femtosecond(x: np.ndarray) -> np.ndarray: ...
def femtosecond(x):  # type: ignore[override]
    return x * FS_TO_AU


# ------------------------------------------
# Complex Absorbing Potential (CAP) helpers
# ------------------------------------------
def _cap_profile(ny: int, Lcap_A: float, dy_A: float, where: str = "both") -> np.ndarray:
    w = np.zeros(ny, float)
    ncap = int(max(1, round(Lcap_A / max(dy_A, 1e-30))))
    if ncap <= 0:
        return w
    if where in ("left", "both"):
        w[:ncap] = np.linspace(0.0, 1.0, ncap, endpoint=True)
    if where in ("right", "both"):
        ramp = np.linspace(0.0, 1.0, ncap, endpoint=True)
        w[-ncap:] = np.maximum(w[-ncap:], ramp[::-1])
    return w

def cap_potential_matrix(
    ny: int,
    dy_A: float,
    Lcap_A: float,
    eta_eV: float = 0.5,
    order: int = 3,
    where: str = "both"
) -> np.ndarray:
    """
    H_CAP (4ny×4ny) = -i*diag(W), W(y)= eV(eta_eV)*[w(y)]^order.
    Se replica en los 4 bloques diagonales.
    """
    if Lcap_A <= 0.0 or eta_eV <= 0.0:
        return np.zeros((4*ny, 4*ny), dtype=complex)
    w = _cap_profile(ny, Lcap_A, dy_A, where=where)
    w = w ** max(1, int(order))
    W_au = eV(float(eta_eV)) * w  # usa tu conversor eV->a.u.
    D = -1j * np.diag(W_au.astype(complex))
    Hcap = np.zeros((4*ny, 4*ny), dtype=complex)
    for b in range(4):
        Hcap[b*ny:(b+1)*ny, b*ny:(b+1)*ny] = D
    return Hcap







# ====================================
# Finite-difference derivative matrices
# ====================================
def partial_derivative_operator(ny: int, dy: float, order: int) -> np.ndarray:
    """
    Build the (ny × ny) finite-difference operator for the 1st or 2nd derivative
    with *central* stencils on a uniform grid.

    Notes
    -----
    - First derivative (central):    d/dy f ≈ [f(i+1) - f(i-1)] / (2 dy)
      We return  **-i (E1 - E-1) / dy**, i.e. *twice* the Hermitian `k_y` operator.
      Therefore, a proper Hermitian k_y is `0.5 * partial_derivative_operator(..., 1)`.
    - Second derivative (central):   d²/dy² f ≈ [f(i+1) - 2 f(i) + f(i-1)] / dy²
      We return **-(E1 - 2 I + E-1) / dy²**, i.e. +k_y² in the usual convention.

    Parameters
    ----------
    ny : int
        Number of grid points.
    dy : float
        Grid spacing (in atomic units).
    order : int
        1 for first derivative-like operator; 2 for second derivative operator.

    Returns
    -------
    np.ndarray (complex)
        The square operator matrix.

    Raises
    ------
    ValueError
        If `order` is not 1 or 2.
    """
    if order == 1:
        # -i (E1 - E-1) / dy
        return -1j * (np.eye(ny, k=1, dtype=complex) - np.eye(ny, k=-1, dtype=complex)) / dy

    if order == 2:
        # -(E1 - 2I + E-1) / dy^2
        return -(
            np.eye(ny, k=1, dtype=complex)
            - 2.0 * np.eye(ny, dtype=complex)
            + np.eye(ny, k=-1, dtype=complex)
        ) / (dy ** 2)

    raise ValueError("order must be 1 or 2")


# =========================
# Wave-packet construction
# =========================
def gaussian_packet(y: np.ndarray, y0: float, sigma: float, k: np.ndarray | float, conj: bool = False) -> np.ndarray:
    """
    Build a complex Gaussian wave packet ψ(y) with plane-wave factor exp(i k (y - y0)).

    Parameters
    ----------
    y : np.ndarray
        1D coordinate array (integer grid indices or physical grid — your choice).
    y0 : float
        Center position (same units as `y`).
    sigma : float
        Gaussian width (same units as `y`).
    k : np.ndarray | float
        Wavenumber. Can be scalar or vector (broadcasted along columns).
    conj : bool, optional
        If True, multiply the packet by i (useful to build orthogonal spinor components).

    Returns
    -------
    np.ndarray
        Array with shape (len(y), n_k) if `k` is vector, or (len(y), 1) if scalar.
        Complex dtype.
    """
    phase = np.exp(1j * np.multiply.outer(y - y0, np.atleast_1d(k)))
    env = np.exp(-((y - y0) ** 2) / (2.0 * sigma ** 2))[:, None]
    packet = env * phase
    return (1j * packet) if conj else packet


# ====================================
# Crank–Nicolson time-evolution (LU)
# ====================================
def build_crank_nicolson_system(hamiltonian: np.ndarray, dt_au: float) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Precompute Crank–Nicolson matrices for the standard Schrödinger sign convention:

        i ∂ψ/∂t = H ψ
        ⇒ (I + i Δt H / 2) ψ_{n+1} = (I − i Δt H / 2) ψ_n

    We therefore build:
        left  = I + i Δt H / 2
        right = I − i Δt H / 2

    and return the LU factorization of `left` for fast solves.

    Parameters
    ----------
    hamiltonian : np.ndarray
        Square (N×N) Hermitian Hamiltonian.
    dt_au : float
        Time step in atomic units.

    Returns
    -------
    left, right, (lu, piv)
        `left` and `right` matrices, and SciPy's LU factorization of `left`.
    """
    n = hamiltonian.shape[0]
    I = np.eye(n, dtype=complex)
    c = 1j * (dt_au * 0.5)  # i Δt / 2

    left  = I + c * hamiltonian
    right = I - c * hamiltonian

    lu, piv = spla.lu_factor(left)
    return left, right, (lu, piv)



def evolve_crank_nicolson(lu: tuple[np.ndarray, np.ndarray],
                          right: np.ndarray,
                          psi0: np.ndarray,
                          n_time_steps: int,
                          show_progress: bool = True) -> np.ndarray:
    N, nE = psi0.shape
    psi = np.zeros((N, n_time_steps, nE), dtype=complex)
    psi[:, 0, :] = psi0

    lu_mat, piv = lu

    # --- Dirichlet BC en extremos ---
    def _enforce_dirichlet(t_idx: int) -> None:
        psi[0, t_idx, :] = 0.0
        psi[-1, t_idx, :] = 0.0

    for t in range(1, n_time_steps):
        rhs = right @ psi[:, t - 1, :]
        psi[:, t, :] = spla.lu_solve((lu_mat, piv), rhs)

        # aplicar Dirichlet
        _enforce_dirichlet(t)

        if show_progress:
            frac = t / (n_time_steps - 1 if n_time_steps > 1 else 1)
            bar = int(50 * frac)
            print(f"\rTime-stepping: [{'#' * bar}{'-' * (50 - bar)}] {int(100 * frac)}%", end="")

    if show_progress:
        print("\nDone.")
    return psi

